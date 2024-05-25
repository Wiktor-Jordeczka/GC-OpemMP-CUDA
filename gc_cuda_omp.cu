#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <set>
#include <cstring>
#include <chrono>
#include <omp.h>
#include <cuda_runtime.h>
#include "gc_cuda_omp.h"
using namespace std;

// Wartości domyślne zdefiniowane w headerze, można zmienić wywołując z parametrami

extern string inFile; // -i (path); plik wejściowy
extern string outFile; // -o (path); plik wyjściowy
extern int populationSize; // -p (int); rozmiar populacji
extern int iterations; // -l (int); maksymalna liczba iteracji (generacji)
extern int mutationChance; // -m (int); prawdopodobieństwo mutacji <0;100>
extern int stopTime; // -s (int); maksymalny czas działania w sekundach
extern bool verbose; // -v (int); czy wypisywać printy i dodatkowe dane wyjściowe
extern int printInterval; // -v (int); co ile generacji wykonać print
extern unsigned int seed; // -r (int); seed do generatora liczb losowych

// struktura osobnika
typedef struct Specimen {
    int* colors; // pointer do przechowywania kolorów
    int numOfColors = 0;
    int numOfConflicts = 0;

    // pomoc do sortowania
    bool operator< (const Specimen& other) const {
        if (numOfConflicts != other.numOfConflicts)
            return this->numOfConflicts < other.numOfConflicts;
        return this->numOfColors < other.numOfColors;
    }
} Specimen;

// Kernel CUDA - każdy wątek oblicza jakość innego osobnika
__global__ void cudaCalculateFitnessKernel(int numOfVertices, int* d_adjacencyMatrix, int* d_colors, int* d_colorNum, int* d_conflictNum, int populationSize) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x; // thread (specimen) number
    if (idx >= populationSize) return; // sprawdzamy czy nie jesteśmy poza populacją (ilość wątków to wielokrotność 256)
    d_conflictNum[idx] = 0; // inicjujemy na 0
    bool* colorSet = new bool[numOfVertices]; // użyte kolory
    std::memset(colorSet, 0, numOfVertices * sizeof(bool)); // inicjujemy jako false

    for (int i = 0; i < numOfVertices; i++)
    {
        for (int j = i + 1; j < numOfVertices; j++) // sprawdzamy tylko od wierzchołka do konca zakresu, by uniknąć powtórzeń
        {
            if (d_adjacencyMatrix[i * numOfVertices + j] == 1 && d_colors[idx * numOfVertices + i] == d_colors[idx * numOfVertices + j]) // szukamy wierzchołka sąsiedniego z tym samym kolorem
            {
                d_conflictNum[idx]++;
            }
        }
        colorSet[d_colors[idx * numOfVertices + i]] = true; // kolor użyty
    }
    
    // poprawiamy kolory, aby były kolejnymi liczbami naturalnymi od 0
    while (d_conflictNum[idx] == 0) { // Poprawiamy tylko dla rozwiązań bezkonfliktowych
        int maxColor = numOfVertices - 1;
        while (maxColor >= 0 && !colorSet[maxColor]) {
            maxColor--;
        }

        bool allColorsPresent = true; // sprawdzamy czy brakuje jakiegoś koloru
        for (int i = 0; i <= maxColor; i++) {
            if (!colorSet[i]) { // poprawiamy kolory
                allColorsPresent = false;
                int missingColor = i;

                // Obniżamy kolory o 1
                for (int j = 0; j < numOfVertices; j++) {
                    if (d_colors[idx * numOfVertices + j] > missingColor) {
                        d_colors[idx * numOfVertices + j]--;
                    }
                }

                colorSet[missingColor] = true;
                colorSet[maxColor] = false;
                break; // Restart sprawdzania
            }
        }
        if (allColorsPresent) { // wszystko ok
            break;
        }
    }

    d_colorNum[idx] = 0; // ustawiamy liczbę kolorów
    for (int i = 0; i < numOfVertices; i++) {
        if (colorSet[i]) {
            d_colorNum[idx]++;
        }
    }

    delete[] colorSet; // zwalniamy pamięć
    return;
}

// obliczanie jakości osobników w populacji
void cudaCalculateFitness(int numOfVertices, int* d_adjacencyMatrix, Specimen* population, int populationSize) {
    // spłaszczona tablica struktów Specimen
    int* h_colors = new int[populationSize * numOfVertices]; // spłaszczone tablice kolorów osobników
    int* h_colorNum = new int[populationSize]; // spłaszczone ilości kolorów osobników
    int* h_conflictNum = new int[populationSize]; // spłaszczone ilości konfliktów osobników

    // spłaszczamy tablicę struktów i podtablice kolorów
    #pragma omp parallel for default(none) shared(populationSize, numOfVertices, h_colors, population)
    for (int i = 0; i < populationSize; i++) {
        for (int j = 0; j < numOfVertices; j++) {
            h_colors[(i * numOfVertices + j)] = population[i].colors[j];
        }
    }

    int* d_colors; // macierz kolorów GPU
    size_t colorsSize = populationSize * numOfVertices * sizeof(int);
    cudaMalloc(&d_colors, colorsSize);
    cudaMemcpy(d_colors, h_colors, colorsSize, cudaMemcpyHostToDevice);
    int* d_colorNum; // wektor il. koloró GPU
    int* d_conflictNum; // wektor il. konfliktów GPU
    size_t numVecSize = populationSize * sizeof(int); // rozmiar wektorów
    // wartości zostaną obliczone na nowo, więc wystarczy alokacja bez kopiowania
    cudaMalloc(&d_colorNum, numVecSize);
    cudaMalloc(&d_conflictNum, numVecSize);

    int blockSize = 256; // rozmiar bloku
    int numBlocks = (populationSize + blockSize - 1) / blockSize; // ilość bloków
    // uruchamiamy kernel
    cudaCalculateFitnessKernel<<<numBlocks, blockSize>>>(numOfVertices, d_adjacencyMatrix, d_colors, d_colorNum, d_conflictNum, populationSize);
    // pobieramy wynikowe dane z GPU
    cudaMemcpy(h_colorNum, d_colorNum, numVecSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_conflictNum, d_conflictNum, numVecSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_colors, d_colors, colorsSize, cudaMemcpyDeviceToHost);

    // kopiujemy spłaszczone dane do populacji
    #pragma omp parallel for default(none) shared(populationSize, numOfVertices, h_colors, population)
    for (int i = 0; i < populationSize; i++) {
        for (int j = 0; j < numOfVertices; j++) {
            population[i].colors[j] = h_colors[(i * numOfVertices + j)];
        }
        population[i].numOfColors = h_colorNum[i];
        population[i].numOfConflicts = h_conflictNum[i];
    }
    // zwalniamy pamięć
    delete[] h_colors;
    delete[] h_colorNum;
    delete[] h_conflictNum;
    // zwalniamy pamięć GPU
    cudaFree(d_colors);
    cudaFree(d_colorNum);
    cudaFree(d_conflictNum);
}

// zwraca losową liczbę naturalną z zakresu [min;max], korzystając z podanego generatora
int randomNumber(int min, int max, mt19937& rng) {
    uniform_int_distribution<int> uni(min, max - 1);
    return uni(rng);
}

// turniej
Specimen tournamentSelection(Specimen* population, int populationSize, mt19937& rng) {
    int tournamentSize = 3;
    Specimen chosenSpecimen = population[randomNumber(0, populationSize - 1, rng)];
    for (int i = 1; i < tournamentSize; ++i) {
        Specimen candidate = population[randomNumber(0, populationSize - 1, rng)];
        if (candidate < chosenSpecimen) {
            chosenSpecimen = candidate;
        }
    }
    return chosenSpecimen;
}

// Krzyżowanie osobników - jednopunktowe
void crossover(Specimen& parent1, Specimen& parent2, Specimen& offspring1, Specimen& offspring2, int numOfVertices, mt19937& rng) {
    int pivot = randomNumber(0, numOfVertices - 1, rng);
    #pragma omp parallel for default(none) shared(pivot, parent1, parent2, offspring1, offspring2, numOfVertices, rng)
    for (int i = 0; i < numOfVertices; ++i) {
        if (i <= pivot) {
            offspring1.colors[i] = parent1.colors[i];
            offspring2.colors[i] = parent2.colors[i];
        }
        else {
            offspring1.colors[i] = parent2.colors[i];
            offspring2.colors[i] = parent1.colors[i];
        }
    }
}

// mutacja losowego wierzchołka
void mutateOld(Specimen& s, int numOfColors, int numOfVertices, mt19937& rng) {
    s.colors[randomNumber(0, numOfVertices, rng)] = randomNumber(0, numOfColors, rng); 
}

// Sprawdzamy akutalną liczbę kolorów najlepszego rozwiązania i usuwamy ich potencjalne nadwyżki u innych osobników
void correct(Specimen* population, int numOfColors, int numOfVertices, int populationSize, mt19937& rng){
    #pragma omp parallel for default(none) shared(populationSize, numOfVertices, numOfColors, population, rng)
    for (int i = 0; i < populationSize; i++)
        if (population[i].numOfColors != numOfColors)
            for (int j = 0; j < numOfVertices; j++)
                if (population[i].colors[j] >= numOfColors - 1) // zastępujemy losowym kolorem
                    population[i].colors[j] = randomNumber(0, numOfColors - 2, rng);
}

// inicjalizacja populacji
void initializePopulation(Specimen* population, int populationSize, int numOfVertices, mt19937& rng) {
    #pragma omp parallel for default(none) shared(populationSize, numOfVertices, population, rng)
    for (int i = 0; i < populationSize; ++i) {
        population[i].colors = (int*)malloc(numOfVertices * sizeof(int));
        for (int j = 0; j < numOfVertices; ++j) {
            population[i].colors[j] = randomNumber(0, numOfVertices - 1, rng);
        }
    }
}

int main(int argc, char *argv[]){
    // odczytanie i ustawienie parametrów
    int i = 0;
    while (i < argc) {
        if (argv[i][0] == '-'){ // opcje 
            switch(argv[i][1]){
                case 'v':
                    verbose = true;
                    printInterval = atoi(argv[++i]);
                    break;
                case 'i':
                    inFile = argv[++i];
                    break;
                case 'o':
                    outFile = argv[++i];
                    break;
                case 'p':
                    populationSize = atoi(argv[++i]);
                    break;
                case 'l':
                    iterations = atoi(argv[++i]);
                    break;
                case 'm':
                    mutationChance = atoi(argv[++i]);
                    break;
                case 's':
                    stopTime = atoi(argv[++i]);
                    break;
                case 'r':
                    seed = atoi(argv[++i]);
                    break;
                default:
                    cout<<"unknown option! "<<argv[i]<<endl;
                    break;
            }
        }
        i++; 
    }

    // dopełnienie do parzystej
    if (populationSize % 2)
        populationSize++;

    mt19937 rng(seed); // generator do losowania liczb
    int numOfVertices; // ilość wierzchołków
    ifstream sourceFile(inFile); // plik wejściowy
    
    // wczytujemy graf z pliku
    sourceFile >> numOfVertices;
    int** adjacencyMatrix = new int* [numOfVertices]; // macierz sąsiedztwa
    #pragma omp parallel for default(none) shared(adjacencyMatrix, numOfVertices)
    for (int i = 0; i < numOfVertices; i++)
        adjacencyMatrix[i] = new int[numOfVertices] {}; // macierz sąsiedztwa
    int a, b; // para wierzchołków
    while (sourceFile >> a >> b) { // wczutujemy dane z pliku
        adjacencyMatrix[a - 1][b - 1] = 1;
        adjacencyMatrix[b - 1][a - 1] = 1;
    }
    sourceFile.close();

    // spłaszczamy macierz dla cuda
    int* h_adjacencyMatrix = new int[numOfVertices * numOfVertices]; // spłaszczona macierz sąsiedztwa
    #pragma omp parallel for default(none) shared(adjacencyMatrix, numOfVertices, h_adjacencyMatrix)
    for (int i = 0; i < numOfVertices; i++) {
        for (int j = 0; j < numOfVertices; j++) {
            h_adjacencyMatrix[(i * numOfVertices + j)] = adjacencyMatrix[i][j];
        }
    }
    // kopiujemy na GPU
    int* d_adjacencyMatrix; // macierz sąsiedztwa GPU
    size_t adjMatSize = numOfVertices * numOfVertices * sizeof(int); // rozmiar macierzy sąsiedztwa
    cudaMalloc(&d_adjacencyMatrix, adjMatSize); // alokacja pamięci na GPU
    cudaMemcpy(d_adjacencyMatrix, h_adjacencyMatrix, adjMatSize, cudaMemcpyHostToDevice); // kopiowanie do GPU
    delete[] h_adjacencyMatrix; // zwalniamy pamięć

    // przydzielamy pamięć dla populacji
    Specimen* population = (Specimen*)malloc(populationSize * sizeof(Specimen));
    initializePopulation(population, populationSize, numOfVertices, rng);

    cudaCalculateFitness(numOfVertices, d_adjacencyMatrix, population, populationSize); // sprawdzanie jakości rozwiązania naiwnego
    sort(population, population+populationSize); // sortowanie
    Specimen solution = Specimen(population[0]); // zmienna przetrzymująca optymalne rozwiązanie

    int iteration = 0; // iteracja
    int numOfColors = 0; // ilość kolorów jaką chcemy uzyskać
    auto start = chrono::steady_clock::now(); // Timer
    auto stop = chrono::steady_clock::now(); // zatrzymanie po czasie
    Specimen* newPopulation = (Specimen*)malloc(populationSize * sizeof(Specimen)); // Nowa populacja
    initializePopulation(newPopulation, populationSize, numOfVertices, rng); // przydzielamy pamięć

    // główna pętla algorytmu
    while (iteration < iterations && chrono::duration_cast<chrono::seconds>(stop-start).count() < stopTime) {

        //Sprawdzamy akutalną liczbę kolorów najlepszego rozwiązania i usuwamy ich potencjalne nadwyżki u innych osobników
        numOfColors = solution.numOfColors - 1;
        correct(population, numOfColors, numOfVertices, populationSize, rng); // korekta kolorów

        if (verbose && iteration % printInterval == 0) // print
            cout << endl << "Generacja: " << iteration << " Aktualnie poszukuje rozwiazania dla: " << numOfColors << " kolorow, aktualna liczba konfliktow: " << population[0].numOfConflicts;

        // Turnieje i krzyżowanie (wybór nowej populacji)
        #pragma omp parallel for default(none) shared(populationSize, numOfVertices, population, newPopulation, rng)
        for (int i = 0; i < populationSize; i+=2) {
            Specimen parent1 = tournamentSelection(population, populationSize, rng);
            Specimen parent2 = tournamentSelection(population, populationSize, rng);
            Specimen offspring1;
            offspring1.colors = (int*)malloc(numOfVertices * sizeof(int));
            Specimen offspring2;
            offspring2.colors = (int*)malloc(numOfVertices * sizeof(int));
            crossover(parent1, parent2, offspring1, offspring2, numOfVertices, rng);
            newPopulation[i] = offspring1;
            newPopulation[i + 1] = offspring2;
        }

        // Mutacje
        #pragma omp parallel for default(none) shared(populationSize, numOfVertices, numOfColors, mutationChance, newPopulation, rng)
        for (int i = 0; i < populationSize; i++) {
            if (randomNumber(0, 100, rng) < mutationChance) {
                mutateOld(newPopulation[i], numOfColors, numOfVertices, rng);
            }
        }

        // obliczamy jakość osobników w populacji na GPU
        cudaCalculateFitness(numOfVertices, d_adjacencyMatrix, newPopulation, populationSize);

        // kopiujemy nową populację w miejsce starej
        #pragma omp parallel for default(none) shared(populationSize, numOfVertices, population, newPopulation)
        for (int i = 0; i < populationSize; ++i) {
            memcpy(population[i].colors, newPopulation[i].colors, numOfVertices * sizeof(int));
            population[i].numOfColors = newPopulation[i].numOfColors;
            population[i].numOfConflicts = newPopulation[i].numOfConflicts;
        }
        
        sort(population, population+populationSize); // sortujemy tablicę tak, że najlepsze rozwiązanie ma indeks 0, a najgorsze 'populacja - 1'

        // zapamiętujemy rozwiązanie, gdy znajdziemy kolorowanie bezkonfliktowe z mniejszą ilością kolorów
        if (solution.numOfColors > population[0].numOfColors && population[0].numOfConflicts == 0)
            solution = Specimen(population[0]); 
        iteration++;
        stop = chrono::steady_clock::now();
    }

    // zwalnianie pamięci populacji tymczasowej
    #pragma omp parallel for default(none) shared(populationSize, newPopulation)
    for (int i = 0; i < populationSize; ++i) { 
        free(newPopulation[i].colors);
    }
    free(newPopulation);

    // Zapisanie wyniku do pliku
    fstream output(outFile, ios::out);
    output << "Czas w sekundach: " << chrono::duration_cast<chrono::seconds>(stop-start).count() << endl;
    output << "Osobnik w zmiennej solution: " << " konfliktow: " << solution.numOfConflicts << " kolorow: " << solution.numOfColors << endl;
    for (int j = 0; j < numOfVertices; j++) {
        output << solution.colors[j] << " ";
    }
    if(verbose){
        output << endl << endl;
        output << "Osobniki w ostatniej iteracji, posortowane rosnąco wg ilości konfliktów: " <<endl;
        for (int i = 0; i < populationSize; i++) {
            output << "Osobnik " << i << " konfliktow: " << population[i].numOfConflicts << " kolorow: " << population[i].numOfColors << endl;
            for (int j = 0; j < numOfVertices; j++) {
                output << population[i].colors[j] << " ";
            }
            output << endl;
        }
    }

    // zwalniamy pamięć populacji i macierzy
    delete[] population;
    #pragma omp parallel for default(none) shared(adjacencyMatrix, numOfVertices)
    for (int i = 0; i < numOfVertices; i++)
        delete[] adjacencyMatrix[i];
    delete[] adjacencyMatrix;
}