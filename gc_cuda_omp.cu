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
using namespace std;

typedef struct Specimen {
    int* colors; // pointer do przechowywania kolorów
    int numOfColors = 0;
    int numOfConflicts = 0;

    bool operator< (const Specimen& other) const { // pomoc do sortowania
        if (numOfConflicts != other.numOfConflicts)
            return this->numOfConflicts < other.numOfConflicts;
        return this->numOfColors < other.numOfColors;
    }
} Specimen;

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

int randomNumber(int min, int max) { // zwraca losową liczbę naturalną z zakresu [min;max]
    random_device rd;
    mt19937 rng(rd());
    uniform_int_distribution<int> uni(min, max - 1);
    return uni(rng);
}

Specimen tournamentSelection(Specimen* population, int populationSize) { // turniej
    int tournamentSize = 3;
    Specimen chosenSpecimen = population[randomNumber(0, populationSize - 1)];
    for (int i = 1; i < tournamentSize; ++i) {
        Specimen candidate = population[randomNumber(0, populationSize - 1)];
        if (candidate < chosenSpecimen) {
            chosenSpecimen = candidate;
        }
    }
    return chosenSpecimen;
}

// Krzyżowanie
void crossover(Specimen& parent1, Specimen& parent2, Specimen& offspring1, Specimen& offspring2, int numOfVertices) {
    int pivot = randomNumber(0, numOfVertices - 1);
    #pragma omp parallel for default(none) shared(pivot, parent1, parent2, offspring1, offspring2, numOfVertices)
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

void mutateOld(Specimen& s, int numOfColors, int numOfVertices) {  // Mutacja, modyfikacja starego osobnika
    s.colors[randomNumber(0, numOfVertices)] = randomNumber(0, numOfColors); // mutacja losowego wierzchołka
}

void correct(Specimen* population, int numOfColors, int numOfVertices, int populationSize)
{ //Sprawdzamy akutalną liczbę kolorów najlepszego rozwiązania i usuwamy ich potencjalne nadwyżki u innych osobników
    #pragma omp parallel for default(none) shared(populationSize, numOfVertices, numOfColors, population)
    for (int i = 0; i < populationSize; i++)
        if (population[i].numOfColors != numOfColors)
            for (int j = 0; j < numOfVertices; j++)
                if (population[i].colors[j] >= numOfColors - 1)
                    population[i].colors[j] = randomNumber(0, numOfColors - 2);
}

void initializePopulation(Specimen* population, int populationSize, int numOfVertices) {
    #pragma omp parallel for default(none) shared(populationSize, numOfVertices, population)
    for (int i = 0; i < populationSize; ++i) {
        population[i].colors = (int*)malloc(numOfVertices * sizeof(int));
        for (int j = 0; j < numOfVertices; ++j) {
            population[i].colors[j] = randomNumber(0, numOfVertices - 1);
        }
    }
}

#define inFile "queen7_7.txt" // Plik wejściowy
#define outFile "result_gc_cuda_omp.txt"
#define printInterval 100 // co ile generacji wykonać print

int main()
{
    const int populationSize = 1000; // ustawienie całkowitej populacji
    const int random_vertices = 30; // ilość losowo pokolorowanych wierzchołków przy tworzeniu populacji
    const int iterations = 1000; // Maksymalna liczba iteracji (generacji)
    int mutationChance = 50; // Tutaj wpisujemy prawdopodobieństwo mutacji <0;100>
    int stopTime = 60; // Maksymalny czas działania
    
    random_device random_generator; // generator do losowania liczb
    int numOfVertices; // ilość wierzchołków
    ifstream sourceFile(inFile); // plik wejściowy
    sourceFile >> numOfVertices;
    int** adjacencyMatrix = new int* [numOfVertices];
    #pragma omp parallel for default(none) shared(adjacencyMatrix, numOfVertices)
    for (int i = 0; i < numOfVertices; i++)
        adjacencyMatrix[i] = new int[numOfVertices] {}; // macierz adjacencji
    int a, b; // para wierzchołków
    while (sourceFile >> a >> b) { // wczutujemy dane z pliku
        adjacencyMatrix[a - 1][b - 1] = 1;
        adjacencyMatrix[b - 1][a - 1] = 1;
    }
    sourceFile.close();

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

    Specimen* population = (Specimen*)malloc(populationSize * sizeof(Specimen)); // przydzielamy pamięć dla populacji
    initializePopulation(population, populationSize, numOfVertices);

    cudaCalculateFitness(numOfVertices, d_adjacencyMatrix, population, populationSize); // sprawdzanie jakości rozwiązania naiwnego
    sort(population, population+populationSize); // sortowanie
    Specimen solution = Specimen(population[0]); // zmienna przetrzymująca optymalne rozwiązanie

    // PĘTLA 
    int iteration = 0;
    int numOfColors = 0;
    auto start = chrono::steady_clock::now(); // Timer
    auto stop = chrono::steady_clock::now(); // zatrzymanie po czasie
    while (iteration < iterations && chrono::duration_cast<chrono::seconds>(stop-start).count() < stopTime) {
        //Sprawdzamy akutalną liczbę kolorów najlepszego rozwiązania i usuwamy ich potencjalne nadwyżki u innych osobników
        numOfColors = solution.numOfColors - 1;
        correct(population, numOfColors, numOfVertices, populationSize); // korekta kolorów

        if (iteration % printInterval == 0) // print
            cout << endl << "Generacja: " << iteration << " Aktualnie poszukuje rozwiazania dla: " << numOfColors << " kolorow, aktualna liczba konfliktow: " << population[0].numOfConflicts;

        Specimen* newPopulation = (Specimen*)malloc(populationSize * sizeof(Specimen)); // Nowa populacja
        initializePopulation(newPopulation, populationSize, numOfVertices); // przydzielamy pamięć

        // Turnieje i krzyżowanie (wybór nowej populacji)
        #pragma omp parallel for default(none) shared(populationSize, numOfVertices, population, newPopulation)
        for (int i = 0; i < populationSize; i+=2) {
            Specimen parent1 = tournamentSelection(population, populationSize);
            Specimen parent2 = tournamentSelection(population, populationSize);
            Specimen offspring1;
            offspring1.colors = (int*)malloc(numOfVertices * sizeof(int));
            Specimen offspring2;
            offspring2.colors = (int*)malloc(numOfVertices * sizeof(int));
            crossover(parent1, parent2, offspring1, offspring2, numOfVertices);
            newPopulation[i] = offspring1;
            newPopulation[i + 1] = offspring2;
        }

        // MUTACJE
        #pragma omp parallel for default(none) shared(populationSize, numOfVertices, numOfColors, mutationChance, newPopulation)
        for (int i = 0; i < populationSize; i++) {
            if (randomNumber(0, 100) < mutationChance) {
                mutateOld(newPopulation[i], numOfColors, numOfVertices); //mutacja przez zmienienie osobnika 
            }
        }

        cudaCalculateFitness(numOfVertices, d_adjacencyMatrix, newPopulation, populationSize);

        #pragma omp parallel for default(none) shared(populationSize, numOfVertices, population, newPopulation)
        for (int i = 0; i < populationSize; ++i) { // kopiujemy nową populację w miejsce starej
            memcpy(population[i].colors, newPopulation[i].colors, numOfVertices * sizeof(int));
            population[i].numOfColors = newPopulation[i].numOfColors;
            population[i].numOfConflicts = newPopulation[i].numOfConflicts;
        }

        #pragma omp parallel for default(none) shared(populationSize, newPopulation)
        for (int i = 0; i < populationSize; ++i) { // zwalnianie pamięci
            free(newPopulation[i].colors);
        }
        free(newPopulation);
        
        sort(population, population+populationSize); // sortujemy tablicę tak, że najlepsze rozwiązanie ma indeks 0, a najgorsze 'populacja - 1'
        if (solution.numOfColors > population[0].numOfColors && population[0].numOfConflicts == 0)
            solution = Specimen(population[0]); // zapamiętujemy rozwiązanie, gdy znajdziemy rozwiązanie z mniejszą ilością kolorów
        iteration++;
        stop = chrono::steady_clock::now();
    }

    // Wypisanie osobników na końcu
    fstream output(outFile, ios::out);
    output << "osobnik w zmiennej solution: " << " konfliktow: " << solution.numOfConflicts << " kolorow: " << solution.numOfColors << endl;
    for (int j = 0; j < numOfVertices; j++) {
        output << solution.colors[j] << " ";
    }
    output << endl << endl;
    for (int i = 0; i < populationSize; i++) {
        output << "Osobnik " << i << " konfliktow: " << population[i].numOfConflicts << " kolorow: " << population[i].numOfColors << endl;
        for (int j = 0; j < numOfVertices; j++) {
            output << population[i].colors[j] << " ";
        }
        output << endl;
    }
    cout << endl << "Czas w sekuncach: " << chrono::duration_cast<chrono::seconds>(stop-start).count() << endl;
    // zwalniamy pamięć
    delete[] population;
    #pragma omp parallel for default(none) shared(adjacencyMatrix, numOfVertices)
    for (int i = 0; i < numOfVertices; i++)
        delete[] adjacencyMatrix[i];
    delete[] adjacencyMatrix;
}