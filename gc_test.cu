#include <iostream>
#include <fstream>
#include <random>
#include <algorithm>
#include <cstring>
#include <chrono>
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

__global__ void calculateFitnessKernel(int numOfVertices, int* adjacencyMatrix, Specimen* population, int populationSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= populationSize) return;

    Specimen specimen = population[idx];
    specimen.numOfConflicts = 0;
    bool* colorSet = new bool[numOfVertices];
    memset(colorSet, 0, numOfVertices * sizeof(bool));

    for (int i = 0; i < numOfVertices; i++) {
        for (int j = i + 1; j < numOfVertices; j++) {
            if (adjacencyMatrix[i * numOfVertices + j] == 1 && specimen.colors[i] == specimen.colors[j]) {
                specimen.numOfConflicts++;
            }
        }
        colorSet[specimen.colors[i]] = true;
    }

    while (specimen.numOfConflicts == 0) {
        int maxColor = numOfVertices - 1;
        while (maxColor >= 0 && !colorSet[maxColor]) {
            maxColor--;
        }

        bool allColorsPresent = true;
        for (int i = 0; i <= maxColor; i++) {
            if (!colorSet[i]) {
                allColorsPresent = false;
                int missingColor = i;

                for (int j = 0; j < numOfVertices; j++) {
                    if (specimen.colors[j] > missingColor) {
                        specimen.colors[j]--;
                    }
                }

                colorSet[missingColor] = true;
                colorSet[maxColor] = false;
                break;
            }
        }

        if (allColorsPresent) {
            break;
        }
    }

    specimen.numOfColors = 0;
    for (int i = 0; i < numOfVertices; i++) {
        if (colorSet[i]) {
            specimen.numOfColors++;
        }
    }

    population[idx] = specimen;
    delete[] colorSet;
}


void calculateFitness(int numOfVertices, int** adjacencyMatrix, Specimen &specimen) // sprawdzanie jakości rozwiązania
{
    specimen.numOfConflicts = 0;
    bool* colorSet = new bool[numOfVertices]; // użyte kolory
    std::memset(colorSet, 0, numOfVertices * sizeof(bool));

    for (int i = 0; i < numOfVertices; i++)
    {
        for (int j = i + 1; j < numOfVertices; j++) // sprawdzamy tylko od wierzchołka do konca zakresu, by uniknąć powtórzeń
        {
            if (adjacencyMatrix[i][j] == 1 && specimen.colors[i] == specimen.colors[j]) // szukamy wierzchołka sąsiedniego z tym samym kolorem
            {
                specimen.numOfConflicts++;
            }
        }
        colorSet[specimen.colors[i]] = true; // kolor użyty
    }

    while (specimen.numOfConflicts == 0) { //Poprawiamy tylko dla rozwiązań bezkonfliktowych
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
                    if (specimen.colors[j] > missingColor) {
                        specimen.colors[j]--;
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

    specimen.numOfColors = 0; // ustawiamy liczbę kolorów
    for (int i = 0; i < numOfVertices; i++) {
        if (colorSet[i]) {
            specimen.numOfColors++;
        }
    }

    delete[] colorSet; // zwalniamy pamięć
    return;
}

int randomNumber(int min, int max) { // zwraca losową liczbę naturalną z zakresu [min;max]
    random_device rd;
    mt19937 rng(rd());
    uniform_int_distribution<int> uni(min, max - 1);
    return uni(rng);
}

int rng(int min, int max) { // if cuda breaks
    return min + rand() % ((max + 1) - min);
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
    for (int i = 0; i < populationSize; i++)
        if (population[i].numOfColors != numOfColors)
            for (int j = 0; j < numOfVertices; j++)
                if (population[i].colors[j] >= numOfColors - 1)
                    population[i].colors[j] = randomNumber(0, numOfColors - 2);
}

void initializePopulation(Specimen* population, int populationSize, int numOfVertices) {
    for (int i = 0; i < populationSize; ++i) {
        population[i].colors = (int*)malloc(numOfVertices * sizeof(int));
        for (int j = 0; j < numOfVertices; ++j) {
            population[i].colors[j] = randomNumber(0, numOfVertices - 1);
        }
    }
}

#define inFile "queen7_7.txt" // Plik wejściowy
#define outFile "result_gc_seq.txt"
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
    //const int random_vertices = numOfVertices/10; // alt
    int** adjacencyMatrix = new int* [numOfVertices];
    for (int i = 0; i < numOfVertices; i++)
        adjacencyMatrix[i] = new int[numOfVertices] {}; // macierz adjacencji
    int a, b; // para wierzchołków
    while (sourceFile >> a >> b) { // wczutujemy dane z pliku
        adjacencyMatrix[a - 1][b - 1] = 1;
        adjacencyMatrix[b - 1][a - 1] = 1;
    }
    sourceFile.close();
    int* vertexColor = new int [numOfVertices] {}; // kolory wierzchołków
    bool* colorsUsed = new bool [numOfVertices] {}; // pomocnicza

    Specimen* population = (Specimen*)malloc(populationSize * sizeof(Specimen)); // przydzielamy pamięć dla populacji
    initializePopulation(population, populationSize, numOfVertices);

    /*for (int i = 0; i < populationSize; i++) {
        calculateFitness(numOfVertices, adjacencyMatrix, population[i]); // sprawdzanie jakości rozwiązania naiwnego
    }*/
    // Allocate GPU memory
    Specimen* d_population;
    cudaMalloc(&d_population, populationSize * sizeof(Specimen));
    for (int i = 0; i < populationSize; i++) {
        cudaMalloc(&population[i].colors, numOfVertices * sizeof(int));
        cudaMemcpy(d_population[i].colors, population[i].colors, numOfVertices * sizeof(int), cudaMemcpyHostToDevice);
    }

    int* d_adjacencyMatrix;
    cudaMalloc(&d_adjacencyMatrix, numOfVertices * numOfVertices * sizeof(int));
    for (int i = 0; i < numOfVertices; i++) {
        cudaMemcpy(d_adjacencyMatrix + i * numOfVertices, adjacencyMatrix[i], numOfVertices * sizeof(int), cudaMemcpyHostToDevice);
    }

    cudaMemcpy(d_population, population, populationSize * sizeof(Specimen), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (populationSize + blockSize - 1) / blockSize;
    calculateFitnessKernel<<<numBlocks, blockSize>>>(numOfVertices, d_adjacencyMatrix, d_population, populationSize);

    cudaMemcpy(population, d_population, populationSize * sizeof(Specimen), cudaMemcpyDeviceToHost);
    for (int i = 0; i < populationSize; i++) {
        cudaMemcpy(population[i].colors, d_population[i].colors, numOfVertices * sizeof(int), cudaMemcpyDeviceToHost);
    }

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
        for (int i = 0; i < populationSize; i++) {
            if (randomNumber(0, 100) < mutationChance) {
                mutateOld(newPopulation[i], numOfColors, numOfVertices); //mutacja przez zmienienie osobnika 
            }
        }

        /*for (int i = 0; i < populationSize; i++) {
            calculateFitness(numOfVertices, adjacencyMatrix, newPopulation[i]); // sprawdzanie jakości rozwiązania
        }*/
        // Transfer new population to GPU
        cudaMemcpy(d_population, newPopulation, populationSize * sizeof(Specimen), cudaMemcpyHostToDevice);
        calculateFitnessKernel<<<numBlocks, blockSize>>>(numOfVertices, d_adjacencyMatrix, d_population, populationSize);
        cudaMemcpy(newPopulation, d_population, populationSize * sizeof(Specimen), cudaMemcpyDeviceToHost);
        for (int i = 0; i < populationSize; i++) {
            cudaMemcpy(newPopulation[i].colors, d_population[i].colors, numOfVertices * sizeof(int), cudaMemcpyDeviceToHost);
        }

        for (int i = 0; i < populationSize; ++i) { // kopiujemy nową populację w miejsce starej
            memcpy(population[i].colors, newPopulation[i].colors, numOfVertices * sizeof(int));
            population[i].numOfColors = newPopulation[i].numOfColors;
            population[i].numOfConflicts = newPopulation[i].numOfConflicts;
        }

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
    delete[] population;
}