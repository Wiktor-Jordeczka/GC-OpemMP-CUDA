#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <set>
#include <cstring>
#include <chrono>
#include <omp.h>
#include "gc_omp.h"
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

// obliczanie jakości osobnika
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

    // poprawiamy kolory, aby były kolejnymi liczbami naturalnymi od 0
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
    auto start = chrono::steady_clock::now(); // Timer
    auto stop = chrono::steady_clock::now();

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
    for (int i = 0; i < numOfVertices; i++)
        adjacencyMatrix[i] = new int[numOfVertices] {}; // macierz sąsiedztwa
    int a, b; // para wierzchołków
    while (sourceFile >> a >> b) { // wczutujemy dane z pliku
        adjacencyMatrix[a - 1][b - 1] = 1;
        adjacencyMatrix[b - 1][a - 1] = 1;
    }
    sourceFile.close();

    // przydzielamy pamięć dla populacji
    Specimen* population = (Specimen*)malloc(populationSize * sizeof(Specimen));
    initializePopulation(population, populationSize, numOfVertices, rng);

    #pragma omp parallel for default(none) shared(populationSize, numOfVertices, adjacencyMatrix, population)
    for (int i = 0; i < populationSize; i++) { // sprawdzanie jakości rozwiązania naiwnego
        calculateFitness(numOfVertices, adjacencyMatrix, population[i]); 
    }
    sort(population, population+populationSize); // sortowanie
    Specimen solution = Specimen(population[0]); // zmienna przetrzymująca optymalne rozwiązanie

    int iteration = 0; // iteracja
    int numOfColors = 0; // ilość kolorów jaką chcemy uzyskać
    Specimen* newPopulation = (Specimen*)malloc(populationSize * sizeof(Specimen)); // Nowa populacja
    initializePopulation(newPopulation, populationSize, numOfVertices, rng); // przydzielamy pamięć

    // główna pętla algorytmu
    while (iteration < iterations && chrono::duration_cast<chrono::seconds>(stop-start).count() < stopTime) { // zatrzymanie po czasie

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

        // obliczamy jakość osobników w populacji
        #pragma omp parallel for default(none) shared(populationSize, numOfVertices, adjacencyMatrix, newPopulation)
        for (int i = 0; i < populationSize; i++) {
            calculateFitness(numOfVertices, adjacencyMatrix, newPopulation[i]); // sprawdzanie jakości rozwiązania
        }

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
    stop = chrono::steady_clock::now(); // koniec mierzenia czasu

    // Zapisanie wyniku do pliku
    fstream output(outFile, ios::out);
    output << "Czas w nanosekundach: " << (stop-start).count() << endl;
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