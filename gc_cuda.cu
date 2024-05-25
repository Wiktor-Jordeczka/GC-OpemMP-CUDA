#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <set>
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

__global__ void cudaCalculateFitnessKernel(int numOfVertices, int* d_adjacencyMatrix, int* d_colors, int* d_colorNum, int* d_conflictNum, int populationSize) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x; // thread (specimen) number
    //printf("Kernel! %d", idx);
    if (idx >= populationSize) return; // sprawdzamy czy nie jesteśmy poza populacją (ilość wątków to wielokrotność 256)
    //else printf("Kernel! %d ", idx);

    /*Specimen &specimen = d_population[idx];
    specimen.numOfConflicts = 0;
    extern __shared__ bool colorSet[];*/ // Dynamic shared memory
    d_conflictNum[idx] = 0; // inicjujemy na 0
    bool* colorSet = new bool[numOfVertices]; // użyte kolory
    std::memset(colorSet, 0, numOfVertices * sizeof(bool)); // inicjujemy jako false

    for (int i = 0; i < numOfVertices; i++)
    {
        for (int j = i + 1; j < numOfVertices; j++) // sprawdzamy tylko od wierzchołka do konca zakresu, by uniknąć powtórzeń
        {
            if (d_adjacencyMatrix[i * numOfVertices + j] == 1 && d_colors[idx * numOfVertices + i] == d_colors[idx * numOfVertices + j]) // szukamy wierzchołka sąsiedniego z tym samym kolorem
            {
                //specimen.numOfConflicts++;
                d_conflictNum[idx]++;
            }
        }
        //colorSet[specimen.colors[i]] = true;
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

    /*for (int i = 0; i < numOfVertices; i++) {
        colorSet[i] = false;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < numOfVertices; i += blockDim.x) {
        for (int j = i + 1; j < numOfVertices; j++) {
            if (adjacencyMatrix[i * numOfVertices + j] == 1 && specimen.colors[i] == specimen.colors[j]) {
                atomicAdd(&specimen.numOfConflicts, 1);
            }
        }
        colorSet[specimen.colors[i]] = true;
    }
    __syncthreads();

    if (threadIdx.x == 0 && specimen.numOfConflicts == 0) {
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
                maxColor--;
                break;
            }
        }

        if (allColorsPresent) {
            int count = 0;
            for (int i = 0; i < numOfVertices; i++) {
                if (colorSet[i]) {
                    count++;
                }
            }
            specimen.numOfColors = count;
        }
    }*/
}

void cudaCalculateFitness(int numOfVertices, /*int* h_adjacencyMatrix*/int* d_adjacencyMatrix, Specimen* population, int populationSize) {
    //int* h_adjacencyMatrix = new int[numOfVertices * numOfVertices]; // spłaszczona macierz sąsiedztwa
    // spłaszczona tablica struktów Specimen
    int* h_colors = new int[populationSize * numOfVertices]; // spłaszczone tablice kolorów osobników
    int* h_colorNum = new int[populationSize]; // spłaszczone ilości kolorów osobników
    int* h_conflictNum = new int[populationSize]; // spłaszczone ilości konfliktów osobników

    // spłaszczamy macierz sąsiedztwa WYSTARCZY TO ZROBIĆ RAZ!!!!!!!!!!!!!
    //#pragma omp parallel for collapse(2)
    /*for (int i = 0; i < numOfVertices; i++) {
        for (int j = 0; j < numOfVertices; j++) {
            h_adjacencyMatrix[(i * numOfVertices + j)] = adjacencyMatrix[i][j];
            //cout << "pos " <<i * numOfVertices + j;
        }
    }*/

    // spłaszczamy tablicę struktów i podtablice kolorów
    //#pragma omp parallel for collapse(2)
    for (int i = 0; i < populationSize; i++) {
        for (int j = 0; j < numOfVertices; j++) {
            h_colors[(i * numOfVertices + j)] = population[i].colors[j];
            //cout << " " <<h_colors[(i * numOfVertices + j)];
        }
        //cout<<" new specimen "<<i<<endl;
    }
    
    // WSTARCZY RAZ!!!
    /*int* d_adjacencyMatrix; // macierz sąsiedztwa GPU
    size_t adjMatSize = numOfVertices * numOfVertices * sizeof(int); // rozmiar macierzy sąsiedztwa
    cudaMalloc(&d_adjacencyMatrix, adjMatSize); // alokacja pamięci na GPU
    cudaMemcpy(d_adjacencyMatrix, h_adjacencyMatrix, adjMatSize, cudaMemcpyHostToDevice); // kopiowanie do GPU*/

    /*for (int i = 0; i < numOfVertices; ++i) {
        for (int j = 0; j < numOfVertices; ++j) {
            cout<<adjacencyMatrix[i][j]<<" ";
        }
        cout<<endl;
    }*/

    /*for (int i = 0; i < numOfVertices; ++i) {
        for (int j = 0; j < numOfVertices; ++j) {
            cout<<h_adjacencyMatrix[(i * numOfVertices + j)]<<" ";
        }
        cout<<endl;
    }*/

    //cudaError_t my_cudaError; // do testowania błędów
    /*int* test_adjacencyMatrix = new int[numOfVertices * numOfVertices];
    cudaMemcpy(test_adjacencyMatrix, d_adjacencyMatrix, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < numOfVertices; ++i) {
        for (int j = 0; j < numOfVertices; ++j) {
            cout<<test_adjacencyMatrix[(i * numOfVertices + j)]<<" ";
        }
        cout<<endl;
    }*/

    //cout<<"copied!"<<endl;

    int* d_colors; // macierz kolorów GPU
    size_t colorsSize = populationSize * numOfVertices * sizeof(int);
    cudaMalloc(&d_colors, colorsSize);
    cudaMemcpy(d_colors, h_colors, colorsSize, cudaMemcpyHostToDevice);
    /*int* test_colors = new int[populationSize * numOfVertices];
    cudaMemcpy(test_colors, d_colors, colorsSize, cudaMemcpyDeviceToHost);
    for (int i = 0; i < populationSize; ++i) {
        for (int j = 0; j < numOfVertices; ++j) {
            cout<<test_colors[(i * numOfVertices + j)]<<" ";
        }
        cout<<" new specimen "<<i<<endl;
    }*/
    int* d_colorNum; // wektor il. koloró GPU
    int* d_conflictNum; // wektor il. konfliktów GPU
    size_t numVecSize = populationSize * sizeof(int); // rozmiar wektorów
    // wartości zostaną obliczone na nowo, więc wystarczy alokacja bez kopiowania
    cudaMalloc(&d_colorNum, numVecSize);
    cudaMalloc(&d_conflictNum, numVecSize);

    //cout<<"launching kernels!"<<endl;

    int blockSize = 256; // rozmiar bloku
    int numBlocks = (populationSize + blockSize - 1) / blockSize; // ilość bloków
    //int sharedMemSize = numOfVertices * sizeof(bool); // ???
    // uruchamiamy kernel
    cudaCalculateFitnessKernel<<<numBlocks, blockSize/*, sharedMemSize*/>>>(numOfVertices, d_adjacencyMatrix, d_colors, d_colorNum, d_conflictNum, populationSize);
    // cudaDeviceSynchronize(); // czekamy aż kernel skończy, raczej niepotrzebne, ponieważ cudaMemcpy wymusza synchronizację

    //cout<<"after kernels!"<<endl;
    // pobieramy wynikowe dane z GPU
    cudaMemcpy(h_colorNum, d_colorNum, numVecSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_conflictNum, d_conflictNum, numVecSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_colors, d_colors, colorsSize, cudaMemcpyDeviceToHost);

    //cout<<"after copy back!"<<endl;

    // kopiujemy spłaszczone dane do populacji
    //#pragma omp parallel for collapse(2)
    for (int i = 0; i < populationSize; i++) {
        for (int j = 0; j < numOfVertices; j++) {
            population[i].colors[j] = h_colors[(i * numOfVertices + j)];
            //cout << " " <<population[i].colors[j];
        }
        population[i].numOfColors = h_colorNum[i];
        population[i].numOfConflicts = h_conflictNum[i];
        //cout<<" colors "<<population[i].numOfColors<<endl;
        //cout<<" conflicts "<<population[i].numOfConflicts<<endl;
        //cout<<" new specimen "<<i<<endl;
    }

    /*Specimen* h_population = (Specimen*)malloc(populationSize * sizeof(Specimen)); // Nowa populacja
    initializePopulation(h_population, populationSize, numOfVertices); // przydzielamy pamięć*/

    // można ominąć?
    /*Specimen* d_population;
    cout<<"alloc1!"<<endl;
    // fails?
    cudaMalloc(&d_population, populationSize * sizeof(Specimen));
    cudaMemcpy(d_population, population, populationSize * sizeof(Specimen), cudaMemcpyHostToDevice);
    Specimen* test_population = (Specimen*)malloc(populationSize * sizeof(Specimen));
    my_cudaError = cudaMemcpy(test_population, d_population, populationSize * sizeof(Specimen), cudaMemcpyDeviceToHost);*/
    /*cout<<"cuda err "<<my_cudaError<<endl;
    cout<<" test pop!"<<endl;
    cout<<population[5].colors[0]<<" test pop!"<<endl;
    cout<<test_population[5].colors[0]<<" test pop!"<<endl;
    cout<<population[5].colors[1]<<" test pop!"<<endl;
    cout<<test_population[5].colors[1]<<" test pop!"<<endl;*/

    /*cout<<"alloc2!"<<endl;
    for (int i = 0; i < populationSize; i++) {
        my_cudaError = cudaMalloc(&d_population[i].colors, numOfVertices * sizeof(int));
        cout<<"cpy2!"<<endl;
        cudaMemcpy(d_population[i].colors, population[i].colors, numOfVertices * sizeof(int), cudaMemcpyHostToDevice);
    }*/

    /*cout<<"alloc3!"<<endl;
    int blockSize = 256;
    int numBlocks = (populationSize + blockSize - 1) / blockSize;
    int sharedMemSize = numOfVertices * sizeof(bool);
    cudaCalculateFitnessKernel<<<numBlocks, blockSize, sharedMemSize>>>(numOfVertices, d_adjacencyMatrix, d_population, populationSize);*/

    /*for (int i = 0; i < populationSize; i++) {
        cudaMemcpy(&population[i].numOfConflicts, &d_population[i].numOfConflicts, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(population[i].colors, d_population[i].colors, numOfVertices * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_population[i].colors);
    }*/
    // zwalniamy pamięć
    delete[] h_colors;
    delete[] h_colorNum;
    delete[] h_conflictNum;
    // zwalniamy pamięć GPU
    //cudaFree(d_adjacencyMatrix);
    cudaFree(d_colors);
    cudaFree(d_colorNum);
    cudaFree(d_conflictNum);
}

/*void calculateFitness(int numOfVertices, int** adjacencyMatrix, Specimen &specimen) // sprawdzanie jakości rozwiązania
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
}*/

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

/*__global__ void crossoverKernel(Specimen* population, Specimen* newPopulation, int numOfVertices, int populationSize) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= populationSize / 2) return;

    int parent1Idx = idx * 2;
    int parent2Idx = parent1Idx + 1;

    Specimen parent1 = population[parent1Idx];
    Specimen parent2 = population[parent2Idx];

    Specimen offspring1;
    Specimen offspring2;

    offspring1.colors = new int[numOfVertices];
    offspring2.colors = new int[numOfVertices];

    int pivot = randomNumber(0, numOfVertices - 1);

    for (int i = 0; i < numOfVertices; ++i) {
        if (i <= pivot) {
            offspring1.colors[i] = parent1.colors[i];
            offspring2.colors[i] = parent2.colors[i];
        } else {
            offspring1.colors[i] = parent2.colors[i];
            offspring2.colors[i] = parent1.colors[i];
        }
    }

    newPopulation[parent1Idx] = offspring1;
    newPopulation[parent2Idx] = offspring2;
}*/

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
#define outFile "result_gc_cuda.txt"
#define printInterval 10 // co ile generacji wykonać print

int main()
{
    const int populationSize = 10000; // ustawienie całkowitej populacji
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

    int* h_adjacencyMatrix = new int[numOfVertices * numOfVertices]; // spłaszczona macierz sąsiedztwa
    for (int i = 0; i < numOfVertices; i++) {
        for (int j = 0; j < numOfVertices; j++) {
            h_adjacencyMatrix[(i * numOfVertices + j)] = adjacencyMatrix[i][j];
            //cout << "pos " <<i * numOfVertices + j;
        }
    }
    // kopiujemy na GPU
    int* d_adjacencyMatrix; // macierz sąsiedztwa GPU
    size_t adjMatSize = numOfVertices * numOfVertices * sizeof(int); // rozmiar macierzy sąsiedztwa
    cudaMalloc(&d_adjacencyMatrix, adjMatSize); // alokacja pamięci na GPU
    cudaMemcpy(d_adjacencyMatrix, h_adjacencyMatrix, adjMatSize, cudaMemcpyHostToDevice); // kopiowanie do GPU
    delete[] h_adjacencyMatrix; // zwalniamy pamięć
    
    //int* vertexColor = new int [numOfVertices] {}; // kolory wierzchołków
    //bool* colorsUsed = new bool [numOfVertices] {}; // pomocnicza

    Specimen* population = (Specimen*)malloc(populationSize * sizeof(Specimen)); // przydzielamy pamięć dla populacji
    initializePopulation(population, populationSize, numOfVertices);

    /*for (int i = 0; i < populationSize; i++) {
        calculateFitness(numOfVertices, adjacencyMatrix, population[i]); 
    }*/
    cudaCalculateFitness(numOfVertices, /*adjacencyMatrix*//*h_adjacencyMatrix*/d_adjacencyMatrix, population, populationSize); // sprawdzanie jakości rozwiązania naiwnego
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
            calculateFitness(numOfVertices, adjacencyMatrix, population[i]); 
        }*/
        cudaCalculateFitness(numOfVertices, /*adjacencyMatrix*//*h_adjacencyMatrix*/d_adjacencyMatrix, newPopulation, populationSize);

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
    // zwalniamy pamięć
    delete[] population;
    for (int i = 0; i < numOfVertices; i++)
        delete[] adjacencyMatrix[i];
    delete[] adjacencyMatrix;
}