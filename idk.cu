#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <climits>
#include <unordered_set>
#include <fstream>
#include <omp.h>
#include <cuda_runtime.h>

using namespace std;

typedef struct Individual {
    int* chromosome;
    int fitness;
} Individual;

__global__ void calculateFitnessKernel(int* d_adj, int* d_chromosomes, int* d_fitness, int V, int populationSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < populationSize) {
        int fitness = 0;
        int* chromosome = d_chromosomes + idx * V;
        for (int v = 0; v < V; ++v) {
            for (int u = 0; u < V; ++u) {
                if (d_adj[v * V + u] && chromosome[v] == chromosome[u]) {
                    ++fitness;
                }
            }
        }
        d_fitness[idx] = fitness;
    }
}

int calculateFitness(int** adj, int* chromosome, int V) {
    int fitness = 0;
    for (int v = 0; v < V; ++v) {
        for (int u = 0; u < V; ++u) {
            if (adj[v][u] && chromosome[v] == chromosome[u]) {
                ++fitness;
            }
        }
    }
    return fitness;
}

int getRandomNumber(int min, int max) {
    return min + rand() % ((max + 1) - min);
}

void initializePopulation(Individual* population, int populationSize, int V, int numColors) {
#pragma omp parallel for
    for (int i = 0; i < populationSize; ++i) {
        population[i].chromosome = (int*)malloc(V * sizeof(int));
        for (int j = 0; j < V; ++j) {
            population[i].chromosome[j] = getRandomNumber(0, numColors - 1);
        }
    }
}

Individual tournamentSelection(const Individual* population, int populationSize) {
    int tournamentSize = 3;
    Individual bestIndividual = population[getRandomNumber(0, populationSize - 1)];
    for (int i = 1; i < tournamentSize; ++i) {
        Individual candidate = population[getRandomNumber(0, populationSize - 1)];
        if (candidate.fitness < bestIndividual.fitness) {
            bestIndividual = candidate;
        }
    }
    return bestIndividual;
}

void crossover(const Individual& parent1, const Individual& parent2, Individual& offspring1, Individual& offspring2, int V) {
    int crossoverPoint = getRandomNumber(0, V - 1);
    for (int i = 0; i < V; ++i) {
        if (i <= crossoverPoint) {
            offspring1.chromosome[i] = parent1.chromosome[i];
            offspring2.chromosome[i] = parent2.chromosome[i];
        }
        else {
            offspring1.chromosome[i] = parent2.chromosome[i];
            offspring2.chromosome[i] = parent1.chromosome[i];
        }
    }
}

void mutate1(Individual& individual, int numColors, int V) {
    int mutationPoint = getRandomNumber(0, V - 1);
    individual.chromosome[mutationPoint] = getRandomNumber(0, numColors - 1);
}

void swapMutation(Individual& individual, int V) {
    int point1 = getRandomNumber(0, V - 1);
    int point2 = getRandomNumber(0, V - 1);
    while (point1 == point2) {
        point2 = getRandomNumber(0, V - 1);
    }
    swap(individual.chromosome[point1], individual.chromosome[point2]);
}

void shiftMutation(Individual& individual, int numColors, int V) {
    int start = getRandomNumber(0, V - 1);
    int end = getRandomNumber(0, V - 1);
    while (start == end) {
        end = getRandomNumber(0, V - 1);
    }
    if (start > end) {
        swap(start, end);
    }
    for (int i = start; i <= end; ++i) {
        individual.chromosome[i] = (individual.chromosome[i] + 1) % numColors;
    }
}

void recombinationMutation(Individual& individual1, Individual& individual2, int V) {
    for (int i = 0; i < V; ++i) {
        if (rand() % 2 == 0) {
            swap(individual1.chromosome[i], individual2.chromosome[i]);
        }
    }
}

void mutate(Individual& individual, Individual& individual2, int numColors, int V) {
    int mutationType = getRandomNumber(0, 3);
    switch (mutationType) {
    case 0:
        swapMutation(individual, V);
        break;
    case 1:
        shiftMutation(individual, numColors, V);
        break;
    case 2:
        recombinationMutation(individual, individual2, V);
        break;
    case 3:
        mutate1(individual, numColors, V);
        break;
    }
}

bool compareFitness(const Individual& a, const Individual& b) {
    return a.fitness < b.fitness;
}

void sortPopulation(Individual* population, int populationSize) {
    sort(population, population + populationSize, compareFitness);
}

void calculatePopulationFitness(int** adj, Individual* population, int populationSize, int V) {
#pragma omp parallel for
    for (int i = 0; i < populationSize; ++i) {
        population[i].fitness = calculateFitness(adj, population[i].chromosome, V);
    }
}

void calculatePopulationFitnessCUDA(int** adj, Individual* population, int populationSize, int V) {
    int* h_adj = new int[V * V];
    int* h_chromosomes = new int[populationSize * V];
    int* h_fitness = new int[populationSize];

    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            h_adj[i * V + j] = adj[i][j];
        }
    }

    for (int i = 0; i < populationSize; ++i) {
        for (int j = 0; j < V; ++j) {
            h_chromosomes[i * V + j] = population[i].chromosome[j];
        }
    }

    int* d_adj;
    int* d_chromosomes;
    int* d_fitness;

    cudaMalloc(&d_adj, V * V * sizeof(int));
    cudaMalloc(&d_chromosomes, populationSize * V * sizeof(int));
    cudaMalloc(&d_fitness, populationSize * sizeof(int));

    cudaMemcpy(d_adj, h_adj, V * V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_chromosomes, h_chromosomes, populationSize * V * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (populationSize + blockSize - 1) / blockSize;

    calculateFitnessKernel << <gridSize, blockSize >> > (d_adj, d_chromosomes, d_fitness, V, populationSize);

    cudaMemcpy(h_fitness, d_fitness, populationSize * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < populationSize; ++i) {
        population[i].fitness = h_fitness[i];
    }

    delete[] h_adj;
    delete[] h_chromosomes;
    delete[] h_fitness;

    cudaFree(d_adj);
    cudaFree(d_chromosomes);
    cudaFree(d_fitness);
}

Individual geneticAlgorithm(int** adj, int V, int populationSize, int numGenerations, int numColors) {
    Individual* population = (Individual*)malloc(populationSize * sizeof(Individual));

    initializePopulation(population, populationSize, V, numColors);

     calculatePopulationFitnessCUDA(adj, population, populationSize, V);

    sortPopulation(population, populationSize);

    Individual bestIndividual = population[0];

    for (int generation = 0; generation < numGenerations; ++generation) {
        Individual* newPopulation = (Individual*)malloc(populationSize * sizeof(Individual));
        initializePopulation(newPopulation, populationSize, V, numColors);

#pragma omp parallel for
        for (int i = 0; i < populationSize / 2; ++i) {
            Individual parent1 = tournamentSelection(population, populationSize);
            Individual parent2 = tournamentSelection(population, populationSize);
            Individual offspring1;
            offspring1.chromosome = (int*)malloc(V * sizeof(int));
            Individual offspring2;
            offspring2.chromosome = (int*)malloc(V * sizeof(int));
            crossover(parent1, parent2, offspring1, offspring2, V);

            if (getRandomNumber(0, 100) < 60) {
                mutate(offspring1, offspring2, numColors, V);
            }
            newPopulation[i] = offspring1;
            newPopulation[i + populationSize / 2] = offspring2;
        }
        calculatePopulationFitnessCUDA(adj, newPopulation, populationSize, V);
#pragma omp parallel for
        for (int i = 0; i < populationSize; ++i) {
            memcpy(population[i].chromosome, newPopulation[i].chromosome, V * sizeof(int));
            population[i].fitness = newPopulation[i].fitness;
        }

#pragma omp parallel for
        for (int i = 0; i < populationSize; ++i) {
            free(newPopulation[i].chromosome);
        }
        free(newPopulation);

        sortPopulation(population, populationSize);

        if (population[0].fitness < bestIndividual.fitness) {
            bestIndividual = population[0];
        }

        //cout << "Generation " << generation << " Best Fitness: " << bestIndividual.fitness << endl;

        if (bestIndividual.fitness == 0) break;
    }

    free(population);
    return bestIndividual;
}

int readMatrix(int size, int** a, const char* filename) {
    FILE* pf;
    pf = fopen(filename, "r");
    if (pf == NULL)
        return 0;

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            fscanf(pf, "%d", &a[i][j]);
        }
    }

    fclose(pf);
    return 1;
}

int main() {
    int populationSize = 100;
    int numGenerations = 3000;
    int numColors = 73;
    int N = 128;
    int** adj = (int**)malloc(N * sizeof(int*));
    if (adj == NULL) {
        printf("Memory allocation error for adj");
        return 1;
    }
    for (int i = 0; i < N; ++i) {
        adj[i] = (int*)malloc(N * sizeof(int));
        if (adj[i] == NULL) {
            printf("Memory allocation error for adj %d\n", i);
            return 1;
        }
    }

    const char* filename = "adjacency_matrix.txt";
    if (!readMatrix(N, adj, filename)) {
        cerr << "Error reading matrix from file" << endl;
        return 1;
    }
    clock_t start, end;
    double cpu_time_used;

    srand(time(NULL));
    start = clock();
    Individual bestSolution = geneticAlgorithm(adj, N, populationSize, numGenerations, numColors);

    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("Time : %f\n", cpu_time_used);
    std::cout << "Best Solution Fitness: " << bestSolution.fitness << endl;
    std::cout << "Coloring: ";
    for (int i = 0; i < N; ++i) {
        std::cout << bestSolution.chromosome[i] << " ";
    }
    std::cout << endl;

    for (int i = 0; i < N; ++i) {
        free(adj[i]);
    }
    free(adj);

    return 0;
}