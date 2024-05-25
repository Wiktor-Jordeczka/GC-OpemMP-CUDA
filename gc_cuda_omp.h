#include <string>

std::string inFile = "queen7_7.txt"; // -i (path); plik wejściowy
std::string outFile = "result_gc_cuda_omp.txt"; // -o (path); plik wyjściowy
int populationSize = 1000; // -p (int); rozmiar populacji
int iterations = 1000; // -l (int); maksymalna liczba iteracji (generacji)
int mutationChance = 50; // -m (int); prawdopodobieństwo mutacji <0;100>
int stopTime = 60; // -s (int); maksymalny czas działania w sekundach
bool verbose = false; // -v (int); czy wypisywać printy i dodatkowe dane wyjściowe
int printInterval = 100; // -v (int); co ile generacji wykonać print
std::random_device rd; // losowy seed
unsigned int seed = rd(); // -r (int); seed do generatora liczb losowych