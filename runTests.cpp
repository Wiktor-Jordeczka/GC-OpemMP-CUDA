#include <iostream>
#include <sstream>
#include <random>
using namespace std;

int main(){
    for(int i=1000; i<=10000; i+= 1000){
        string executable = "gc_omp.exe";
        string inFile = "input/queen7_7.txt"; // -i (path); plik wejściowy
        string outFile = "output/result_gc_cuda_omp_"+to_string(i)+".txt"; // -o (path); plik wyjściowy
        string populationSize = to_string(i); // -p (int); rozmiar populacji
        string iterations = to_string(1000); // -l (int); maksymalna liczba iteracji (generacji)
        string mutationChance = to_string(50); // -m (int); prawdopodobieństwo mutacji <0;100>
        string stopTime = to_string(60); // -s (int); maksymalny czas działania w sekundach
        string printInterval = to_string(100); // -v (int); co ile generacji wykonać print
        std::random_device rd; // losowy seed
        string seed = to_string(rd()); // -r (int); seed do generatora liczb losowych

        string str = executable+" -i "+inFile+" -o "+outFile+" -p "+populationSize+" -l "+iterations+" -m "+mutationChance+" -s "+stopTime+/*" -v "+printInterval+*/" -r "+seed;
        system(str.c_str());
    }
    return 0;
}