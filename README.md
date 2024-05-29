# Kolorowanie grafu z wykorzystaniem algorytmu genetycznego oraz techologii przetwarzania równoległego OpenMP i CUDA
## Opis programów i analiza wydajności
[Exportowane z DokuWiki](DokuWikiExport.pdf)
## Opis Repozytorium
Dostępne są pliki:
 - gc_seq z algorytmem sekwencyjnym
 - gc_omp z algorytmem równoległym z wykorzystaniem OpenMP
 - gc_cuda_omp z algorytmem równoległym z wykorzystaniem CUDA + OpenMP
 - runTests przydatny do uruchamiania serii testów
 - gene generujący graf spójny o zadanej ilości wierzchołków i wypełnieniu krawędzi
 - katalog input/ zawierający gotowe instancje testowe
## Kompilacja
### Kompilatory
1. g++ 12.2.0
2. nvcc, Cuda compilation tools, release 12.5, V12.5.40
### Komendy
1. g++ .\gc_seq.cpp -o .\gc_seq.exe -O2
2. g++ .\gc_omp.cpp -o .\gc_omp.exe -O2 -fopenmp
3. nvcc .\gc_cuda_omp.cu -o gc_cuda_omp.exe -O2 -Xcompiler /openmp
4. g++ .\gene.cpp -o .\gene.exe -O2
5. g++ .\runTests.cpp -o .\runTests.exe -O2
