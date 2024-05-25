## Kompilacja
### Kompilatory
1. g++ 12.2.0
2. nvcc, Cuda compilation tools, release 12.5, V12.5.40
### Komendy
1. g++ .\gc_seq.cpp -o .\gc_seq.exe -O2
2. nvcc .\gc_cuda_omp.cu -o gc_cuda_omp.exe -O2 -Xcompiler /openmp
