#include <stdio.h>
#include <stdlib.h>

#define MAX_NUM 10 
#define MIN_NUM -10 

int main(int argc, char const *argv[])
{
    // Matrix Size
    int N = 4000;

    // Generate NxN square matrices A and B
    float* A = (float*)malloc(N*N*sizeof(float));
    float* B = (float*)malloc(N*N*sizeof(float));
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i*N+j] = (float)(rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM);
            B[i*N+j] = (float)(rand() % (MAX_NUM - MIN_NUM + 1) + MIN_NUM);
        }
    }
    // Initialize space for result C
    float* C = (float*)malloc(N*N*sizeof(float));

    // 1) VRAM Allocation
    
    // 2) Copy data from RAM to VRAM
    
    // 3) GPU Matrix Multiplication (Kernel Function)
    
    // 4) Copy results from VRAM to RAM

    // 5) Free VRAM
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}