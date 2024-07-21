void CUDA_CHECK(cudaError_t err) 
{
    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d \n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

// Device array pointers
float* d_A;
float* d_B;
float* d_C;

// Device memory allocation
cudaError_t err_A = cudaMalloc((void**) &d_A, N*N*sizeof(float));
CUDA_CHECK(err_A);

cudaError_t err_B = cudaMalloc((void**) &d_B, N*N*sizeof(float));
CUDA_CHECK(err_B);

cudaError_t err_C = cudaMalloc((void**) &d_C, N*N*sizeof(float));
CUDA_CHECK(err_C);