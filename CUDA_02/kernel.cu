__global__ void sq_mat_mul_kernel(float* d_A, float* d_B, float* d_C, int N)
{
    // Identifying the thread mapping
    // Working on C[i,j]
    int i = blockDim.y*blockIdx.y + threadIdx.y;
    int j = blockDim.x*blockIdx.x + threadIdx.x;

    // Check the edge cases
    if (i < N && j < N)
    {
        // Value at C[i,j]
        float value = 0;
        // Loop over elements of the two vectors
        for (int k = 0; k < N; k++)
        {
            // Multiply and add
            value += A[i*N+k] * B[k*N+j];
        }

        // Assigning calculated value
        C[i*N+j] = value;
    }
}