__global__ void sq_mat_mul_kernel(float* d_A, float* d_B, float* d_C, int N)
{
    // Identifying the thread mapping
    // Working on C[i,j]
    int row = blockDim.y*blockIdx.y + threadIdx.y;
    int col = blockDim.x*blockIdx.x + threadIdx.x;

    // Check the edge cases
    if (row < N && col < N)
    {
        // Value at C[row,col]
        float value = 0;
        // Loop over elements of the two vectors
        for (int k = 0; k < N; k++)
        {
            // Multiply and add
            value += A[row*N+k] * B[k*N+col];
        }

        // Assigning calculated value
        C[row*N+col] = value;
    }
}