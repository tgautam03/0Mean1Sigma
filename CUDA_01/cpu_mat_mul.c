void sq_mat_mul_cpu(float* A, float* B, float* C, int N)
{
    // Loop over rows of A
    for (int i = 0; i < N; i++)
    {
        // Loop over columns of B
        for (int j = 0; j < N; j++)
        {
            // Value at C[i,j]
            float value = 0;
            // Loop over elements of the two vectors
            for (int k = 0; k < N; k++)
            {
                value += A[i*N+k] * B[k*N+j];
            }

            // Assigning calculated value
            C[i*N+j] = value;
        }
    }
}