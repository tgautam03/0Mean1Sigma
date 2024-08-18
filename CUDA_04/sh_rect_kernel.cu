__global__ void tiled_mat_mul_kernel(float* A, float* B, float* C, int N1, int N2, int N3)
{
    // Details regarding this thread
    int by = blockIdx.y;
    int bx = blockIdx.x; 

    int ty = threadIdx.y;
    int tx = threadIdx.x; 

    // Working on C[i,j]
    int i = TILE_WIDTH*by + ty;
    int j = TILE_WIDTH*bx + tx;

    // Allocating shared memory
    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    // Parallel mat mul
    float value = 0;
    for (int phase = 0; phase < ceil((float)N2/TILE_WIDTH); phase++)
    {
        // Load Tiles into shared memory
        if ((i < N1) && ((phase*TILE_WIDTH+tx) < N2))
          sh_A[ty][tx] = A[(i)*N2 + phase*TILE_WIDTH+tx];
        else
          sh_A[ty][tx] = 0.0f;

        if (((phase*TILE_WIDTH + ty) < N2) && (j < N3))
          sh_B[ty][tx] = B[(phase*TILE_WIDTH + ty)*N3+j];
        else
          sh_B[ty][tx] = 0.0f;
        __syncthreads();

        // Dot product
        for (int k = 0; k < TILE_WIDTH; k++)
            value += sh_A[ty][k] * sh_B[k][tx];
        __syncthreads();
    }
    // Assigning calculated value
    if ((i < N1) && (j < N3))
      C[i*N3+j] = value;
}