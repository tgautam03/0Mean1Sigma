__global__ void tiled_sq_mat_mul_kernel(float* A, float* B, float* C, int N)
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
    for (int phase = 0; phase < ceil((float)N/TILE_WIDTH); phase++) // Ceiling function to ensure that extra phase at the boundary 
    {
        // Load Tiles into shared memory by checking locations
        if ((i < N) && ((phase*TILE_WIDTH+tx) < N))
          sh_A[ty][tx] = A[(i)*N + phase*TILE_WIDTH+tx];
        else
          sh_A[ty][tx] = 0.0f;

        if (((phase*TILE_WIDTH + ty) < N) && (j < N))
          sh_B[ty][tx] = B[(phase*TILE_WIDTH + ty)*N+j];
        else
          sh_B[ty][tx] = 0.0f;
        __syncthreads();

        // Dot product
        for (int k = 0; k < TILE_WIDTH; k++)
            value += sh_A[ty][k] * sh_B[k][tx];
        __syncthreads();
    }
    // Assigning calculated value by checking location
    if ((i < N) && (j < N))
      C[i*N+j] = value;
}