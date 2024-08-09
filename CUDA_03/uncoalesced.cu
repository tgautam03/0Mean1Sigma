int col = blockDim.y*blockIdx.y + threadIdx.y;
int row = blockDim.x*blockIdx.x + threadIdx.x;