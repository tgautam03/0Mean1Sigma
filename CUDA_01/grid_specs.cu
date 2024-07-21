// Grid Specs (x, y, z) 
dim3 dim_block(4, 3, 1);
dim3 dim_grid(ceil(N/4.0), ceil(N/3.0), 1);
// Kernel Execution
sq_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, N);