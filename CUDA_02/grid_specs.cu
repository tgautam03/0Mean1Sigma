// Grid Specs (x, y, z) 
dim3 dim_block(8, 8, 1);
dim3 dim_grid(ceil(32/8.0), ceil(32/8.0), 1);
// Kernel Execution
sq_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, N);