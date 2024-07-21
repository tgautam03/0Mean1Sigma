// Copying A and B to device memory
cudaError_t err_A_ = cudaMemcpy(d_A, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
CUDA_CHECK(err_A_);

cudaError_t err_B_ = cudaMemcpy(d_B, B, N*N*sizeof(float), cudaMemcpyHostToDevice);
CUDA_CHECK(err_B_);

// Kernel execution

// Copy back results
cudaError_t err_C_ = cudaMemcpy(C, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost);
CUDA_CHECK(err_C_);