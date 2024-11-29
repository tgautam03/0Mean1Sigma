void cpu_matmul(float *A_mat, float *B_mat, float *C_mat, int C_n_rows, int C_n_cols, int A_n_cols)
{
    for (int row = 0; row < C_n_rows; row++)
    {
        for (int col = 0; col < C_n_cols; col++)
        {
            float val = 0.0f;
            for (int k = 0; k < A_n_cols; k++)
            {
                val += A_mat[row*A_n_cols + k] * B_mat[k*C_n_cols + col];
            }
            C_mat[row*C_n_cols + col] = val;
        }
    }
}