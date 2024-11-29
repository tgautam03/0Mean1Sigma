#include <Eigen/Dense>

int main(int argc, char const *argv[])
{
    // Generate Eigen square matrices A, B and C
    Eigen::MatrixXd A_eigen(n, n);
    Eigen::MatrixXd B_eigen(n, n);
    Eigen::MatrixXd C_eigen(n, n);

    // Initialize Matrices
    // .
    // .
    // .

    // Matrix Multiplication
    C_eigen = A_eigen * B_eigen;
    
    return 0;
}
