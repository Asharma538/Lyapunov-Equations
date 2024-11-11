#include <bits/stdc++.h>
#include <Eigen/Dense>
#include <iomanip>
#include <complex>
#include <chrono>

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>


using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std::chrono;

/* macros */
#define ll long long
#define repeat(i,s,e) for(ll i=s;i<e;i++)
#define print(s) std::cout<<s<<'\n';


// Kronecker product of two matrices
MatrixXd kron(const MatrixXd& A, const MatrixXd& B) {
    int aRows = A.rows(), aCols = A.cols();
    int bRows = B.rows(), bCols = B.cols();
    MatrixXd result(aRows * bRows, aCols * bCols);
    
    for (int i = 0; i < aRows; ++i) {
        for (int j = 0; j < aCols; ++j) {
            result.block(i * bRows, j * bCols, bRows, bCols) = A(i, j) * B;
        }
    }
    return result;
}

// Kronecker sum of two matrices
MatrixXd kronsum(const MatrixXd& A, const MatrixXd& B) {
    int nA = A.rows(), nB = B.rows();

    // Identity matrices
    MatrixXd I_A = MatrixXd::Identity(nA, nA);
    MatrixXd I_B = MatrixXd::Identity(nB, nB);

    // Kronecker sum: kron(A, I_B) + kron(I_A, B)
    MatrixXd term1 = kron(A, I_B);
    MatrixXd term2 = kron(I_A, B);

    return term1 + term2;
}

// Vectorize a matrix (column-major order)
VectorXd vectorize(const MatrixXd& A) {
    return Eigen::Map<const VectorXd>(A.data(), A.size());
}

// Unvectorize a vector into a matrix
MatrixXd unvectorize(const VectorXd& vec, int rows) {
    int cols = vec.size() / rows;
    return Eigen::Map<const MatrixXd>(vec.data(), rows, cols);
}

// Inverse of a matrix (Eigen provides built-in inverse function)
MatrixXd inv(const MatrixXd& A) {
    return A.inverse(); // Eigen's built-in inverse method
}

// Function to print matrix (for debugging purposes)
void printMatrix(const MatrixXd& M) {
    std::cout << M << std::endl;
}


int main(){
    /* Dimension of the matrices - input file */
    std::ifstream dimension_file("input/n.txt");


    /* Matrices A and W  - input files */
    std::ifstream matrix_A_file("input/matrix_A.txt");
    std::ifstream matrix_W_file("input/matrix_W.txt");


    /* Reading the Dimension of Matrices */
    ll n;
    dimension_file>>n;
    /* Variable Definitions |  AX + X(A.T) = W , all matrices are of order nxn */
    MatrixXd A(n, n),W(n, n),X(n, n);

    /* Reading the Matrices from their respective files */
    repeat(i,0,n) repeat(j,0,n) matrix_A_file>>A(i,j);
    repeat(i,0,n) repeat(j,0,n) matrix_W_file>>W(i,j);

    /* Performing Schur Decomposition of the Matrix A */
    MatrixXd T,Q;

    auto start = high_resolution_clock::now();

    Eigen::initParallel();
    Eigen::RealSchur<MatrixXd> schur(A);
    T = schur.matrixT();
    Q = schur.matrixU();

    // Keeping the block sizes in R
    VectorXd R;

    // Constructing R
    ll i = n-1;
    while(i>=0){
        // Increasing the size of R
        R.conservativeResize(R.size()+1);

        // Assigning the block size to the new element just added
        if ( i-1 >= 0 and (abs(T(i,i) - std::conj(T(i-1,i-1))) < 1e-12) ){
            R(R.size()-1) = 2; i -= 2;
        }
        else {
            R(R.size()-1) = 1; i -= 1;
        }
    }

    ll C_size = n;
    ll sigmar = 0;

    // The matrix C  = Q.T @ W @ Q
    // MatrixXd C = Q.transpose() * W * Q;
    MatrixXd C = W;

    // The matrix Z = Q.T @ X @ Q
    MatrixXd Z = MatrixXd::Zero(n,n);

    // Maintaining index
    int ir = 0;
    // r -> Block size
    for(int r : R){
        int cut_ind = C_size - sigmar - r;

        // block args are : ( row_begin,col_begin,block_rows,block_cols )
        // Slicing C
        MatrixXd C11 = C.block(0, 0, cut_ind, cut_ind);
        MatrixXd C12 = C.block(0, cut_ind, cut_ind, r);
        MatrixXd C21 = C.block(cut_ind, 0, r, cut_ind);
        MatrixXd C22 = C.block(cut_ind, cut_ind, r, r);
        
        // Slicing U
        MatrixXd T11 = T.block(0,0,cut_ind,cut_ind);
        MatrixXd T12 = T.block(0,cut_ind,cut_ind,r);
        MatrixXd T21 = T.block(cut_ind,0,r,cut_ind);
        MatrixXd T22 = T.block(cut_ind,cut_ind,r,r);

        // Solving the 4th Equation
        MatrixXd Z22 = unvectorize(inv(kronsum(T22,T22)) * vectorize(C22),r);
        Z.block(cut_ind,cut_ind,r,r) = Z22;

        if (ir == R.size() - 1) break;

        // Solving the 2nd and 3rd Equation
        MatrixXd D12 = C12 - (T12 * Z22);
        int sigmaj = 0;

        MatrixXd D12_tilde = D12;

        int ij = 0;
        for(int j : R.segment(ir + 1, R.size() - (ir + 1))){
            int partition_ind = cut_ind - sigmaj - j;
            int sigmak = 0;
            int ik = 0;
            for (int k : R.segment(ir+1,ij)){
                int loop_ind = cut_ind - sigmak - k;
                D12_tilde.block(partition_ind,0,j,D12_tilde.cols()) -= T.block(partition_ind,loop_ind,j,k) * Z.block(loop_ind,cut_ind,k,r);
                sigmak+=k;
                ik+=1;
            }
            
            Z.block(partition_ind,cut_ind,j,r) = unvectorize( inv(kronsum(T.block(partition_ind,partition_ind,j,j),T22)) * vectorize(D12_tilde.block(partition_ind,0,j,D12_tilde.cols())), r);
            sigmaj+=j;
            ij+=1;
        }
        Z.block(cut_ind,0,r,cut_ind) = Z.block(0,cut_ind,cut_ind,r).transpose();
        C.block(0,0,cut_ind,cut_ind) -= (T.block(0,cut_ind,cut_ind,r) * Z.block(cut_ind,0,r,cut_ind)) + (Z.block(0,cut_ind,cut_ind,r) * T.block(0,cut_ind,cut_ind,r).transpose());

        sigmar+=r;
        ir += 1;
    }
    X = Q * Z * Q.transpose();

    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop-start);
    std::cout<<duration.count()<<'\n';

    return 0;
}
// nvcc -I /usr/local/include/eigen-3.4.0/ parallel_solver.cu -o Acu