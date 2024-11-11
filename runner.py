import numpy as np
from scipy.sparse import random as sparse_random
from scipy.sparse import coo_matrix
from scipy import stats
import subprocess
import os


print("Compiling `A.cu` and `A.cpp`")
cuda_compile_command = "nvcc -I /usr/local/include/eigen-3.4.0/ parallel_solver.cu -o Acu"
cuda_compile_result = subprocess.run(cuda_compile_command, shell=True, capture_output=True, text=True)

cpp_compile_command = "nvcc -I /usr/local/include/eigen-3.4.0/ solver.cpp -o Acpp"
cpp_compile_result = subprocess.run(cpp_compile_command, shell=True, capture_output=True, text=True)
print("Compilation complete")

def generate_matrix(n):
    # Generate a random matrix
    matrix = np.random.rand(n, n)
    # Normalize each column to sum to 1
    matrix /= matrix.sum(axis=0)
    return matrix

# sizes = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
sizes = [100, 200, 500, 1000, 2000, 5000]


open("./output/results.txt", 'w').write("")
open("./output/results.txt", 'w').write("n | output_of_A.cpp | output_of_A.cu\n")


for size in sizes:
    print("Starting for size", size)

    matrix = generate_matrix(size)
    np.savetxt(f'./input/matrix_A.txt', matrix, delimiter=' ', fmt='%f')

    matrix = generate_matrix(size)
    np.savetxt(f'./input/matrix_W.txt', matrix, delimiter=' ', fmt='%f')

    matrix_A_size = os.path.getsize('./input/matrix_A.txt')
    matrix_W_size = os.path.getsize('./input/matrix_W.txt')

    print(f'Size of matrix: {matrix_W_size} bytes')

    open(f'./input/n.txt', 'w').write(str(size))

    # run the CUDA program
    cuda_run_command = "./Acu"
    cuda_run_result = subprocess.run(cuda_run_command, shell=True, capture_output=True, text=True)

    # run the C++ program
    cpp_run_command = "./Acpp"
    cpp_run_result = subprocess.run(cpp_run_command, shell=True, capture_output=True, text=True)

    # Record the outputs in a table
    with open(f'./output/results.txt', 'a+') as f:
        f.write(f'{size} | {cpp_run_result.stdout.strip()} | {cuda_run_result.stdout.strip()}\n')

    print(f'Results for size {size} recorded in results.txt')