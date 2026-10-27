from mpi4py import MPI
import cupy as cp
import cupyx.scipy.sparse as sparse
from cupyx.scipy.sparse.linalg import minres
import time

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Ensure exactly 2 processes
if size != 2:
    raise ValueError("This code is designed for exactly 2 GPUs (2 MPI ranks).")

# Set GPU for each rank
cp.cuda.Device(rank).use()

# Parameters for sparse matrix
N = 10000  # Size of the sparse matrix
density = 0.0005 #0.001  # Sparsity level

# Generate random sparse matrix and vector once (not in loop to exclude generation time)
A = sparse.random(N, N, density=density, format='csr', dtype=cp.float32)
b = cp.random.rand(N, dtype=cp.float32)

# Warm up to stabilize performance
solution, _ = minres(A, b)

# Start timing the 100 solves
start_time = time.time()

for _ in range(100):
    solution, info = minres(A, b)
    if info != 0:
        print(f"Rank {rank}: Solution did not converge.")

end_time = time.time()

# Calculate average time per solve
total_time = end_time - start_time
average_time = total_time / 100

# Print results
print(f"Rank {rank} Total Time for 100 solves: {total_time:.4f} seconds")
print(f"Rank {rank} Average Time per solve: {average_time:.4f} seconds")
