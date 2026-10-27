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

# Set a maximum memory pool limit, e.g., 2 GB
memory_pool = cp.cuda.MemoryPool()
memory_pool.set_limit(size=2 * 1024**3)  # 2 GB
cp.cuda.set_allocator(memory_pool.malloc)

# Parameters for sparse matrix
N = 20000  # Increase matrix size as needed
density = 0.0005

# Generate random sparse matrix and vector once
A = sparse.random(N, N, density=density, format='csr', dtype=cp.float32)
b = cp.random.rand(N, dtype=cp.float32)

# Start timing
start_time = time.time()

for i in range(100):
    solution, info = minres(A, b)
    if info != 0:
        print(f"Rank {rank}: Solution did not converge.")

    # Monitor and clear memory if usage is above a set limit
    memory_used = cp.get_default_memory_pool().used_bytes()
    max_limit = 2 * 1024**3  # 2 GB limit

    if memory_used > max_limit:
        cp.get_default_memory_pool().free_all_blocks()

end_time = time.time()

# Calculate average time per solve
total_time = end_time - start_time
average_time = total_time / 100

print(f"Rank {rank} Total Time for 100 solves: {total_time:.4f} seconds")
print(f"Rank {rank} Average Time per solve: {average_time:.4f} seconds")
