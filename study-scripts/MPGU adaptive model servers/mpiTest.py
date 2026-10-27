from mpi4py import MPI
import cupy as cp
import cupyx.scipy.sparse as sparse
from cupyx.scipy.sparse.linalg import minres

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Check if we have exactly two processes
if size != 2:
    raise ValueError("This code is designed for exactly 2 GPUs (2 MPI ranks).")

# Set GPU for each rank
cp.cuda.Device(rank).use()

# Parameters for sparse matrix
N = 10000  # Size of the sparse matrix
density = 0.001  # Sparsity level

# Generate random sparse matrices on each rank
if rank == 0:
    A = sparse.random(N, N, density=density, format='csr', dtype=cp.float32)
    b = cp.random.rand(N, dtype=cp.float32)
elif rank == 1:
    A = sparse.random(N, N, density=density, format='csr', dtype=cp.float32)
    b = cp.random.rand(N, dtype=cp.float32)

# Solve the system using minres for each matrix
solution, info = minres(A, b)

# Gather solutions on rank 0
if rank == 0:
    print(f"Rank 0 Solution Info: {info}")
elif rank == 1:
    print(f"Rank 1 Solution Info: {info}")

# Print or process solutions if needed
if info == 0:
    print(f"Rank {rank}: Solution converged successfully.")
else:
    print(f"Rank {rank}: Solution did not converge.")

MPI.Finalize()
