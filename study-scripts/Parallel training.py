# export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=10  
# sudo nvidia-cuda-mps-control -d
# sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from multiprocessing import Process

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(10, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Function to train a model on the GPU
def train_model_on_gpu():
    # Ensure GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Initialize model, loss function, and optimizer
    model = SimpleModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Generate random data on the GPU
    X_train = torch.tensor(np.random.random((1000, 10)), dtype=torch.float32).to(device)
    y_train = torch.tensor(np.random.random((1000, 10)), dtype=torch.float32).to(device)

    # Training loop
    for epoch in range(1000):  # Small number of epochs for demonstration
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    print("Training completed for one model on GPU")

# Create and start 10 processes for concurrent GPU training
processes = []
for _ in range(50):
    p = Process(target=train_model_on_gpu)
    processes.append(p)
    p.start()

# Wait for all processes to finish
for p in processes:
    p.join()

print("All training tasks completed on GPU.")


###################### WITH TRAINING TIMES ############################
# export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=5


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from multiprocessing import Pool, Manager
import time

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(10, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Function to train a model on the GPU, with retry in case of failure
def train_model_on_gpu(process_id, times_list):
    success = False
    attempt = 0
    while not success:
        try:
            attempt += 1
            start_time = time.time()

            # Ensure GPU is available
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # Initialize model, loss function, and optimizer
            model = SimpleModel().to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Generate random data on the GPU
            X_train = torch.tensor(np.random.random((1000, 10)), dtype=torch.float32).to(device)
            y_train = torch.tensor(np.random.random((1000, 10)), dtype=torch.float32).to(device)

            # Training loop
            for epoch in range(1000):  # Adjust epochs as needed
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()

            end_time = time.time()
            training_time = end_time - start_time

            # Append results to the shared list and print completion
            times_list.append((process_id, training_time))
            print(f"Training completed for model {process_id} on GPU in {training_time:.2f} seconds (Attempt {attempt})")
            success = True  # Mark as successful

        except RuntimeError as e:
            if "CUDA error" in str(e):
                print(f"Process {process_id} encountered a CUDA error on attempt {attempt}. Retrying...")
                time.sleep(1)  # Optional short delay before retrying
            else:
                print(f"Process {process_id} encountered an unexpected error: {e}")
                break  # Stop retrying for non-CUDA errors

# Main function to manage concurrent training
import time
def main():
    s=time.time()
    manager = Manager()
    times_list = manager.list()  # Shared list to collect times

    # Use Pool to manage concurrent processes with a limit on active processes
    num_processes = 10  # Limit number of concurrent processes
    total_models = 50   # Total models to train
    with Pool(processes=num_processes) as pool:
        pool.starmap(train_model_on_gpu, [(i, times_list) for i in range(total_models)])

    # Display all training times
    print("\nAll training tasks completed on GPU.")
    for process_id, training_time in times_list:
        print(f"Process {process_id} finished training in {training_time:.2f} seconds")
    print('Total Time:',time.time()-s)
if __name__ == "__main__":
    main()

# 1000x10 1000x10 1000 16 process at once with %10 MPS finishes  50 training (concurrent) in 11 seonds 
# 1000x10 1000x10 1000 epoch Single process finishes 2.78 with %100 GPU MPS setting
# 1000x10 1000x10 1000 epoch CPU and GPU Benchmarks are 2.51 and 2.68 respectively. Thus there is a definitive gain here (10 times better with GPU)
# 1500x10 - 1500x10 5000 epoch training results 7.10s GPU  22.61s on CPU  50 training 36.29 parallel in GPU per training 
# (lost its edge against single GPU process however a lot more advantages compared to GPU)

################# SINGLE TRAINING ON GPU AND CPU BENCHMARKS ###################
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(10, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Function to train a model
def train_model(device):
    model = SimpleModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Generate random data on the device
    X_train = torch.tensor(np.random.random((1000, 10)), dtype=torch.float32).to(device)
    y_train = torch.tensor(np.random.random((1000, 10)), dtype=torch.float32).to(device)

    # Training loop
    for epoch in range(1000):  # Adjust epochs as needed
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

# Measure GPU time without MPS
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running training on GPU without MPS...")
    start_time = time.time()
    train_model(device)
    gpu_time = time.time() - start_time
    print(f"Training completed on GPU without MPS in {gpu_time:.2f} seconds")

# Measure CPU time
device = torch.device("cpu")
print("Running training on CPU...")
start_time = time.time()
train_model(device)
cpu_time = time.time() - start_time
print(f"Training completed on CPU in {cpu_time:.2f} seconds")
