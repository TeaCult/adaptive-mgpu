
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

# Function to train a model on the GPU
def train_model_on_gpu(process_id, trained_models):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Initialize model, loss function, and optimizer
    model = SimpleModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Generate random data on the GPU
    X_train = torch.tensor(np.random.random((1000, 10)), dtype=torch.float32).to(device)
    y_train = torch.tensor(np.random.random((1000, 10)), dtype=torch.float32).to(device)

    # Training loop
    for epoch in range(100):  # Adjust epochs as needed
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Append the trained model to the shared list
    trained_models[process_id] = model.to('cpu')  # Move to CPU to free up GPU memory

    print(f"Training completed for model {process_id}")

# Function to perform inference on a model with unique input
def perform_inference(process_id, trained_models, results_list):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load the pre-trained model and move it back to GPU for inference
    model = trained_models[process_id].to(device)
    model.eval()

    # Generate a unique random input for this model
    input_data = np.random.random((1, 10))

    # Perform inferences at a rate of 5 inferences per second
    start_time = time.time()
    inference_count = 0
    while inference_count < 5:
        with torch.no_grad():
            input_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)
            output = model(input_tensor)  # Perform inference
            results_list.append((process_id, output.cpu().numpy()))
        inference_count += 1
        time.sleep(1 / 5)  # Adjust to achieve 5 inferences per second

    end_time = time.time()
    print(f"Inference for model {process_id} completed in {end_time - start_time:.2f} seconds")

# Main function to manage training and inference
def main():
    manager = Manager()
    results_list = manager.list()  # Shared list to collect inference results
    trained_models = manager.dict()  # Shared dictionary to store trained models

    # Training phase
    print("Starting model training...")
    num_processes = 10  # Limit number of concurrent processes for training
    total_models = 50   # Total models to train and use for inference

    with Pool(processes=num_processes) as pool:
        pool.starmap(train_model_on_gpu, [(i, trained_models) for i in range(total_models)])

    # Inference phase
    print("\nStarting inference...")
    with Pool(processes=num_processes) as pool:
        pool.starmap(perform_inference, [(i, trained_models, results_list) for i in range(total_models)])

    # Display inference results
    print("\nAll inference tasks completed.")
    for process_id, output in results_list:
        print(f"Process {process_id} output: {output}")

if __name__ == "__main__":
    main()
