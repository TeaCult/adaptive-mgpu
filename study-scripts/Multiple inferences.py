import torch
import torch.nn as nn
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

# Function to perform inference on a model with unique input for each model
def perform_inference(process_id, results_list):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load the model to the appropriate device
    model = SimpleModel().to(device)
    model.eval()

    # Generate a unique random input for each model
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

# Main function to manage concurrent inference
def main():
    manager = Manager()
    results_list = manager.list()  # Shared list to collect results

    # Use Pool to manage concurrent processes with a limit on active processes
    num_processes = 10  # Limit number of concurrent processes
    total_models = 50   # Total models to perform inference on

    with Pool(processes=num_processes) as pool:
        pool.starmap(perform_inference, [(i, results_list) for i in range(total_models)])

    # Display inference results
    print("\nAll inference tasks completed.")
    for process_id, output in results_list:
        print(f"Process {process_id} output: {output}")

if __name__ == "__main__":
    main()

