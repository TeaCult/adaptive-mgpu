from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.optim as optim
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.CRITICAL)

app = Flask(__name__)

# Define a simple neural network model
class SimpleModel(nn.Module):
    def __init__(self, input_size=10, output_size=10):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Data storage
input_data_buffer = []
output_data_buffer = []

# Batch update settings
initial_batch_size = 100
second_stage_batch_size = 50
final_stage_batch_size = 10

update_stage = 0  # Tracks which stage of updating we're in

def update_model(batch_inputs, batch_outputs):
    model.train()
    optimizer.zero_grad()
    predictions = model(batch_inputs)
    loss = criterion(predictions, batch_outputs)
    loss.backward()
    optimizer.step()
    print(f"\033[91mModel updated with batch. Loss: {loss.item()}\033[0m")

@app.route('/predict', methods=['POST'])
def predict():
    global update_stage

    # Get data from the request
    data = request.json
    input_data = torch.tensor(data['input'], dtype=torch.float32).unsqueeze(0)
    actual_output = torch.tensor(data['actual_output'], dtype=torch.float32).unsqueeze(0)

    # Get model prediction
    model.eval()
    with torch.no_grad():
        prediction = model(input_data)
    prediction_response = prediction.squeeze().tolist()

    # Append new data to buffers
    input_data_buffer.append(input_data)
    output_data_buffer.append(actual_output)

    # Determine batch size based on the current stage
    if update_stage == 0:
        batch_size = initial_batch_size
    elif update_stage == 1:
        batch_size = second_stage_batch_size
    else:
        batch_size = final_stage_batch_size

    # Check if we have enough data to perform a model update
    if len(input_data_buffer) >= batch_size:
        # Stack buffered data into a single batch tensor
        batch_inputs = torch.cat(input_data_buffer)
        batch_outputs = torch.cat(output_data_buffer)

        # Update model with batch
        update_model(batch_inputs, batch_outputs)

        # Clear buffers after updating
        input_data_buffer.clear()
        output_data_buffer.clear()

        # Move to the next stage if appropriate
        if update_stage == 0 and batch_size == initial_batch_size:
            update_stage = 1
        elif update_stage == 1 and batch_size == second_stage_batch_size:
            update_stage = 2

    # Return the prediction as JSON
    return jsonify({'prediction': prediction_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
