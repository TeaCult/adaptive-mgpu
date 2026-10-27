from flask import Flask, request, jsonify
import torch
import threading
import time
from queue import Queue

app = Flask(__name__)

# Queue to store incoming requests
request_queue = Queue()

# Batch processing interval (in seconds)
BATCH_INTERVAL = 1.0

# Function to perform batched matrix multiplication
def batch_processor():
    while True:
        time.sleep(BATCH_INTERVAL)  # Wait for the batching interval

        # Collect all requests currently in the queue
        batch = []
        while not request_queue.empty():
            req_data, result_queue = request_queue.get()
            batch.append((req_data, result_queue))

        if batch:
            # Perform batch processing
            matrix_a_list = []
            matrix_b_list = []

            # Collect matrices for batched multiplication
            for req_data, _ in batch:
                matrix_a_list.append(req_data['matrix_a'])
                matrix_b_list.append(req_data['matrix_b'])

            # Convert lists to batched tensors on the GPU
            tensor_a = torch.tensor(matrix_a_list, dtype=torch.float32, device='cuda')
            tensor_b = torch.tensor(matrix_b_list, dtype=torch.float32, device='cuda')

            # Perform batched matrix multiplication
            with torch.no_grad():
                result_tensor = torch.bmm(tensor_a, tensor_b)

            # Send each result back to the respective request
            results = result_tensor.cpu().tolist()
            for i, (_, result_queue) in enumerate(batch):
                result_queue.put(results[i])  # Place result in the queue for the client

# Start the batch processor in a background thread
threading.Thread(target=batch_processor, daemon=True).start()

# Flask endpoint to handle matrix multiplication requests
@app.route('/multiply', methods=['POST'])
def multiply_matrices():
    try:
        # Parse JSON data from request
        data = request.get_json()
        matrix_a = data['matrix_a']
        matrix_b = data['matrix_b']

        # Ensure matrices are 1000x1000
        if len(matrix_a) != 1000 or len(matrix_b) != 1000 or len(matrix_a[0]) != 1000 or len(matrix_b[0]) != 1000:
            return jsonify({'error': 'Matrices must be 1000x1000'}), 400

        # Queue for receiving result
        result_queue = Queue()

        # Add request to batch processing queue
        request_queue.put((data, result_queue))

        # Wait for result from batch processor
        result = result_queue.get()

        # Return the result to the client
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
