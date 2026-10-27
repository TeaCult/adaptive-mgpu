from flask import Flask, request, jsonify
import torch
import time
import logging

app = Flask(__name__)

# Configure logging to output to stdout

@app.route('/multiply', methods=['POST'])
def multiply_matrices():
    try:
        print('process is started')
        # Start timing for data receipt
        receive_start = time.time()

        # Parse JSON data
        data = request.get_json()
        matrix_a = data['matrix_a']
        matrix_b = data['matrix_b']

        # End timing for data receipt
        receive_end = time.time()

        # Convert input matrices to PyTorch tensors
        tensor_a = torch.tensor(matrix_a, dtype=torch.float32, device='cuda')
        tensor_b = torch.tensor(matrix_b, dtype=torch.float32, device='cuda')

        # Ensure matrices are 1000x1000
        if tensor_a.shape != (1000, 1000) or tensor_b.shape != (1000, 1000):
            return jsonify({'error': 'Matrices must be 1000x1000'}), 400

        # Start timing for processing
        process_start = time.time()

        # Perform matrix multiplication
        result = torch.mm(tensor_a, tensor_b)

        # End timing for processing
        process_end = time.time()

        # Convert result back to CPU and list format for JSON serialization
        result_list = result.cpu().tolist()

        # Calculate timing information
        receive_time = (receive_end - receive_start) * 1000  # in ms
        process_time = (process_end - process_start) * 1000  # in ms
        print('data_reciept_time_ms',receive_time)
        print('processing_time_ms',process_time)
        return jsonify({
            'result': result_list,
            'timing': {
                'data_receipt_time_ms': receive_time,
                'processing_time_ms': process_time
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
