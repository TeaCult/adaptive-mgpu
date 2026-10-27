from flask import Flask, request, jsonify
import torch
import time
import logging

app = Flask(__name__)

# Configure logging to output to stdout
logging.basicConfig(level=logging.DEBUG)

@app.route('/multiply', methods=['POST'])
def multiply_matrices():
    try:
        logging.info('Process started')
        
        # Start timing for data receipt
        receive_start = time.time()

        # Parse JSON data
        data = request.get_json()
        matrix_a = data['matrix_a']
        matrix_b = data['matrix_b']

        # End timing for data receipt
        receive_end = time.time()

        # Convert input matrices to PyTorch tensors on the GPU
        gpu_offload_start = time.time()
        tensor_a = torch.tensor(matrix_a, dtype=torch.float32, device='cuda')
        tensor_b = torch.tensor(matrix_b, dtype=torch.float32, device='cuda')
        gpu_offload_end = time.time()

        # Ensure matrices are 1000x1000
        if tensor_a.shape != (1000, 1000) or tensor_b.shape != (1000, 1000):
            return jsonify({'error': 'Matrices must be 1000x1000'}), 400

        # Start timing for GPU processing
        process_start = time.time()

        # Perform matrix multiplication on the GPU
        result = torch.mm(tensor_a, tensor_b)

        # End timing for GPU processing
        process_end = time.time()

        # Transfer the result back to the CPU for JSON serialization
        cpu_offload_start = time.time()
        result_list = result.cpu().tolist()  # Moves result from GPU to CPU
        cpu_offload_end = time.time()

        # Calculate timing information
        receive_time = (receive_end - receive_start) * 1000  # Data receipt time in ms
        gpu_offload_time = (gpu_offload_end - gpu_offload_start) * 1000  # GPU offload time in ms
        process_time = (process_end - process_start) * 1000  # GPU processing time in ms
        cpu_offload_time = (cpu_offload_end - cpu_offload_start) * 1000  # CPU offload time in ms

        # Print timing information for debugging
        logging.info(f'Data receipt time: {receive_time:.2f} ms')
        logging.info(f'GPU offload time: {gpu_offload_time:.2f} ms')
        logging.info(f'Processing time on GPU: {process_time:.2f} ms')
        logging.info(f'CPU offload time: {cpu_offload_time:.2f} ms')

        # Return timing information as part of the response
        return jsonify({
            'result': result_list,
            'timing': {
                'data_receipt_time_ms': receive_time,
                'gpu_offload_time_ms': gpu_offload_time,
                'processing_time_ms': process_time,
                'cpu_offload_time_ms': cpu_offload_time
            }
        })
    except Exception as e:
        logging.error(f'Error: {e}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
