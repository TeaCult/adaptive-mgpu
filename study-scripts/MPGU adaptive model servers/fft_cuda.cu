#include <iostream>
#include <cufft.h>
#include <curand.h>
#include <cuda_runtime.h>

#define N 100000  // Number of data points in each FFT

// Function to check CUDA errors
void checkCudaError(cudaError_t result, const char* errorMessage) {
    if (result != cudaSuccess) {
        std::cerr << errorMessage << " Error: " << cudaGetErrorString(result) << std::endl;
        exit(1);
    }
}

// Function to check cuFFT errors
void checkCuFFTError(cufftResult result, const char* errorMessage) {
    if (result != CUFFT_SUCCESS) {
        std::cerr << errorMessage << " Error: " << result << std::endl;
        exit(1);
    }
}

// Function to check cuRAND errors
void checkCurandError(curandStatus_t result, const char* errorMessage) {
    if (result != CURAND_STATUS_SUCCESS) {
        std::cerr << errorMessage << " Error: " << result << std::endl;
        exit(1);
    }
}

int main(int argc, char* argv[]) {
    // Number of FFTs to perform
    int r = 1;
    if (argc > 1) {
        r = atoi(argv[1]);
    }

    // Allocate device memory for the random data
    cufftComplex* d_data;
    checkCudaError(cudaMalloc(&d_data, sizeof(cufftComplex) * N), "Failed to allocate device memory");

    // Set up the cuRAND generator to fill the array with random values
    curandGenerator_t gen;
    checkCurandError(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT), "Failed to create cuRAND generator");
    checkCurandError(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL), "Failed to set generator seed");

    // cuFFT plan
    cufftHandle plan;
    checkCuFFTError(cufftPlan1d(&plan, N, CUFFT_C2C, 1), "Failed to create cuFFT plan");

    for (int i = 0; i < r; ++i) {
        // Generate random numbers
        checkCurandError(curandGenerateUniform(gen, (float*)d_data, 2 * N), "Failed to generate random numbers");

        // Perform FFT
        checkCuFFTError(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD), "Failed to execute FFT");
        cudaDeviceSynchronize();  // Ensure each FFT finishes before the next iteration

        std::cout << "Completed FFT " << i + 1 << " of " << r << std::endl;
    }

    // Clean up
    cufftDestroy(plan);
    curandDestroyGenerator(gen);
    cudaFree(d_data);

    std::cout << "Completed " << r << " FFTs on " << N << " data points." << std::endl;
    return 0;
}
