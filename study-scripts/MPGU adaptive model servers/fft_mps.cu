#include <iostream>
#include <cufft.h>
#include <curand.h>
#include <cuda_runtime.h>

#define N 100000  // Number of data points in each FFT

// Error-checking functions as defined before
void checkCudaError(cudaError_t result, const char* errorMessage);
void checkCuFFTError(cufftResult result, const char* errorMessage);
void checkCurandError(curandStatus_t result, const char* errorMessage);

int main(int argc, char* argv[]) {
    int r = 1;
    if (argc > 1) {
        r = atoi(argv[1]);
    }

    cufftComplex* d_data;
    checkCudaError(cudaMalloc(&d_data, sizeof(cufftComplex) * N), "Failed to allocate device memory");

    curandGenerator_t gen;
    checkCurandError(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT), "Failed to create cuRAND generator");
    checkCurandError(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL), "Failed to set generator seed");

    cufftHandle plan;
    checkCuFFTError(cufftPlan1d(&plan, N, CUFFT_C2C, 1), "Failed to create cuFFT plan");

    // Create a CUDA stream
    cudaStream_t stream;
    checkCudaError(cudaStreamCreate(&stream), "Failed to create CUDA stream");

    for (int i = 0; i < r; ++i) {
        // Generate random data in the specified stream
        checkCurandError(curandSetStream(gen, stream), "Failed to set stream for cuRAND");
        checkCurandError(curandGenerateUniform(gen, (float*)d_data, 2 * N), "Failed to generate random numbers");

        // Execute FFT in the specified stream
        checkCuFFTError(cufftSetStream(plan, stream), "Failed to set stream for cuFFT");
        checkCuFFTError(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD), "Failed to execute FFT");

        checkCudaError(cudaStreamSynchronize(stream), "Stream synchronization failed");
        std::cout << "Completed FFT " << i + 1 << " of " << r << std::endl;
    }

    // Clean up
    cudaStreamDestroy(stream);
    cufftDestroy(plan);
    curandDestroyGenerator(gen);
    cudaFree(d_data);

    std::cout << "Completed " << r << " FFTs on " << N << " data points." << std::endl;
    return 0;
}
