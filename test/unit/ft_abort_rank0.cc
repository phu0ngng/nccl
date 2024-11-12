#include <iostream>
#include <thread>
#include <chrono>
#include <nccl.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <sys/wait.h>

#define NUM_GPUS 2
#define NUM_ELEMENTS 1000000
#define NUM_ALL_REDUCES 32

void checkCudaError(cudaError_t result) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void checkNcclError(ncclResult_t result) {
    if (result != ncclSuccess) {
        std::cerr << "NCCL Error: " << ncclGetErrorString(result) << std::endl;
        //exit(EXIT_FAILURE);
    }
}

void runAllReduceProcess(int rank, int num_devices, ncclUniqueId id) {
    int count;
    checkCudaError(cudaGetDeviceCount(&count));
    if (count < num_devices) {
        if (rank == 0) std::cout << "Not enough ndevs: detected " << count << " devs, but required " << num_devices << " [SKIP]" << std::endl;
        return;
    }
    checkCudaError(cudaSetDevice(rank));

    cudaStream_t stream;
    ncclComm_t comm;
    checkCudaError(cudaStreamCreate(&stream));
    checkNcclError(ncclCommInitRank(&comm, num_devices, id, rank));

    float* device_tensor;
    float* host_tensor;
    checkCudaError(cudaMalloc(&device_tensor, NUM_ELEMENTS * sizeof(float)));
    checkCudaError(cudaMallocHost(&host_tensor, NUM_ELEMENTS * sizeof(float)));
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        host_tensor[i] = 1.0f;
    }
    checkCudaError(cudaMemcpy(device_tensor, host_tensor, NUM_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));

    // for some reason this all-reduce is necessary
    checkNcclError(ncclAllReduce(device_tensor, device_tensor, NUM_ELEMENTS, ncclFloat32, ncclSum, comm, stream));

    checkCudaError(cudaStreamSynchronize(stream));
    checkCudaError(cudaDeviceSynchronize());

    // Perform multiple all-reduce operations
    if (rank == 0) {
        for (int i = 0; i < NUM_ALL_REDUCES; i++) {
            checkNcclError(ncclAllReduce(device_tensor, device_tensor, NUM_ELEMENTS, ncclFloat32, ncclSum, comm, stream));
        }
    }

    std::cout << "ABORT " << rank << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    checkNcclError(ncclCommAbort(comm));
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;


    checkCudaError(cudaMemcpy(host_tensor, device_tensor, NUM_ELEMENTS * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaError(cudaDeviceSynchronize());
    // Clean up
    checkCudaError(cudaFree(device_tensor));
    checkCudaError(cudaStreamDestroy(stream));
    //checkNcclError(ncclCommDestroy(comm));

    checkCudaError(cudaDeviceSynchronize());
    std::cout << "Process " << rank << " completed all-reduce operations successfully!" << std::endl;
    return;
}

int main() {
    ncclUniqueId id;
    checkNcclError(ncclGetUniqueId(&id));

    // Fork the process for each GPU
    pid_t pids[NUM_GPUS];
    for (int i = 0; i < NUM_GPUS; i++) {
        pids[i] = fork();
        if (pids[i] == 0) {
            runAllReduceProcess(i, NUM_GPUS, id);
            exit(0);
        } else if (pids[i] < 0) {
            std::cerr << "Failed to fork process for GPU " << i << std::endl;
            return EXIT_FAILURE;
        }
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        waitpid(pids[i], NULL, 0);
    }

    return 0;
}
