#include <nccl.h>
#include <iostream>
#include <cstdlib>
#include <cstring>

// Manual test framework - simple assertion macros
#define ASSERT_EQ(val1, val2) \
    do { \
        auto v1 = (val1); \
        auto v2 = (val2); \
        if (v1 != v2) { \
            std::cerr << "ASSERTION FAILED at " << __FILE__ << ":" << __LINE__ \
                      << " - Expected " << #val1 << " == " << #val2 \
                      << " but got " << v1 << " != " << v2 << std::endl; \
            exit(1); \
        } \
    } while(0)

#define ASSERT_NE(val1, val2) \
    do { \
        auto v1 = (val1); \
        auto v2 = (val2); \
        if (v1 == v2) { \
            std::cerr << "ASSERTION FAILED at " << __FILE__ << ":" << __LINE__ \
                      << " - Expected " << #val1 << " != " << #val2 \
                      << " but got " << v1 << " == " << v2 << std::endl; \
            exit(1); \
        } \
    } while(0)

ncclComm_t* test_setup(int* nVis) {
    (void) setenv("NCCL_P2P_USE_CUDA_MEMCPY", "1", 1);
    (void) setenv("NCCL_CHECK_POINTERS", "1", 0);

    cudaError_t cuda_result = cudaGetDeviceCount(nVis);
    if (cuda_result != cudaSuccess) {
        std::cerr << "SETUP FAILED: cudaGetDeviceCount returned " << cuda_result << std::endl;
        exit(1);
    }

    ncclComm_t* comms = (ncclComm_t*)calloc(sizeof(ncclComm_t), *nVis);
    if (comms == nullptr) {
        std::cerr << "SETUP FAILED: calloc returned nullptr" << std::endl;
        exit(1);
    }

    ncclResult_t nccl_result = ncclCommInitAll(comms, *nVis, NULL);
    if (nccl_result != ncclSuccess) {
        std::cerr << "SETUP FAILED: ncclCommInitAll returned " << nccl_result << std::endl;
        free(comms);
        exit(1);
    }

    return comms;
}

void test_teardown(ncclComm_t* comms) {
    if (comms != nullptr) {
        free(comms);
    }
}

void test_useMemcpy_skip_registration(ncclComm_t* comms) {
    // Test that registration is skipped when P2P_USE_CUDA_MEMCPY is enabled
    const int size = 1024;
    void* buff = nullptr;
    ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&buff, size));
    ASSERT_NE((void * )nullptr, buff);

    void* handle;
    ASSERT_EQ(ncclSuccess, ncclCommRegister(comms[0], buff, size, &handle));
    // When useMemcpy is enabled, registration should be skipped and handle should be NULL
    ASSERT_EQ((void *)nullptr, handle);

    // Test deregistration with NULL handle (should succeed)
    ASSERT_EQ(ncclSuccess, ncclCommDeregister(comms[0], handle));

    ASSERT_EQ(cudaSuccess, cudaFree(buff));
}

// Main function to run the test
int main(int argc, char** argv) {
    std::cout << "Running register_memcpyTest..." << std::endl;

    int nVis = 0;
    ncclComm_t* comms = nullptr;

    std::cout << "Setting up test..." << std::endl;
    comms = test_setup(&nVis);

    std::cout << "Running test_useMemcpy_skip_registration..." << std::endl;
    test_useMemcpy_skip_registration(comms);

    std::cout << "Cleaning up test..." << std::endl;
    test_teardown(comms);

    std::cout << "All tests PASSED!" << std::endl;
    return 0;
}

// EOF

