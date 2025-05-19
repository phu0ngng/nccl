#include "ncclCommon_test.cuh"

#include <fstream>
#include <string>

class ncclMemAlloc_test : public ncclOutputTest { 
    protected:
    virtual void SetUp() override {
        ncclOutputTest::SetUp();
        overrideEnvVariable("NCCL_DEBUG_SUBSYS", "ALL");  
    }      
};
TEST_F(ncclMemAlloc_test, basic) {
    int dev = 0;
    ncclResult_t ncclErr = ncclSuccess;
    
    ASSERT_EQ(cudaSuccess, cudaSetDevice(dev));
    ASSERT_EQ(cudaSuccess, cudaFree(0));
    
    cudaDeviceProp prop;
    ASSERT_EQ(cudaSuccess, cudaGetDeviceProperties(&prop, dev));
    const size_t totalMem = prop.totalGlobalMem;
    
    std::vector<void*> allocations;
    size_t chunkSize = totalMem / 2;  // Start with 50% of total memory
    const size_t minChunk = 512 << 20;  // 512 MB minimum
    size_t allocated = 0;
    
    while (chunkSize >= minChunk) {
        void* ptr = nullptr;
        overrideEnvVariable("NCCL_DEBUG", "INFO");
        ncclErr = ncclMemAlloc(&ptr, chunkSize);
        printf("Attempting to allocate GPU memory %zu MB (Total: %zu MB)\n", chunkSize >> 20, totalMem >> 20);
        if (ncclErr == ncclSuccess) {   
            allocations.push_back(ptr);
            allocated += chunkSize;
            // Optimize next chunk size based on remaining memory
            const size_t remaining = totalMem - allocated;
            chunkSize = (remaining >= chunkSize) ? chunkSize : remaining;
        } else {
            // Allocation failed finally
            ASSERT_EQ(ncclUnhandledCudaError, ncclErr);
            verifyResult(".*out of memory.*");
            return;
        }
        
    }
    
    for (void* ptr : allocations) ncclMemFree(ptr);

}