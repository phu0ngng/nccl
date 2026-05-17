// Launcher lives in a .cu so nvcc instantiates and registers the kernel.

#include "device/indirect_memcpy.cuh"

namespace nccl_ep {
namespace indirect_memcpy {

cudaError_t launch_indirect_memcpy_tma(
    void* dst,
    const void* src,
    const int32_t* n_rows_ptr,
    int max_rows,
    int row_bytes,
    int num_sms,
    cudaStream_t stream) {
  if (max_rows <= 0 || row_bytes <= 0) return cudaSuccess;
  if ((row_bytes & 0xF) != 0) return cudaErrorInvalidValue;

  constexpr int BLOCK_THREADS = 128;
  constexpr int ROWS_PER_BLOCK = 4;
  constexpr int NB = 2;
  const size_t slab_bytes = static_cast<size_t>(ROWS_PER_BLOCK) * row_bytes;
  // cp.async.bulk caps single-transfer size at 128KB.
  if (slab_bytes > (128u * 1024u)) return cudaErrorInvalidValue;
  const size_t smem_bytes = NB * slab_bytes;

  // Opt-in once to the device's full dyn-smem budget (minus reserved).
  static thread_local bool opt_in_done = false;
  if (!opt_in_done) {
    int dev = 0;
    cudaGetDevice(&dev);
    int max_smem = 0;
    cudaDeviceGetAttribute(&max_smem,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    if (max_smem <= 0) max_smem = 200 * 1024;
    int reserved = 0;
    cudaDeviceGetAttribute(&reserved,
                           cudaDevAttrReservedSharedMemoryPerBlock, dev);
    cudaError_t e = cudaFuncSetAttribute(
        reinterpret_cast<const void*>(
            &indirect_memcpy_tma_kernel<BLOCK_THREADS, ROWS_PER_BLOCK>),
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        max_smem - reserved - 4096);
    if (e != cudaSuccess) return e;
    opt_in_done = true;
  }
  if (smem_bytes > static_cast<size_t>(228u * 1024u)) {
    return cudaErrorInvalidValue;
  }

  const int blocks_needed = (max_rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
  const int blocks = num_sms > 0
      ? (blocks_needed < num_sms ? blocks_needed : num_sms)
      : blocks_needed;

  void* args[] = {
      &dst, const_cast<const void**>(&src),
      const_cast<const int32_t**>(&n_rows_ptr), &row_bytes,
  };
  return cudaLaunchKernel(
      reinterpret_cast<const void*>(
          &indirect_memcpy_tma_kernel<BLOCK_THREADS, ROWS_PER_BLOCK>),
      dim3(blocks), dim3(BLOCK_THREADS), args, smem_bytes, stream);
}

}  // namespace indirect_memcpy
}  // namespace nccl_ep
