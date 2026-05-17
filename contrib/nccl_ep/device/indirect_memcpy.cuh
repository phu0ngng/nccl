// Indirect TMA memcpy: copies `*n_rows_ptr` rows from src to dst on-device.

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace nccl_ep {
namespace indirect_memcpy {

namespace detail {

__device__ __forceinline__ unsigned cvta_to_shared(const void* p) {
#if defined(__CUDA_ARCH__)
  return static_cast<unsigned>(__cvta_generic_to_shared(const_cast<void*>(p)));
#else
  (void)p;
  return 0u;
#endif
}

}  // namespace detail

// Two-slab pipeline per block: NB=2 G2S loads in flight while one S2G
// store is in the bulk_group. Smem = NB · ROWS_PER_BLOCK · row_bytes.
template <int BLOCK_THREADS = 128, int ROWS_PER_BLOCK = 4>
__global__ void indirect_memcpy_tma_kernel(
    void* __restrict__ dst,
    const void* __restrict__ src,
    const int32_t* __restrict__ n_rows_ptr,
    int row_bytes) {
#if (__CUDA_ARCH__ >= 900)
  constexpr int NB = 2;
  extern __shared__ __align__(16) char smem_buf[];
  __shared__ __align__(8) uint64_t mbar[NB];

  if (threadIdx.x < NB) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n"
                 ::"r"(detail::cvta_to_shared(&mbar[threadIdx.x])));
  }
  __syncthreads();

  const int n_rows = *n_rows_ptr;
  if (n_rows <= 0) return;

  const int slab_size_bytes = ROWS_PER_BLOCK * row_bytes;
  const int slab_stride = gridDim.x * ROWS_PER_BLOCK;

  __shared__ int s_issued_slab[NB];
  __shared__ unsigned s_wait_parity[NB];

  // Prime: issue the first NB G2S loads for this block's slab series.
  if (threadIdx.x == 0) {
    for (int b = 0; b < NB; ++b) {
      const int slab_start = blockIdx.x * ROWS_PER_BLOCK + b * slab_stride;
      s_issued_slab[b] = slab_start;
      s_wait_parity[b] = 0;
      if (slab_start < n_rows) {
        const int rows = min(ROWS_PER_BLOCK, n_rows - slab_start);
        const int bytes = rows * row_bytes;
        const char* src_slab =
            reinterpret_cast<const char*>(src) +
            static_cast<size_t>(slab_start) * row_bytes;
        asm volatile(
            "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
            ::"r"(detail::cvta_to_shared(&mbar[b])), "r"(bytes));
        asm volatile(
            "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
            "[%0], [%1], %2, [%3];\n"
            ::"r"(detail::cvta_to_shared(smem_buf + b * slab_size_bytes)),
              "l"(src_slab),
              "r"(bytes),
              "r"(detail::cvta_to_shared(&mbar[b])));
      }
    }
  }
  __syncthreads();

  int b = 0;
  while (true) {
    const int slab_start = s_issued_slab[b];
    if (slab_start >= n_rows) break;
    const int rows = min(ROWS_PER_BLOCK, n_rows - slab_start);
    const int bytes = rows * row_bytes;
    char* dst_slab =
        reinterpret_cast<char*>(dst) +
        static_cast<size_t>(slab_start) * row_bytes;

    // Wait for buffer b's G2S to finish (parity-flipping mbarrier).
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "WAITDB_%=:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "@!p bra WAITDB_%=;\n"
        "}\n"
        ::"r"(detail::cvta_to_shared(&mbar[b])), "r"(s_wait_parity[b]));

    if (threadIdx.x == 0) {
      s_wait_parity[b] ^= 1;
      asm volatile(
          "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
          ::"l"(dst_slab),
            "r"(detail::cvta_to_shared(smem_buf + b * slab_size_bytes)),
            "r"(bytes));
      asm volatile("cp.async.bulk.commit_group;\n" ::);

      // Issue next load into buffer b (NB slabs ahead).
      const int next = slab_start + NB * slab_stride;
      s_issued_slab[b] = next;
      if (next < n_rows) {
        const int next_rows = min(ROWS_PER_BLOCK, n_rows - next);
        const int next_bytes = next_rows * row_bytes;
        const char* next_src_slab =
            reinterpret_cast<const char*>(src) +
            static_cast<size_t>(next) * row_bytes;
        asm volatile(
            "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
            ::"r"(detail::cvta_to_shared(&mbar[b])), "r"(next_bytes));
        asm volatile(
            "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
            "[%0], [%1], %2, [%3];\n"
            ::"r"(detail::cvta_to_shared(smem_buf + b * slab_size_bytes)),
              "l"(next_src_slab),
              "r"(next_bytes),
              "r"(detail::cvta_to_shared(&mbar[b])));
      }

      // Throttle outstanding bulk_group stores to NB-1.
      asm volatile("cp.async.bulk.wait_group.read %0;\n" ::"n"(NB - 1));
    }
    __syncthreads();
    b = (b + 1) % NB;
  }

  if (threadIdx.x == 0) {
    asm volatile("cp.async.bulk.wait_group.read 0;\n" ::);
  }
  __syncthreads();
#else
  (void)dst;
  (void)src;
  (void)n_rows_ptr;
  (void)row_bytes;
#endif
}

// Host-side launcher; `num_sms` should match the dispatch SM budget.
cudaError_t launch_indirect_memcpy_tma(
    void* dst,
    const void* src,
    const int32_t* n_rows_ptr,
    int max_rows,
    int row_bytes,
    int num_sms,
    cudaStream_t stream);

}  // namespace indirect_memcpy
}  // namespace nccl_ep
