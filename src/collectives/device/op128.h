/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef OP128_H_
#define OP128_H_ 

inline __device__ void load128(const uint64_t* ptr, uint64_t &v0, uint64_t &v1) {
  asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];"
      : "=l"(v0), "=l"(v1) : "l"(ptr));
}

inline __device__ void store128(uint64_t* ptr, uint64_t v0, uint64_t v1) {
  asm volatile("st.volatile.global.v2.u64 [%2], {%0,%1};"
      :: "l"(v0), "l"(v1), "l"(ptr));
}

inline __device__ uint64_t* shmemCvtPtr(volatile uint64_t* shmemGenericPtr) {
  uint64_t* shmemAsmPtr;
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(shmemAsmPtr) : "l"(shmemGenericPtr));
  return shmemAsmPtr;
}

inline __device__ void loadShmem128(uint64_t* shmemAsmPtr, uint64_t &v0, uint64_t &v1) {
  asm volatile("ld.volatile.shared.v2.u64 {%0,%1}, [%2];"
      : "=l"(v0), "=l"(v1) : "l"(shmemAsmPtr));
}

inline __device__ void storeShmem128(uint64_t* shmemAsmPtr, uint64_t v0, uint64_t v1) {
  asm volatile("st.volatile.shared.v2.u64 [%2], {%0,%1};"
      :: "l"(v0), "l"(v1), "l"(shmemAsmPtr));
}

inline __device__ void copyToShmem128(const uint64_t* srcPtr, uint64_t* shmemAsmPtr) {
  asm volatile(
      "{"
      "  .reg .u64 u0;"
      "  .reg .u64 u1;"
      "  ld.volatile.global.v2.u64 {u0,u1}, [%0];"
      "  st.volatile.shared.v2.u64 [%1], {u0,u1};"
      "}"
      :: "l"(srcPtr), "l"(shmemAsmPtr));
}

inline __device__ void copyFromShmem128(uint64_t* shmemAsmPtr, uint64_t* dstPtr) {
  asm volatile("{"
      "  .reg .u64 u0;"
      "  .reg .u64 u1;"
      "  ld.volatile.shared.v2.u64 {u0,u1}, [%0];"
      "  st.volatile.global.v2.u64 [%1], {u0,u1};"
      "}"
      :: "l"(shmemAsmPtr), "l"(dstPtr));
}
#endif
