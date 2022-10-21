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

template<typename T>
inline __device__ void loadShmemMisaligned128(T *ptr, uint64_t &v0, uint64_t &v1) {
  union {
    uint32_t tmp4[4];
    uint64_t tmp8[2];
  };
  if(sizeof(T) < 4) {
    uint32_t *ptr4 = reinterpret_cast<uint32_t*>(reinterpret_cast<uintptr_t>(ptr) & -uintptr_t(4));
    #pragma unroll
    for(int e=0; e < 4; e++) {
      // Produce 4 bytes of sub-register type by reading 2 4-byte
      // aligned values and shifting.
      uint32_t lo, hi;
      asm("ld.shared.b32 %0,[%1];" : "=r"(lo) : "l"(ptr4+e+0));
      asm("ld.shared.b32 %0,[%1];" : "=r"(hi) : "l"(ptr4+e+1));
      tmp4[e] = __funnelshift_r(lo, hi, 8*(int(reinterpret_cast<uintptr_t>(ptr))%4));
    }
  }
  else if(sizeof(T) == 4) {
    #pragma unroll
    for(int e=0; e < 4; e++)
      asm("ld.shared.b32 %0,[%1];" : "=r"(tmp4[e]) : "l"(ptr+e));
  }
  else /*sizeof(T)==8*/ {
    #pragma unroll
    for(int e=0; e < 2; e++)
      asm("ld.shared.b64 %0,[%1];" : "=l"(tmp8[e]) : "l"(ptr+e));
  }
  v0 = tmp8[0];
  v1 = tmp8[1];
}


template<typename T>
__device__ __forceinline__ uint32_t cvta_to_shared(T* ptr) {
  uintptr_t ans;
  asm("cvta.to.shared.u64 %0, %1;" : "=l"(ans) : "l"(ptr));
  return uint32_t(ans);
}
template<typename T>
__device__ __forceinline__ uintptr_t cvta_to_global(T* ptr) {
  uintptr_t ans;
  asm("cvta.to.global.u64 %0, %1;" : "=l"(ans) : "l"(ptr));
  return ans;
}

template<typename T>
__device__ __forceinline__ T* cvta_from_shared(uint32_t shptr) {
  T* ans;
  asm("cvta.shared.u64 %0, %1;" : "=l"(ans) : "l"(uint64_t(shptr)));
  return ans;
}
template<typename T>
__device__ __forceinline__ T* cvta_from_global(uintptr_t gptr) {
  T* ans;
  asm("cvta.global.u64 %0, %1;" : "=l"(ans) : "l"(gptr));
  return ans;
}

template<int Size>
union BytePack;
template<>
union BytePack<1> {
  uint8_t u8[1];
};
template<>
union BytePack<2> {
  BytePack<1> half[2];
  uint8_t u8[2];
  uint16_t u16[1];
};
template<>
union BytePack<4> {
  BytePack<2> half[2];
  uint8_t u8[4];
  uint16_t u16[2];
  uint32_t u32[1];
};
template<>
union BytePack<8> {
  BytePack<4> half[2];
  uint8_t u8[8];
  uint16_t u16[4];
  uint32_t u32[2];
  uint64_t u64[1];
};
template<>
union BytePack<16> {
  alignas(16) BytePack<8> half[2];
  uint8_t u8[16];
  uint16_t u16[8];
  uint32_t u32[4];
  uint64_t u64[2];
  ulong2 ul2;
};

template<typename T>
__device__ __forceinline__ BytePack<sizeof(T)> toPack(T value)  {
  union { BytePack<sizeof(T)> p; T v; };
  v = value;
  return p;
}
template<typename T>
__device__ __forceinline__ T fromPack(BytePack<sizeof(T)> pack)  {
  union { BytePack<sizeof(T)> p; T v; };
  p = pack;
  return v;
}

template<int Size>
__device__ __forceinline__ BytePack<Size> ld_global(uintptr_t addr);
template<>
__device__ __forceinline__ BytePack<1> ld_global<1>(uintptr_t addr) {
  uint32_t tmp;
  asm("ld.volatile.global.b8 %0, [%1];" : "=r"(tmp) : "l"(addr));
  BytePack<1> ans;
  ans.u8[0] = (uint8_t)tmp;
  return ans;
}
template<>
__device__ __forceinline__ BytePack<2> ld_global<2>(uintptr_t addr) {
  BytePack<2> ans;
  asm("ld.volatile.global.b16 %0, [%1];" : "=h"(ans.u16[0]) : "l"(addr));
  return ans;
}
template<>
__device__ __forceinline__ BytePack<4> ld_global<4>(uintptr_t addr) {
  BytePack<4> ans;
  asm("ld.volatile.global.b32 %0, [%1];" : "=r"(ans.u32[0]) : "l"(addr));
  return ans;
}
template<>
__device__ __forceinline__ BytePack<8> ld_global<8>(uintptr_t addr) {
  BytePack<8> ans;
  asm("ld.volatile.global.b64 %0, [%1];" : "=l"(ans.u64[0]) : "l"(addr));
  return ans;
}
template<>
__device__ __forceinline__ BytePack<16> ld_global<16>(uintptr_t addr) {
  BytePack<16> ans;
  asm("ld.volatile.global.v2.b64 {%0, %1}, [%2];" : "=l"(ans.u64[0]), "=l"(ans.u64[1]) : "l"(addr));
  return ans;
}

template<int Size>
__device__ __forceinline__ void st_global(uintptr_t addr, BytePack<Size> value);
template<>
__device__ __forceinline__ void st_global(uintptr_t addr, BytePack<1> value) {
  asm("st.volatile.global.b8 [%0], %1;" :: "l"(addr), "r"((uint32_t)value.u8[0]) : "memory");
}
template<>
__device__ __forceinline__ void st_global(uintptr_t addr, BytePack<2> value) {
  asm("st.volatile.global.b16 [%0], %1;" :: "l"(addr), "h"(value.u16[0]) : "memory");
}
template<>
__device__ __forceinline__ void st_global(uintptr_t addr, BytePack<4> value) {
  asm("st.volatile.global.b32 [%0], %1;" :: "l"(addr), "r"(value.u32[0]) : "memory");
}
template<>
__device__ __forceinline__ void st_global(uintptr_t addr, BytePack<8> value) {
  asm("st.volatile.global.b64 [%0], %1;" :: "l"(addr), "l"(value.u64[0]) : "memory");
}
template<int Size>
__device__ __forceinline__ void st_global(uintptr_t addr, BytePack<16> value) {
  asm("st.volatile.global.v2.b64 [%0], {%1, %2};" :: "l"(addr), "l"(value.u64[0]), "l"(value.u64[1]) : "memory");
}

#endif
