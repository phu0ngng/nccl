/*
 * NVIDIA_COPYRIGHT_BEGIN
 *
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 * NVIDIA_COPYRIGHT_END
 */

#if !defined(__BUILTIN_OCG_ASYNC_H)
#define __BUILTIN_OCG_ASYNC_H 1

extern "C"
{
    /*
     *     f32 ops
     */
    __device__ float1 __nv_ptx_builtin_ocg_ld_mc_add_f32(uint64_t addr);
    __device__ float2 __nv_ptx_builtin_ocg_ld_mc_add_f32x2(uint64_t addr);
    __device__ float4 __nv_ptx_builtin_ocg_ld_mc_add_f32x4(uint64_t addr);

    /*
     *     f64 ops
     */
    __device__ double __nv_ptx_builtin_ocg_ld_mc_add_f64(uint64_t addr);

    /*
     *     f16 ops
     */
    __device__ uint1 __nv_ptx_builtin_ocg_ld_mc_add_f16x2(uint64_t addr);
    __device__ uint2 __nv_ptx_builtin_ocg_ld_mc_add_f16x4(uint64_t addr);
    __device__ uint4 __nv_ptx_builtin_ocg_ld_mc_add_f16x8(uint64_t addr);

    __device__ uint1 __nv_ptx_builtin_ocg_ld_mc_min_f16x2(uint64_t addr);
    __device__ uint2 __nv_ptx_builtin_ocg_ld_mc_min_f16x4(uint64_t addr);
    __device__ uint4 __nv_ptx_builtin_ocg_ld_mc_min_f16x8(uint64_t addr);

    __device__ uint1 __nv_ptx_builtin_ocg_ld_mc_max_f16x2(uint64_t addr);
    __device__ uint2 __nv_ptx_builtin_ocg_ld_mc_max_f16x4(uint64_t addr);
    __device__ uint4 __nv_ptx_builtin_ocg_ld_mc_max_f16x8(uint64_t addr);

    __device__ uint1 __nv_ptx_builtin_ocg_ld_mc_f32add_f16x2(uint64_t addr);
    __device__ uint2 __nv_ptx_builtin_ocg_ld_mc_f32add_f16x4(uint64_t addr);
    __device__ uint4 __nv_ptx_builtin_ocg_ld_mc_f32add_f16x8(uint64_t addr);


    /*
     *     bf16 ops
     */
    __device__ uint1 __nv_ptx_builtin_ocg_ld_mc_add_bf16x2(uint64_t addr);
    __device__ uint2 __nv_ptx_builtin_ocg_ld_mc_add_bf16x4(uint64_t addr);
    __device__ uint4 __nv_ptx_builtin_ocg_ld_mc_add_bf16x8(uint64_t addr);

    __device__ uint1 __nv_ptx_builtin_ocg_ld_mc_min_bf16x2(uint64_t addr);
    __device__ uint2 __nv_ptx_builtin_ocg_ld_mc_min_bf16x4(uint64_t addr);
    __device__ uint4 __nv_ptx_builtin_ocg_ld_mc_min_bf16x8(uint64_t addr);

    __device__ uint1 __nv_ptx_builtin_ocg_ld_mc_max_bf16x2(uint64_t addr);
    __device__ uint2 __nv_ptx_builtin_ocg_ld_mc_max_bf16x4(uint64_t addr);
    __device__ uint4 __nv_ptx_builtin_ocg_ld_mc_max_bf16x8(uint64_t addr);

    __device__ uint1 __nv_ptx_builtin_ocg_ld_mc_f32add_bf16x2(uint64_t addr);
    __device__ uint2 __nv_ptx_builtin_ocg_ld_mc_f32add_bf16x4(uint64_t addr);
    __device__ uint4 __nv_ptx_builtin_ocg_ld_mc_f32add_bf16x8(uint64_t addr);


    /*
     *     u32 ops
     */
    __device__ uint32_t __nv_ptx_builtin_ocg_ld_mc_add_u32(uint64_t addr);
    __device__ uint32_t __nv_ptx_builtin_ocg_ld_mc_min_u32(uint64_t addr);
    __device__ uint32_t __nv_ptx_builtin_ocg_ld_mc_max_u32(uint64_t addr);
    __device__ uint32_t __nv_ptx_builtin_ocg_ld_mc_and_u32(uint64_t addr);
    __device__ uint32_t __nv_ptx_builtin_ocg_ld_mc_or_u32(uint64_t addr);
    __device__ uint32_t __nv_ptx_builtin_ocg_ld_mc_xor_u32(uint64_t addr);

    /*
     *     s32 ops
     */
    __device__ uint32_t __nv_ptx_builtin_ocg_ld_mc_add_s32(uint64_t addr);
    __device__ uint32_t __nv_ptx_builtin_ocg_ld_mc_min_s32(uint64_t addr);
    __device__ uint32_t __nv_ptx_builtin_ocg_ld_mc_max_s32(uint64_t addr);
    __device__ uint32_t __nv_ptx_builtin_ocg_ld_mc_and_s32(uint64_t addr);
    __device__ uint32_t __nv_ptx_builtin_ocg_ld_mc_or_s32(uint64_t addr);
    __device__ uint32_t __nv_ptx_builtin_ocg_ld_mc_xor_s32(uint64_t addr);


    /*
     *     u64 ops
     */
    __device__ uint64_t __nv_ptx_builtin_ocg_ld_mc_add_u64(uint64_t addr);
    __device__ uint64_t __nv_ptx_builtin_ocg_ld_mc_min_u64(uint64_t addr);
    __device__ uint64_t __nv_ptx_builtin_ocg_ld_mc_max_u64(uint64_t addr);
    __device__ uint64_t __nv_ptx_builtin_ocg_ld_mc_and_u64(uint64_t addr);
    __device__ uint64_t __nv_ptx_builtin_ocg_ld_mc_or_u64(uint64_t addr);
    __device__ uint64_t __nv_ptx_builtin_ocg_ld_mc_xor_u64(uint64_t addr);


    /*
     *     s64 ops
     */
    __device__ uint64_t __nv_ptx_builtin_ocg_ld_mc_min_s64(uint64_t addr);
    __device__ uint64_t __nv_ptx_builtin_ocg_ld_mc_max_s64(uint64_t addr);
}

#endif
