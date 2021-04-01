/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_profilesyscall_h__
#define __cuda_etbl_tools_profilesyscall_h__

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"

#include "cuda_etbl/tools_common_types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/*
 * \brief Profile-syscall provides functions for accessing and
 * controlling __profile syscall
 *
 * {79d88030-325b-11e1-b86c-0800200c9a66}
 */
CU_DEFINE_UUID(CU_ETID_ToolsProfileSyscall,
    0x79d88030, 0x325b, 0x11e1, 0xb8, 0x6c, 0x08, 0x00, 0x20, 0x0c, 0x9a, 0x66);

typedef struct CUetblToolsProfileSyscall_st {
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;

    /*
     * Get the offset that indicates a full warp-slot buffer. A buffer
     * offset greater than or equal to this value indicates that the
     * warp-slot buffer is full and can record no additional records.
     */
    CUresult (CUDAAPI *etiGetFullProfileSyscallBufferOffset)(
        CUcontext ctx, 
        uint32_t *fullOffset);

    /*
     * Configure the size and kind of the buffer to use for the
     * profile records. 'size' inputs the requested total buffer
     * size. Profiling is disable if 'size' == 0. 'warpSlotBufferCnt'
     * returns the number of warp-slot buffers created within the
     * allocated buffer, and 'size' returns the size of each warp-slot
     * buffer.
     *
     * 'kind' is used to request device or zero-copy memory. Currently
     * unused (device memory is always used).
     */
    CUresult (CUDAAPI *etiConfigureProfileSyscallBuffer)(
        CUcontext ctx, 
        uint32_t *size,
        uint32_t *warpSlotBufferCnt,
        int32_t kind);

    /*
     * Flush the warp-slot buffers and their sizes into the provided
     * output buffers, 'recordBuffer' and 'sizeBuffer'.
     * 'recordBufferSize' inputs the size of 'recordBuffer' and
     * returns the size of data flushed into the buffer.
     * 'sizeBufferSize' inputs the size of 'sizeBuffer' and returns
     * the size of data flushed into the buffer.
     */
    CUresult (CUDAAPI *etiFlushProfileSyscallBuffer)(
        CUcontext ctx, 
        void *recordBuffer,
        uint32_t *recordBufferSize,
        uint32_t *sizeBuffer,
        uint32_t *sizeBufferSize);

} CUetblToolsProfileSyscall;


#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
