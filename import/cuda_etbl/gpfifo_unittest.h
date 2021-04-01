/*
 * Copyright 1993-2019 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __gpfifo_unittest_h__
#define __gpfifo_unittest_h__

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct CUetblGPFIFOUnitTestChannel_st *CUetblGPFIFOUnitTestChannel;

//------------------------------------------------------------------
// Backdoor driver API for driver unit tests which can't (or are
// hard to) test via the public API
//------------------------------------------------------------------

CU_DEFINE_UUID(CU_ETID_GPFIFOUnitTest,
    0x2bae8c4c, 0xcb7b, 0x40b0, 0xbe, 0x8b, 0x01, 0x41, 0x37, 0xeb, 0x1d, 0xbd);

typedef struct CUetblGPFIFOUnitTest_st {
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;

    // Get a channel for a stream.
    // Channel use is an enum value from CUtools_channel_use_type.
    CUresult (CUDAAPI *GetChannel)(CUstream hStream, uint32_t channelUse, CUetblGPFIFOUnitTestChannel *hChannel);

    // Gets a number of gpfifo entries in entries and waiting pad in pad.
    CUresult (CUDAAPI *GetEntries)(CUetblGPFIFOUnitTestChannel hChannel, size_t *entryCount, size_t *padCount);

    // Gets size of the pushbuffer.
    // It also synchronizes the channel.
    // Channel use is an enum value from CUtools_channel_use_type.
    CUresult (CUDAAPI *GetPushbufferSize)(CUetblGPFIFOUnitTestChannel hChannel, size_t *pushbufferSize, size_t *maxPushSize, size_t *pushbufferAlignment);

    // Synchronizes a channel and clean up GPFIFO
    CUresult (CUDAAPI *SynchronizeChannel)(CUetblGPFIFOUnitTestChannel hChannel);

    // Gets size of maximum inline copy
    CUresult (CUDAAPI *GetMaxInlineCopy)(CUetblGPFIFOUnitTestChannel hChannel, size_t *maxInline);

    // Submits a pushbuffer
    CUresult (CUDAAPI *SubmitPushbuffer)(CUstream hStream, CUetblGPFIFOUnitTestChannel channel, const uint32_t *pushbuffer, size_t pushbufferSizeInWords);
} CUetblGPFIFOUnitTest;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
