/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _CHANNEL_RESET_TEST_H_
#define _CHANNEL_RESET_TEST_H_

#include "cuda.h"
#include "cuda_uuid.h"

#ifdef __cplusplus
extern "C" {
#endif

// {D7850D9B-C31F-4f32-9366-56DD5A199597}
CU_DEFINE_UUID(CU_ETID_channelResetTest, 
0xd7850d9b, 0xc31f, 0x4f32, 0x93, 0x66, 0x56, 0xdd, 0x5a, 0x19, 0x95, 0x97);

typedef enum namedChannel_enum {
    CHAN_TYPE_COMPUTE = 0,
    CHAN_TYPE_HTOD    = 1,
    CHAN_TYPE_DTOH    = 2
} namedChannel;

// The export table containing functions needed for the channel reset test.
typedef struct CUetblChannelResetTest_st {

    CUresult (*getChannelFromContext_etbl) (CUcontext ctx, namedChannel cType, void **channel);
    CUresult (*resetChannel_etbl)          (void *channel);

} CUetblChannelResetTest;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // _CHANNEL_RESET_TEST_H_
