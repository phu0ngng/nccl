/*
 * Copyright 1993-2017 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _tools_volta_dci_h_
#define _tools_volta_dci_h_

// Refer- https://p4viewer.nvidia.com/get/sw/compiler/gpgpu/doc/spec/DCI/Volta_DCI.txt

typedef struct CUtoolsVoltaParam0_st {
    NvU32 blockDim[3]; // x, y, z
    NvU32 gridDim[3]; // x, y, z
    NvU64 sharedWindowBase;
    NvU64 localWindowBase;
    NvU32 userStackPointer;
    NvU32 dynamicSharedMemSize;
    NvU64 gridId;
    NvU64 qmdAddr;
    NvU64 constBank0Addr;
    NvU64 constBank1Addr;
    NvU64 constBank3Addr;
    NvU64 constBank4Addr;
    NvU64 constBank5Addr;
    NvU64 constBank6Addr;
    NvU64 textureQueryDescTableAddr;
    NvU64 samplerQueryDescTableAddr;
    NvU64 surfaceQueryDescTableAddr;
    NvU32 envRegs[32];
    NvU32 mpsContextId;
    NvU32 numVirtualSMs;
    NvU32 isCoopLaunch;
    NvU32 reserved0[19];
} CUtoolsVoltaParam0;

#endif //_tools_volta_dci_h_
