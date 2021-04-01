/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _tools_pascal_dci_h_
#define _tools_pascal_dci_h_

// Refer- https://p4viewer.nvidia.com/get/sw/compiler/gpgpu/doc/spec/DCI/Pascal_DCI.txt

#pragma pack(push, 4)
typedef struct CUtoolsPascalParam0_st {
    NvU32 sharedWindowBaseLo;
    NvU32 localWindowBaseLo;
    NvU32 blockDim[3]; // x, y, z
    NvU32 gridDim[3]; // x, y, z
    NvU32 userStackPointer;
    NvU32 crsSize;
    NvU64 gridId;
    NvU32 envRegs[32];
    NvU64 txqHeaderPoolVaddr;
    NvU64 txqSamplerPoolVaddr;
    NvU64 surfaceQueryDesctable;
    NvU64 constBank0Addr;
    NvU64 constBank3Addr;
    NvU64 constBank4Addr;
    NvU64 constBank5Addr;
    NvU64 constBank6Addr;
    NvU64 constBank1Addr;
    NvU32 contextIdProxy;
    NvU32 dynamicSharedMemSize;
    NvU32 sharedWindowBaseHi;
    NvU32 localWindowBaseHi;
    NvU32 partitionAsSM;
} CUtoolsPascalParam0;
#pragma pack(pop)

#endif //_tools_pascal_dci_h_
