/*
 * Copyright 1993-2012 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _tools_maxwell_dci_h_
#define _tools_maxwell_dci_h_

typedef struct CUmaxwellGridParams_st {
    NvU32 sharedWindowBase;
    NvU32 localWindowBase;
    NvU32 blockDim[3]; // x, y, z
    NvU32 gridDim[3]; // x, y, z
    NvU32 userStackPointer;
    NvU32 crsSize;
    NvU64 gridId;
    NvU32 envRegs[32];
    NvU64 txqHeaderPoolVaddr;
    NvU64 txqSamplerPoolVaddr;
    NvU64 txqSurfacePoolVaddr;
    NvU64 constBank0Addr;
    NvU64 constBank3Addr;
    NvU64 constBank4Addr;
    NvU64 constBank5Addr;
    NvU64 constBank6Addr;
    NvU64 constBank1Addr;
    NvU32 proxyContextId;
} CUmaxwellGridParams;

#endif //_tools_maxwell_dci_h_