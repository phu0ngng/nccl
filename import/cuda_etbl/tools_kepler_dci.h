/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _tools_kepler_dci_h_
#define _tools_kepler_dci_h_

#pragma pack(push, 4)
typedef struct CUtoolsKeplerParam0_st {
    NvU32 texLdgRelocs[8];
    NvU32 sharedWindowBase;
    NvU32 localWindowBase;
    NvU32 blockDim[3]; // x, y, z
    NvU32 gridDim[3]; // x, y, z
    NvU32 gridId32; // Deprecated (see DCI)
    NvU32 userStackPointer;
    NvU32 crsSize;
    NvU32 envRegs[32];
    NvU64 txqHeaderPoolVaddr;
    NvU64 txqSamplerPoolVaddr;
    NvU64 gridId;
    NvU32 padding;
    NvU64 constBank0Addr;
    NvU64 constBank3Addr;
    NvU64 constBank4Addr;
    NvU64 constBank5Addr;
    NvU64 constBank6Addr;
} CUtoolsKeplerParam0;
#pragma pack(pop)

#endif //_tools_kepler_dci_h_