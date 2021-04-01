/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_mps_h__
#define __cuda_etbl_tools_mps_h__

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_stdint.h"

#include "cuda_etbl/tools_common_types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus


/// This table allows tools like debuggers to contain MPS specific behavior
CU_DEFINE_UUID(CU_ETID_ToolsMPS,
    0x6089B836, 0x5861, 0x43F5, 0xBB, 0xD9, 0xAC, 0x63, 0x38, 0x78, 0xB6, 0xB1 );

typedef enum CUtoolsMPSMode_enum {
    CU_TOOLS_MPS_MODE_INVALID  = 0,   // Device is not capable of MPS
    CU_TOOLS_MPS_MODE_DISABLED = 1,   // Device is capable of MPS, but not running it
    CU_TOOLS_MPS_MODE_ENABLED  = 2,   // Device is capable of MPS and is running it

    /* Add new modes before this line */
    CU_TOOLS_MPS_MODE_SIZE,
    CU_TOOLS_MPS_MODE_FORCE_INT = 0x7fffffffU,
} CUtoolsMPSMode;

typedef enum CUtoolsMPSRole_enum {
    CU_TOOLS_MPS_ROLE_INVALID = 0,
    CU_TOOLS_MPS_ROLE_CLIENT  = 1,
    CU_TOOLS_MPS_ROLE_SERVER  = 2,

    /* Add new roles before this line */
    CU_TOOLS_MPS_ROLE_SIZE,
    CU_TOOLS_MPS_ROLE_FORCE_INT = 0x7fffffffU,
} CUtoolsMPSRole;

typedef uint64_t CUtoolsMPSClientId;

typedef struct CUtoolsMPSRoleInfo_st {
    uint32_t struct_size;       // Size of structure
    uint16_t mode;              // The mode of MPS and the device (Use type CUtoolsMPSMode)
    uint16_t role;              // The current role for the driver.(Use type CUtoolsMPSRole)
    CUtoolsMPSClientId clientId;// A handle to the client. (sizeof(uint64_t))
} CUtoolsMPSRoleInfo;

typedef enum CUtoolsMPSAcquireClientExclusiveFlags_enum {
    CU_TOOLS_MPS_ACQUIRE_CLIENT_EXCLUSIVE_FLAGS_DEFAULT = 0,

    CU_TOOLS_MPS_ACQUIRE_CLIENT_EXCLUSIVE_FLAGS_ALL = 0xffffffffU,
} CUtoolsMPSAcquireClientExclusiveFlags;

typedef enum CUtoolsMPSReleaseClientExclusiveFlags_enum {
    CU_TOOLS_MPS_RELEASE_CLIENT_EXCLUSIVE_FLAGS_DEFAULT = 0,

    CU_TOOLS_MPS_RELEASE_CLIENT_EXCLUSIVE_FLAGS_ALL = 0xffffffffU,
} CUtoolsMPSReleaseClientExclusiveFlags;

typedef struct CUetblToolsMPS_st {
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;

    // Get the current role of the client under MPS (if MPS is enabled)
    // \param dev - device which could be running MPS
    // \params pRoleInfo - Pointer to struct containing information.
    CUresult (CUDAAPI *GetRoleInfo)(
        CUdevice dev,
        CUtoolsMPSRoleInfo *pRoleInfo);

    // Acquire client exclusive access
    // See the functional spec for information about client exclusive
    // \param flags. For now, this must be CU_TOOLS_MPS_ACQUIRE_CLIENT_EXCLUSIVE_FLAGS_DEFAULT
    CUresult (CUDAAPI *AcquireClientExclusive)(
        CUtoolsMPSAcquireClientExclusiveFlags flags);

    // Release client exclusive access
    // See the functional spec for information about client exclusive mode
    // \param flags. For now, this must be CU_TOOLS_MPS_RELEASE_CLIENT_EXCLUSIVE_FLAGS_DEFAULT
    CUresult (CUDAAPI *ReleaseClientExclusive)(
        CUtoolsMPSReleaseClientExclusiveFlags flags);

} CUetblToolsMPS;


#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
