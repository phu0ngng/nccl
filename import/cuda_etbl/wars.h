/*
 * Copyright 2012-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_wars_h__
#define __cuda_etbl_wars_h__

#include "cuda.h"
#include "cuda_uuid.h"
#include "nvtypes.h"
#include "tools_callbacks.h"
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

//------------------------------------------------------------------
// Backdoor driver API for various WARs
//------------------------------------------------------------------

CU_DEFINE_UUID(CU_ETID_WARS,
    0x58834512, 0xc2a7, 0x591e, 0x21, 0x77, 0xef, 0x35, 0x56, 0x18, 0x87, 0x09);

typedef struct CUetblWars_t {
    size_t struct_size;
    void            (*etiWarDisableKeplerL1WarOnHighEndGpus)(void);

    // Func:        etiWarGetPatchMode(patchType,*mode)
    //
    // Desc:        Query the current 'mode' for a given patch type.
    //              By convention:
    //                  mode=0 means disabled
    //                  mode=1 means enabled, standard behavior.
    //
    //              Specific patch types may have other optional modes, which will
    //              have mode values >1.
    //
    // Returns:    CUDA_SUCCESS                The patch-mode was returned. 
    //             CUDA_ERROR_INVALID_VALUE    Bad 'patchType' or 'mode' value.
    CUresult        (*etiWarGetPatchMode)(CUtools_patch_type patchType, NvU32* mode /*out*/);

    // Func:        etiWarSetPatchMode(patchType,mode)
    // 
    // Purpose:    This function allows tools to enable/disable certain patch WARs,
    //             or to select optional 'modes' for those patches.
    // 
    // Notes:      The patch-mode can only changed before any patches have been created.
    //             (i.e. at startup).  Tools which late-attach cannot change the patch mode.
    //
    // Args:       patchType       Which patch-WAR mode to set.
    //             mode            The mode to set.
    //                             =0 means disabled
    //                             =1 means enabled (normal mode)
    //                             Other non-zero values may be allowed, depending on the patchType.
    //
    // Returns:    CUDA_SUCCESS                The patch-mode was changed. 
    //             CUDA_ERROR_INVALID_VALUE    Bad 'patchType' or 'mode' value.
    //             CUDA_ERROR_NOT_PERMITTED    Patches of this type have already been created.  It's too late to change the mode. 
    // 
    CUresult        (*etiWarSetPatchMode)(CUtools_patch_type patchType, NvU32 mode);

} CUetblWars;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // __cuda_etbl_wars_h__

