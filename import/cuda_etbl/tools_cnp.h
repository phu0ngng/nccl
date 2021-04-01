/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_cnp_h__
#define __cuda_etbl_tools_cnp_h__

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_stdint.h"
#include "tools_cnp_device.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// Set of device function pointers to register for device-side Tools
// API callbacks.  Specify zero for a function pointer to ignore that
// callback.
typedef struct CUtoolsCnpDeviceCallbacks_v1_st
{
    uint32_t version;
    uint32_t struct_size;

    uint32_t OnQmdLaunch;
    uint32_t OnQmdCompleting;
    uint32_t OnQmdSubmitted;
    uint32_t OnCtaBeforeSave;
    uint32_t OnCtaAfterRestore;
} CUtoolsCnpDeviceCallbacks_v1;

typedef struct CUtoolsCnpDeviceCallbacks_v2_st // Changed to take 64-bit PCs
{
    uint32_t version;
    uint32_t struct_size;

    uint64_t OnQmdLaunch_v2; // Changed to take a 64-bit start PC param, so rename to disambiguate
    uint64_t OnQmdCompleting;
    uint64_t OnQmdSubmitted;
    uint64_t OnCtaBeforeSave;
    uint64_t OnCtaAfterRestore;
} CUtoolsCnpDeviceCallbacks_v2;

// This structure describes the internal layout of some CNP data structures.
// If any of the offsetof (oo) or size (sz) fields becomes invalid, it will
// be assigned a value of -1u.
typedef struct CUtoolsCnpDebuggerInfo_st {
    uint32_t struct_size;

    uint32_t oo_cql_gridQmd;              // offsetof(CNPqmdLaunch, GridQmd);
    uint32_t oo_cql_queueQmd;             // offsetof(CNPqmdLaunch, QueueQmd);
    uint32_t oo_cql_paramConstDPtr;       // offsetof(CNPqmdLaunch, paramConstDptr);
    uint32_t sz_cql_paramConstDPtr;

    uint32_t oo_paramConst_selfQmdVAddr;  // offset of the selfQMD-pointer in param const bank
    uint32_t sz_paramConst_selfQmdVAddr;
    uint32_t oo_paramConst_startPc;       // offset of the user's start PC in param const bank
    uint32_t sz_paramConst_startPc;

    uint32_t oo_ccc_qmdLaunchPtr;         // offsetof(CNPctaCtx, gridOrigin);
    uint32_t sz_ccc_qmdLaunchPtr;
    uint32_t oo_ccc_contDataPtr;          // offsetof(CNPctaCtx, ctaContinuationData);
    uint32_t sz_ccc_contDataPtr;
    uint32_t oo_ccc_blockIdx_x;
    uint32_t sz_ccc_blockIdx_x;
    uint32_t oo_ccc_blockIdx_y;
    uint32_t sz_ccc_blockIdx_y;
    uint32_t oo_ccc_blockIdx_z;
    uint32_t sz_ccc_blockIdx_z;

    uint32_t contDataBufferFormatVersion;
} CUtoolsCnpDebuggerInfo;

typedef enum CUtools_cnp_option_enum
{
    CU_TOOLS_CNP_OPTION_INVALID     = 0,    // For GetOption, the option was not valid on a given architecture.
                                            // For SetOption, the option is to be ignored when setting.
    CU_TOOLS_CNP_OPTION_ENABLE      = 1,    // Set the option
    CU_TOOLS_CNP_OPTION_DISABLE     = 2,    // Clear the option

    // --- always add new constants to the end here ---
    CU_TOOLS_CNP_OPTION_SIZE,
    CU_TOOLS_CNP_OPTION_FORCE_INT   = 0x7fffffff
} CUtools_cnp_option;

// This structure contains internal CNP options that can be queried/set by tools.
// This is meant to change driver internal behavior and each option has specific
// side effects.
// Options:
//  r1HoldsUserStackPointer : This option controls whether CNP system code will
//                            assume that register R1 holds the user's stack pointer
//                            (as is default in ABI compilations). This is used
//                            by the system code to optimize the amount of
//                            local memory in Lmem Hi that is save/restored when doing
//                            a CNP initiated Preempt/restore. This option has
//                            an impact on GK110.
typedef struct CUtoolsCnpOptions_st {
    uint32_t struct_size;
    CUtools_cnp_option r1HoldsUserStackPointer; // sizeof(uint32_t)
} CUtoolsCnpOptions;


// This table contains functions specific to CNP.
CU_DEFINE_UUID(CU_ETID_ToolsCnp,
    0xf66339c8, 0xf717, 0x41e2, 0xa5, 0x74, 0x91, 0x43, 0x31, 0x4d, 0xff, 0x9f);

typedef struct CUetblToolsCnp_st
{
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;

    // Specify a set of function pointers to be used for device-side callbacks for
    // a given context.  If cnpCallbacks is zero, disable all callbacks.  If using
    // a newer version of the CUtoolsCnpDeviceCallbacks struct, simply cast to
    // const CUtoolsCnpDeviceCallbacks_v1* when passing its address -- all versions
    // have a binary-compatible header that allows communicating the version.
    CUresult (CUDAAPI *SetDeviceCallbacks)(
        CUcontext ctx,
        const CUtoolsCnpDeviceCallbacks_v1 *cnpDeviceCallbacks);

    CUresult (CUDAAPI *GetCnpDebuggerInfo)(
        CUcontext ctx,
        CUtoolsCnpDebuggerInfo *info);

    // Get the system software CUfuncs that wrap kernel launches when user functions
    // call cnp syscalls. This function should only be called after a user module
    // has been loaded which references cnp syscalls.
    // entryFunc - If a kernel launch can use cnp, this CUfunc will run as a
    // preamble to the kernel before branching into user code.
    // exitFunc - This function will run after user code exits after any kernel
    // that ran the entry preamble.
    CUresult (CUDAAPI *GetCnpEntryExitFuncs)(
        CUcontext ctx,
        CUfunction *entryFunc,
        CUfunction *exitFunc);

    /// \brief Set up context wide CNP options
    /// \param ctx The context of interest
    /// \param options The options to be set
    CUresult (CUDAAPI *SetCnpOptions)(
        CUcontext ctx,
        CUtoolsCnpOptions *options);

    /// \brief Get the context wide CNP options
    /// \param ctx The context of interest
    /// \param options The options
    CUresult (CUDAAPI *GetCnpOptions)(
        CUcontext ctx,
        CUtoolsCnpOptions *options);

    /// \brief Enable/disable CNP EPLB (launch queue). Call it between
    /// CU_TOOLS_CBID_INIT_INITIALIZED and
    /// CU_TOOLS_CBID_RESOURCE_CONTEXT_INITIALIZE_STARTING, inclusive
    /// \param flag set to 1 to disable launch queue
    CUresult (CUDAAPI *DisableLaunchQueue)(
        uint32_t flag);

    /// \brief Enable/disable CNP control interface.
    /// \param flag set to 1 to enable cnp control interface
    CUresult (CUDAAPI *CtxEnableCnpCtrl)(
        uint32_t flag);


    /// \brief Get the Sked reflected address
    /// \param ctx The context of interest
    /// \param skedAddress The returned Device VA of the sked reflected address
    CUresult (CUDAAPI *GetSkedReflectedAddress)(
        CUcontext ctx,
        uint64_t *skedReflectedAddress);
} CUetblToolsCnp;


#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
