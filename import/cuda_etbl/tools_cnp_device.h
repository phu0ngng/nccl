/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef _tools_cnp_device_h_
#define _tools_cnp_device_h_

#include "cuda_stdint.h"
// =============================================================================
// Global switch to enable support for device-sided ToolsAPI.
// Off by default. Can be enabled by a tool on the host side.
// Notes: Any overhead introduced by ToolsAPI needs to be guarded by this flag.
//        That way the impact of non-instrumented application is kept as low as
//        possible.
typedef enum CUtools_cnp_device_callbacks_version_enum
{
    CU_TOOLS_CNP_DEVICE_CALLBACKS_DISABLED          = 0,
    CU_TOOLS_CNP_DEVICE_CALLBACKS_V1                = 1,
    CU_TOOLS_CNP_DEVICE_CALLBACKS_V2                = 2,
    // --- Always add new constants to the end here ---
    CU_TOOLS_CNP_DEVICE_CALLBACKS_VERSION_SIZE,
    CU_TOOLS_CNP_DEVICE_CALLBACKS_VERSION_FORCE_INT = 0x7fffffff
} CUtools_cnp_device_callbacks_version;

// =============================================================================
// Tools API types that hide internals from the users of tools API to assure
// that changes in the driver do not break the tools.

typedef struct CNPstream_st* CUtoolsCnpStream;
typedef struct CNPqmdLaunch_st* CUtoolsCnpQmdLaunchHandle;
typedef struct CNPctaCtx_st* CUtoolsCnpCtaCtxHandle;

// =============================================================================
// Prototypes for all callback functions. Please see the public interface 
// CUtools_cnp_device_callback_ids in tools_cnp.h

// QMD Launch Callback
typedef void (*CUtoolsCnpCallbackOnQmdLaunch)(
    CUtoolsCnpStream stream,
    CUtoolsCnpQmdLaunchHandle childQmdLaunch,
    uint32_t childStartPc,
    uint64_t param0ToolsBase);
// v2 changes to use a 64-bit start PC
typedef void (*CUtoolsCnpCallbackOnQmdLaunch_v2)(
    CUtoolsCnpStream stream,
    CUtoolsCnpQmdLaunchHandle childQmdLaunch,
    uint64_t childStartPc,
    uint64_t param0ToolsBase);

// QMD Completing Callback
typedef void (*CUtoolsCnpCallbackOnCompleting)(
    CUtoolsCnpQmdLaunchHandle completingQmdLaunch);

// QMD Queued Callback
typedef void (*CUtoolsCnpCallbackOnQmdSubmitted)(
    CUtoolsCnpQmdLaunchHandle submittedQmdLaunch);

// CTA Before Save Callback
typedef void (*CUtoolsCnpCallbackOnCtaBeforeSave)(
    CUtoolsCnpCtaCtxHandle ctaCtxHandle);

// CTA After Restore Callback
typedef void (*CUtoolsCnpCallbackOnCtaAfterRestore)(
    CUtoolsCnpCtaCtxHandle ctaCtxHandle);

// =============================================================================
// Prototypes for helper functions provides by tools API. Those functions can
// be used by the tools' module if required by adding them to the modules const
// bank. That way we can avoid adding more constants into the sycall module and
// we also don't need to add any of them as a true, compiler-exposed syscall.

typedef CUtoolsCnpQmdLaunchHandle (*CUtoolsCnpGetSelfQmdLaunch)(void);

typedef void* (*CUtoolsCnpGetGridQmd)(
    CUtoolsCnpQmdLaunchHandle qmdLaunch);

typedef void* (*CUtoolsCnpGetQueueQmd)(
    CUtoolsCnpQmdLaunchHandle qmdLaunch);

typedef void* (*CUtoolsCnpGetGridParams)(
    CUtoolsCnpQmdLaunchHandle qmdLaunch);

// DEPRECATED, DO NOT USE - use CUtoolsCnpGetStartPc64 or 64-bit startPC callback parameter
typedef uint32_t (*CUtoolsCnpGetStartPc)(void);

typedef uint64_t (*CUtoolsCnpGetStartPc64)(void);

// Get the CTA ID for the current thread. This function handles both, reading
// from S2Rs or from LMEM. The decision is made on the IsQueue flag of the
// current QMD.  On Volta+, these are not necessary since the S2Rs correctly
// handle preempt/restore of these values.  The individual functions are less
// efficient than a single call to CUtoolsCnpGetCtaBlockIdx, and are deprecated.
typedef uint32_t (*CUtoolsCnpGetCtaX)(void);    // DEPRECATED, use CUtoolsCnpGetCtaBlockIdx
typedef uint32_t (*CUtoolsCnpGetCtaY)(void);    // DEPRECATED, use CUtoolsCnpGetCtaBlockIdx
typedef uint32_t (*CUtoolsCnpGetCtaZ)(void);    // DEPRECATED, use CUtoolsCnpGetCtaBlockIdx
typedef void (*CUtoolsCnpGetCtaBlockIdx)(uint32_t* x, uint32_t* y, uint32_t* z);

typedef void (*CUtoolsCnpGetSteeringEnabled)(
    CUtoolsCnpQmdLaunchHandle qmdLaunch,
    uint32_t* steeringEnabled,
    uint32_t* steeringExpected);

#endif //_tools_cnp_device_h_
