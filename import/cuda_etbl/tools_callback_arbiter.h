/*
 * Copyright 2010-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_callback_arbiter_h__
#define __cuda_etbl_tools_callback_arbiter_h__

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"

#include "cuda_etbl/tools_common_types.h"
#include "cuda_etbl/tools_cuda_api_meta.h"
#include "cuda_etbl/tools_callbacks.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*CUtoolsArbiterCbFunc)(void *userData);

typedef enum CUtools_cba_client_enum
{
    CU_TOOLS_CBA_CLIENT_INVALID         = 0,
    CU_TOOLS_CBA_CLIENT_DEBUGGER        = 1,
    CU_TOOLS_CBA_CLIENT_MEMCHECK        = 2,
    CU_TOOLS_CBA_CLIENT_PROFILER        = 3,

    // --- always add new constants to the end here ---
    CU_TOOLS_CBA_CLIENT_SIZE,
    CU_TOOLS_CBA_CLIENT_MAX             = 8,
} CUtools_cba_client;

typedef enum CUtools_cba_action_enum
{
    CU_TOOLS_CBA_ACTION_INVALID         = 0,
    CU_TOOLS_CBA_ACTION_SUBSCRIBE       = 1,
    CU_TOOLS_CBA_ACTION_UNSUBSCRIBE     = 2,

    CU_TOOLS_CBA_ACTION_SIZE,
    CU_TOOLS_CBA_ACTION_FORCE_INT       = 0x7fffffff
} CUtools_cba_action;

CU_DEFINE_UUID(CU_ETID_ToolsCallbackArbiter,
    0x3ec98cf8, 0xfd53, 0x469e, 0xba, 0x59, 0x1e, 0x2b, 0x87, 0x3e, 0xf, 0x91);

typedef struct CUetblToolsCallbackArbiter_st
{
    /// This export table supports versioning by adding to the end without changing
    /// the ETID.  The struct_size field will always be set to the size in bytes of
    /// the entire export table structure.
    size_t struct_size;

    /// Retrieves an array of the domains supported by this version of the driver.
    /// @param pDomainCount = Out param, number of supported domains.
    /// @param pDomainList = Out param, address of an array of supported domains.
    /// @return CUDA_SUCCESS if this function is implemented.
    CUresult (CUDAAPI *SupportedDomains)(
        size_t *pDomainCount,
        CUtoolsCallbackDomainTable *pDomainTable);

    /// Returns CUDA_SUCCESS if supported.
    /// Retrieves an array of the domains supported by this version of the driver.
    /// @param domain = Domain to query.
    /// @param pCallbackCount = Out param, number of supported callbacks in domain.
    /// @param pCallbackList = Out param, address of an array of supported callbacks in domain.
    CUresult (CUDAAPI *SupportedCallbacksInDomain)(
        size_t *pCallbackCount,
        CUtoolsCallbackDescriptorTable *pCallbackDescriptorTable,
        CUtools_cb_domain domain);

    /// Subscribe for callbacks for a specific domain.
    /// Subscribing and enabling are controlled independently, and per subscriber.
    /// All callbacks are disabled by default when a new subscription is made.
    /// @param callback Client's callback function pointer.
    /// @param userdata When the callback is invoked, this value is passed in as
    ///     the callback's 'userdata' parameter.  Analogous to the client's 'this' pointer.
    /// @param clientType the type of the client
    /// @return A subscriberId is returned.  subscriberId must be tracked by the client.
    /// threadsafety: safe to call any time
    /// processsafety : This function must NOT be called dynamically
    CUresult (CUDAAPI *Subscribe)(
        CUtoolsCbSubscriberHandle *pSubscriberHandle,
        CUtools_cba_client clientType,
        CUtoolsCbFunc callback,
        void *userdata);

    /// @return Returns CUDA_SUCCESS if this invocation was responsible for
    ///     removing the subscription.
    /// threadsafety: safe to call any time.  You are guaranteed to receive no
    /// future callbacks after Unsubscribe has returned.
    CUresult (CUDAAPI *Unsubscribe)(
        CUtoolsCbSubscriberHandle subscriberHandle);

    /// Gets or Sets whether the specified {subscriber, area, cbid} is enabled.
    /// Enable or disable a callback for the specified {subscriber, area, cbid}.
    /// threadsafety: Multiple subscribers may call concurrently, but each
    ///     subscriber must serialize access to Get and Set.  In other words:
    ///        If GetCallbackEnabled(sub, d, c) and SetCallbackEnabled(sub, d, c) are
    ///        called concurrently, the results are undefined.
    CUresult (CUDAAPI *CallbackEnabled)(
        uint32_t *pEnabled,
        CUtoolsCbSubscriberHandle subscriberHandle,
        CUtools_cb_domain domain,
        CUtoolsCbId cbid);

    CUresult (CUDAAPI *EnableCallback)(
        uint32_t enable,
        CUtoolsCbSubscriberHandle subscriberHandle,
        CUtools_cb_domain domain,
        CUtoolsCbId cbid);

    CUresult (CUDAAPI *EnableAllCallbacksInDomain)(
        uint32_t enable,
        CUtoolsCbSubscriberHandle subscriberHandle,
        CUtools_cb_domain domain);

    CUresult (CUDAAPI *EnableAllCallbacks)(
        uint32_t enable,
        CUtoolsCbSubscriberHandle subscriberHandle);

    /// This function returns if the Arbiter is busy (i.e. there are actions pending
    /// in the arbiter that will prevent a dynamic function call to subscribe/unsubscribe
    /// from succeeding.
    /// @param action : The type of action that will be attempted
    /// @return isUnsafe : nonzero if a dynamic call to subscribe/unsubscribe is allowed
    /// threadsafety : this function is unsafe to call normally within the process.
    /// ONLY SHOULD BE CALLED WITH THE APPLICATION FROZEN BY THE DEBUGGER
    CUresult (CUDAAPI *CanMakeDynamicCall)(CUtools_cba_client clientType, uint32_t *isUnsafe);

    /// This function allows a pending action callback to be added. This is used in the dynamic
    /// function call scenario to add an action that will occur after after the current action
    /// that would cause the dynamic call to block completes. The callback will be called in
    /// the context of the thread that caused the blocking action. This callback will only
    /// be called by one thread.
    /// The pending action is cleared immediately before the action is executed.
    /// Only 1 pending action can be present. The pending action must include some
    /// signal at the end of the function that will signal the debugger that the
    /// application is ready
    /// @param func : Pointer to function
    /// @param data : Pointer to data passed into function call
    /// @param action : The action after which the pending callback should be added
    /// ONLY SHOULD BE CALLED WITH THE APPLICATION FROZEN BY THE DEBUGGER
    CUresult (CUDAAPI *AddPendingAction) (CUtoolsArbiterCbFunc func, void *data);

} CUetblToolsCallbackArbiter;

#ifdef __cplusplus
}
#endif

#endif
