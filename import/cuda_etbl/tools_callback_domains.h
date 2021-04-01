/*
 * Copyright 2010-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_callback_domains_h__
#define __cuda_etbl_tools_callback_domains_h__

// Whenever you add a new value to enum CUtools_cb_domain,
// add a corresponding ACTION(name) entry to this GENERATOR macro.
#define FOR_EACH_CALLBACK_DOMAIN(ACTION) \
    ACTION(INIT)                         \
    ACTION(RESOURCE)                     \
    ACTION(LAUNCH)                       \
    ACTION(PROFILER)                     \
    ACTION(SYNCHRONIZE)                  \
    ACTION(TRACE_API_CUDA)               \
    ACTION(TRACE_API_CUDA_RUNTIME)       \
    ACTION(MEMCPY)                       \
    ACTION(MEMSET)                       \
    ACTION(TRAP)                         \
    ACTION(QMD)                          \
    ACTION(TRACE_API_CUDA_ETBL)          \
    ACTION(RESOURCE_INTERNAL)            \
    ACTION(UVM_LITE)                     \
    ACTION(CUDA_RUNTIME_INTERNAL)        \
    ACTION(BATCH_MEMOP)                  \


#endif
