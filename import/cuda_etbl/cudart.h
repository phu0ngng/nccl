/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_cudart_h__
#define __cuda_etbl_cudart_h__

/**
 * Parameters for DeviceSetPrimaryCtxInitParams
 * - these come as pairs of (void *)s
 * - the first (void *) is the name (indicated below)
 * - the second (void *) is called value and is interpreted
 *   based on the name as described below
 */

// denotes the end of a parameter list
#define CU_PRIMARY_CTX_INIT_END                            ((void *)0x00)
// set the context create flags to *(unsigned int *)value
#define CU_PRIMARY_CTX_INIT_FLAGS                          ((void *)0x01)
// set the interop mode to value
#define CU_PRIMARY_CTX_INIT_INTEROP_MODE                   ((void *)0x02)
// set the D3D9 interop device to (IDirect3DDevice9 *)value
#define CU_PRIMARY_CTX_INIT_D3D9_INTEROP_DEVICE            ((void *)0x03)
// set the D3D10 interop device to (ID3D10Device *)value
#define CU_PRIMARY_CTX_INIT_D3D10_INTEROP_DEVICE           ((void *)0x04)
// set the D3D11 interop device to (ID3D11Device *)value
#define CU_PRIMARY_CTX_INIT_D3D11_INTEROP_DEVICE           ((void *)0x05)
// set the VDPAU device to *(VdpDevice *)value
#define CU_PRIMARY_CTX_INIT_VDPAU_INTEROP_DEVICE           ((void *)0x06)
// set the VDPAU getProcAddress to (VdpGetProcAddress *)value
#define CU_PRIMARY_CTX_INIT_VDPAU_INTEROP_GET_PROC_ADDRESS ((void *)0x07)

/**
 * Interop modes for the context create parameter 
 * CU_PRIMARY_CTX_INIT_INTEROP_MODE
 */

// set the interop mode to none
#define CU_PRIMARY_CTX_INTEROP_MODE_NONE                   ((void *)0x01)
// set the interop mode to D3D9
#define CU_PRIMARY_CTX_INTEROP_MODE_D3D9                   ((void *)0x02)
// set the interop mode to D3D10
#define CU_PRIMARY_CTX_INTEROP_MODE_D3D10                  ((void *)0x03)
// set the interop mode to D3D11
#define CU_PRIMARY_CTX_INTEROP_MODE_D3D11                  ((void *)0x04)
// set the interop mode to OpenGL
#define CU_PRIMARY_CTX_INTEROP_MODE_OPENGL                 ((void *)0x05)
// set the interop mode to VDPAU
#define CU_PRIMARY_CTX_INTEROP_MODE_VDPAU                  ((void *)0x06)

#include "cuda.h"
#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

//------------------------------------------------------------------
// Cuda Runtime API Interfaces
//------------------------------------------------------------------

/* This provides backdoor interfaces used by the CUDA Runtime API
 */

CU_DEFINE_UUID(CU_ETID_CudartInterface,
    0x6cfbd56b, 0xf45b, 0x4ae7, 0x89, 0x87, 0xd9, 0x39, 0x12, 0xfd, 0x9d, 0xf9);

typedef struct CUetblCudartInterface_st {
    /* Size of this structure */
    size_t size;
    
    /* Load a fat binary from the runtime
     * - will usually just call cuModuleLoadFatBinary
     * - for some special formats, it will call cuModuleLoadFromFile
     */
    CUresult (CUDAAPI *ModuleLoadFatBinary)(
        CUmodule *module,
        const void *fatCubin);

    /* Retrieve the primary context handle for a device
     */
    CUresult (CUDAAPI *DeviceGetPrimaryCtx)(
        CUcontext *context,
        CUdevice device);

    /* Set parameters for the lazy initialization 
     * - returns CUDA_ERROR_INVALID_VALUE for invalid parameters
     * - returns CUDA_ERROR_PRIMARY_CONTEXT_IS_ACTIVE if the 
     *   primary context has been initialized
     */
    CUresult (CUDAAPI *DeviceSetPrimaryCtxInitParams)(
        CUdevice device,
        void **params);

    /* Initialize the specified device's primary context
     * - returns CUDA_ERROR_PRIMARY_CONTEXT_IS_ACTIVE if the 
     *   primary context has been initialized
     * - returns any of cuCtxCreate's error codes on failure
     * - flags is flags to use if not device-specific flags
     *   were set
     */
    CUresult (CUDAAPI *DeviceInitializePrimaryCtx)(
        CUdevice device,
        unsigned int flags);
    
    /* Deinitialize the specified device's primary context
     * - also reset the context init params to their 
     *   defaults
     * - will return success unless the driver is not
     *   initialized or device is an invalid parameter
     */
    CUresult (CUDAAPI *DeviceResetPrimaryCtx)(
        CUdevice device);

    /* Load a fat binary from the runtime with host relocations
     * - will usually just call an extended cuModuleLoadFatBinary
     * - for some special formats, it will call cuModuleLoadFromFile
     *   and ignore host relocations
     */
    CUresult (CUDAAPI *ModuleLoadFatBinaryWithRelocations)(
        CUmodule *module,
        const void *fatCubin,
        const char **symbolNames,
        void **symbolAddresses,
        unsigned int symbolCount);

    /* Notify that fatCubin handle is destroyed and can be reused
     */
    void (CUDAAPI *FatCubinDestroyed)(
        const void *fatCubin);

    /// \brief Load a module as internal, hidden module. Modules loaded this way
    /// do not show up in tools, such as the profiler or debuggers.
    /// The arguments here are rearranged on purpose to avoid trivial function
    /// call modifications
    CUresult (CUDAAPI *ModuleLoadDataExHidden)(
        const void *data,
        CUmodule *module,
        CUjit_option *options,
        void **optionValues,
        unsigned int numOptions);
} CUetblCudartInterface;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
