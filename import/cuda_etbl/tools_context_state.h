/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_context_state_h__
#define __cuda_etbl_tools_context_state_h__

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"

#include "cuda_etbl/tools_common_types.h"

#include "stdio.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef struct CUtoolsContextState_st *CUtoolsContextStateHandle;

// These callbacks can be used to override the default behavior of calling
// malloc once for each memobj.  This allows tools to better handle cases
// where there's not sufficient system memory to back up all the memobjs,
// or where malloc-per-memobj performance is unacceptable.

// The store callback must copy sizeInBytes bytes from data into some form
// of storage, and insert a reference to it in a map keyed by memObj.
typedef CUresult (CUDAAPI *CUtoolsContextStoreCallback)(
        CUtoolsContextStateHandle hSavedState,
        CUtoolsMemObjHandle memObj,
        size_t sizeInBytes,
        void *data,
        void *callbackContext);
        
// The restore callback must lookup memObj in the map of stored data and
// write into out parameter *data the address from which the driver should
// copy the original data, overwriting in the current contents of the memobj.
// The driver already knows the size of the memobj, and will copy exactly
// that number of bytes from *data.
typedef CUresult (CUDAAPI *CUtoolsContextRestoreCallback)(
        CUtoolsContextStateHandle hSavedState,
        CUtoolsMemObjHandle memObj,
        void **data,
        void *callbackContext);

//  Abstraction for internal type CUmemflagsMapHost_enum
typedef enum CUtools_host_mapping_type_enum
{
    CU_TOOLS_MEMORY_MAP_HOST_NONE   = 0,
    CU_TOOLS_MEMORY_MAP_HOST_VA     = 1,
    // --- always add new constants to the end here ---
    CU_TOOLS_MEMORY_MAP_HOST_SIZE,
    CU_TOOLS_MEMORY_MAP_HOST_FORCE_INT = 0x7fffffff
} CUtools_host_mapping_type;

//  Abstraction for internal type CUmemflagsMapDevice_enum
typedef enum CUtools_device_mapping_type_enum
{
    CU_TOOLS_MEMORY_MAP_DEVICE_NONE                = 0,
    CU_TOOLS_MEMORY_MAP_DEVICE_VA                  = 1,
    CU_TOOLS_MEMORY_MAP_DEVICE_LIMIT_TO_32_BIT     = 2,
    CU_TOOLS_MEMORY_MAP_DEVICE_LIMIT_TO_DEVICE_PTR = 3,  // Same as VA starting with Fermi
    CU_TOOLS_MEMORY_MAP_DEVICE_RANGE               = 4,
    CU_TOOLS_MEMORY_MAP_DEVICE_SPECIAL             = 5,
    // --- always add new constants to the end here ---
    CU_TOOLS_MEMORY_MAP_DEVICE_SIZE,
    CU_TOOLS_MEMORY_MAP_DEVICE_FORCE_INT = 0x7fffffff
} CUtools_device_mapping_type;

// Parameter type for MemBlockGetLocation
typedef struct CUtoolsMemBlockLocation_st
{
    uint32_t struct_size;
    uint32_t physicalLocation;     // One of CUtools_memory_location (in tools_memory.h)
    uint32_t hostMappingType;      // One of CUtools_host_mapping_type
    uint32_t deviceMappingType;    // One of CUtools_device_mapping_type
    uint64_t hostVA;               // Address used by device to access pinned host mem (zero if not mapped)
    uint64_t deviceVA;             // Device virtual address (not device pointer, and before Fermi they were different)
    uint64_t blockSize;            // Size in bytes of the memblock
} CUtoolsMemBlockLocation;

//  Abstraction for internal type CUmemflagsType_enum
typedef enum CUtools_memory_type_enum
{
    CU_TOOLS_MEMORY_TYPE_INVALID          =  0,
    CU_TOOLS_MEMORY_TYPE_GENERIC          =  1,
    CU_TOOLS_MEMORY_TYPE_IMAGE            =  2,
    CU_TOOLS_MEMORY_TYPE_PUSHBUFFER       =  3,
    CU_TOOLS_MEMORY_TYPE_GPFIFOBUFFER     =  4,
    CU_TOOLS_MEMORY_TYPE_FUNCTION         =  5,
    CU_TOOLS_MEMORY_TYPE_CONTEXT_SAVE     =  6,
    CU_TOOLS_MEMORY_TYPE_TEXTURE          =  7,
    CU_TOOLS_MEMORY_TYPE_CLH_OOL          =  8,
    CU_TOOLS_MEMORY_TYPE_CONSTANT         =  9,
    CU_TOOLS_MEMORY_TYPE_VIRTUAL_CHANNEL  = 10,
    CU_TOOLS_MEMORY_TYPE_NOTIFIER         = 11,
    CU_TOOLS_MEMORY_TYPE_SKED_REFLECTED   = 12,
    CU_TOOLS_MEMORY_TYPE_SHARED_SEMAPHORE = 13,
    // --- always add new constants to the end here ---
    CU_TOOLS_MEMORY_TYPE_SIZE,
    CU_TOOLS_MEMORY_TYPE_FORCE_INT = 0x7fffffff
} CUtools_memory_type;

// Parameter type for MemBlockGetAccessibility
typedef struct CUtoolsMemBlockAccessibility_st
{
    uint32_t struct_size;
    uint32_t type;                 // One of CUtools_memory_type
    uint32_t userOwned;            // Bool, non-zero means true
    uint32_t readOnly;             // Bool, non-zero means true
    uint32_t apiVisible;           // Bool, non-zero means true
    uint32_t sharedWithOtherCtx;   // Bool, non-zero means true
} CUtoolsMemBlockAccessibility;


CU_DEFINE_UUID(CU_ETID_ToolsContextState,
    0x1aac58fc, 0x7c05, 0x4f44, 0xa9, 0x77, 0x72, 0xee, 0x9a, 0xef, 0xdb, 0x89);
    
typedef struct CUetblToolsContextState_st {
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;

    /// \brief Query the driver for the amount of state that would need to stored
    /// in order to guarantee the state could be restored after launching a kernel.
    CUresult (CUDAAPI *QueryWritableState)(
        CUfunction func,
        uint32_t *memObjectCount,
        size_t *memObjectMaxSizeInBytes,
        size_t *memObjectsTotalSizeInBytes);
    
    /// \brief Store all writable state such that the state could be restored
    /// after launching a kernel.
    CUresult (CUDAAPI *SaveWritableState)(
        CUtoolsContextStateHandle *phSavedState,
        CUfunction func,
        CUtoolsContextStoreCallback storeCallback,
        void *callbackContext);

    /// \brief Restore state saved by SaveWritableState.
    CUresult (CUDAAPI *RestoreWritableState)(
        CUtoolsContextStateHandle hSavedState,
        CUtoolsContextRestoreCallback restoreCallback,
        void *callbackContext);

    /// \brief Free resources allocated by SaveWritableState.
    CUresult (CUDAAPI *FreeWritableState)(
        CUtoolsContextStateHandle hSavedState);

    /// Get a memblock's size, physical location, and
    /// mappings.  EXPERIMENTAL, do not use in shipping tools
    /// without adequate version safety!
    CUresult (CUDAAPI *MemBlockGetLocation)(
        CUtoolsMemBlockHandle hMemBlock,
        CUtoolsMemBlockLocation *pLocation);

    /// Get info about the type of data in the memblock,
    /// who manages its lifetime, and what restrictions
    /// there are on its accessibility.  EXPERIMENTAL, do not
    /// use in shipping tools without adequate version safety!
    CUresult (CUDAAPI *MemBlockGetAccessibility)(
        CUtoolsMemBlockHandle hMemBlock,
        CUtoolsMemBlockAccessibility *pAccessibility);

} CUetblToolsContextState;


#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
