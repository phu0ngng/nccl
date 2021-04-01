/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_memory_h__
#define __cuda_etbl_tools_memory_h__

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"

#include "cuda_etbl/tools_common_types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus


//  Abstraction for internal type CUtexref_type
typedef enum CUtools_tex_ref_binding_type_enum
{
    CU_TOOLS_TEX_REF_BINDING_TYPE_INVALID              = 0,
    CU_TOOLS_TEX_REF_BINDING_TYPE_DEVICE_ADDRESS       = 1,
    CU_TOOLS_TEX_REF_BINDING_TYPE_ARRAY                = 2,
    CU_TOOLS_TEX_REF_BINDING_TYPE_CPU_ADDRESS          = 3,
    // --- always add new constants to the end here ---
    CU_TOOLS_TEX_REF_BINDING_TYPE_SIZE,
    CU_TOOLS_TEX_REF_BINDING_TYPE_FORCE_INT            = 0x7fffffff
} CUtools_tex_ref_binding_type;

//  Abstraction for internal type CUmemflagsLocation_enum
typedef enum CUtools_memory_location_enum
{
    CU_TOOLS_MEMORY_LOCATION_INVALID   = 0,
    CU_TOOLS_MEMORY_LOCATION_HOST      = 1,
    CU_TOOLS_MEMORY_LOCATION_DEVICE    = 2,
    // --- always add new constants to the end here ---
    CU_TOOLS_MEMORY_LOCATION_SIZE,
    CU_TOOLS_MEMORY_LOCATION_FORCE_INT = 0x7fffffff,
    CU_TOOLS_MEMORY_LOCATION_UNKNOWN   = 0x7fffffff // For detecting untranslated values
} CUtools_memory_location;

//  Abstraction for internal type CUsurfref_type
typedef enum CUtools_surf_ref_binding_type_enum
{
    CU_TOOLS_SURF_REF_BINDING_TYPE_INVALID              = 0,
    CU_TOOLS_SURF_REF_BINDING_TYPE_DEVICE_ADDRESS       = 1,
    CU_TOOLS_SURF_REF_BINDING_TYPE_ARRAY                = 2,
    // --- always add new constants to the end here ---
    CU_TOOLS_SURF_REF_BINDING_TYPE_SIZE,
    CU_TOOLS_SURF_REF_BINDING_TYPE_FORCE_INT            = 0x7fffffff
} CUtools_surf_ref_binding_type;

typedef enum CUtoolsMemAllocType_enum
{
    CU_TOOLS_MEM_ALLOC_TYPE_DEVICE                   = 0,
    CU_TOOLS_MEM_ALLOC_TYPE_HOST_PINNED              = 1,
    CU_TOOLS_MEM_ALLOC_TYPE_HOST_WITH_DEVICE_CACHING = 2,
    // --- always add new constants to the end here ---
    CU_TOOLS_MEM_ALLOC_TYPE_SIZE,
    CU_TOOLS_MEM_ALLOC_TYPE_FORCE_INT                = 0x7fffffff
} CUtoolsMemAllocType;

typedef enum CUtoolsMemAllocPurpose_enum
{
    CU_TOOLS_MEM_ALLOC_PURPOSE_GENERIC    = 0, // Generic memory
    CU_TOOLS_MEM_ALLOC_PURPOSE_USED_BY_HW = 1, // Memory used directly by HW such as semaphores/QMDs
    // --- always add new constants to the end here ---
    CU_TOOLS_MEM_ALLOC_PURPOSE_SIZE,
    CU_TOOLS_MEM_ALLOC_PURPOSE_FORCE_INT  = 0x7fffffff
} CUtoolsMemAllocPurpose;

// Copy method for memcpyDesc
typedef enum CUtoolsMemcpyType_enum {
    CU_TOOLS_MEMCPY_TYPE_DEFAULT        = 0,
    CU_TOOLS_MEMCPY_TYPE_CE             = 1,
    CU_TOOLS_MEMCPY_TYPE_KERNEL         = 2,
    CU_TOOLS_MEMCPY_TYPE_INLINE         = 3,
    // --- always add new constants to the end here ---
    CU_TOOLS_MEMCPY_TYPE_FORCE_INT      = 0x7fffffff
} CUtoolsMemcpyType;

typedef struct CUtoolsMemAllocOptions_st
{
    uint32_t struct_size;

    uint8_t  type;      // Use values from CUtoolsMemAllocType
    uint8_t  purpose;   // Use values from CUtoolsMemAllocPurpose
    uint16_t reserved0; //padding
} CUtoolsMemAllocOptions;

typedef struct CUtoolsTexRefDescriptor_st
{
    uint32_t size;
    uint32_t texRefBindingType;        // one of CUtools_tex_ref_binding_type   

    //  binding data
    void *devptr;
    size_t devptrBytes;
    CUarray hArray;                    // size is sizeof(void*)
    void *pCpuBaseAddress;

    //  If from CUmodule, this is the programmatic name (from source and cubin).
    const char *pName;

    //  Texture setup.
    size_t width;
    size_t height;
    size_t pitch;
    uint32_t elementFormat;            // one of CUarray_format
    uint32_t numChannels;
    uint32_t use2DNoMipMap;
    uint32_t addressMode[3];           // one of CUaddress_mode for each of {x, y, z}
    uint32_t filterMode;               // one of CUfilter_mode

    //  Flags as specified to cuTexRefSetFlags (CU_TRSF_*)
    uint32_t texRefFlags;

    //  Depth dimension.
    size_t depth;

    //  Owning module if from CUmodule.
    CUmodule mod;

    //  Maximum anisotropy level
    uint32_t maxAnisotropy;

    //  Mipmap details
    uint32_t firstMipLevel;
    uint32_t lastMipLevel;
    float minMipLevel;
    float maxMipLevel;
    float mipLevelBias;
    uint32_t mipFilterMode;            // one of CUfilter_mode
} CUtoolsTexRefDescriptor;

typedef struct CUtoolsSurfRefDescriptor_st
{
    uint32_t struct_size;
    uint32_t surfRefBindingType;       // one of CUtools_surf_ref_binding_type

    CUmodule mod;
    const char *pName;                 // If from CUmodule, this is the programmatic name.

    CUarray hArray;                    // array pointer for aliasing blocklinear
    void *devptr;                      // address/size for aliasing malloc
    uint32_t devptrBytes;              //

    uint32_t elementFormat;            // one of CUarray_format
    uint32_t numChannels;
} CUtoolsSurfRefDescriptor;

// Typedefs for function pointers to callbacks, used by
// CtxEnumerateMemBlocks and CtxEnumerateMemObjs.  The
// callbackContext is whatever state you want to provide
// to the callbacks -- it is the same pointer you pass
// into the Enumerate functions.
typedef CUresult (CUDAAPI *CUtoolsMemBlockEnumerateCallback)(
    void *callbackContext,
    CUtoolsMemBlockHandle hMemBlock,
    size_t count);

typedef CUresult (CUDAAPI *CUtoolsMemObjEnumerateCallback)(
    void *callbackContext,
    CUtoolsMemObjHandle hMemObj,
    size_t count);

// Parameter types for Memcpy.  The operands are variant
// types, representing the various types of memory supported
// by the driver.
typedef struct CUtoolsMemcpyBlock3D_st
{
    // Struct not extensible
    uint64_t pitch;     // Offset in bytes between rows
    uint64_t height;    // Number of rows
    uint64_t xInBytes;  // Offset in bytes to origin (x)
    uint64_t y;         // Origin (y)
    uint64_t z;         // Origin (z)
} CUtoolsMemcpyBlock3D;

typedef struct CUtoolsMemcpyOperandArray_st
{
    uint32_t struct_size;
    uint32_t arrayLevelOfDetail;
    CUarray hArray;                   // sizeof(void*)
    void *reserved1;
} CUtoolsMemcpyOperandArray;

typedef struct CUtoolsMemcpyOperandMemObj_st
{
    uint32_t struct_size;
    uint32_t blockValidationOnly;
    CUtoolsMemObjHandle hMemObj;      // sizeof(void*)
    void *reserved0;
    int64_t offset;
    CUtoolsMemcpyBlock3D block3d;
} CUtoolsMemcpyOperandMemObj;

typedef struct CUtoolsMemcpyOperandPageable_st
{
    uint32_t struct_size;
    uint32_t reserved0;
    void *hostAddress;
    void *reserved1;
    CUtoolsMemcpyBlock3D block3d;
} CUtoolsMemcpyOperandPageable;

//  Abstraction for internal type CUImemcpyDataType_enum
typedef enum CUtools_memcpy_operand_type_enum
{
    CU_TOOLS_MEMCPY_OPERAND_TYPE_ARRAY    = 0,
    CU_TOOLS_MEMCPY_OPERAND_TYPE_MEMOBJ   = 1,
    CU_TOOLS_MEMCPY_OPERAND_TYPE_PAGEABLE = 2,
    // --- always add new constants to the end here ---
    CU_TOOLS_MEMCPY_OPERAND_TYPE_SIZE,
    CU_TOOLS_MEMCPY_OPERAND_TYPE_FORCE_INT            = 0x7fffffff
} CUtools_memcpy_operand_type;

typedef struct CUtoolsMemcpyOperand_st
{
    uint32_t type;       // One of CUtools_memcpy_operand_type
    uint32_t reserved0;

    union {
        CUtoolsMemcpyOperandPageable pageable;
        CUtoolsMemcpyOperandArray array;
        CUtoolsMemcpyOperandMemObj memobj;
    } op;
} CUtoolsMemcpyOperand;

typedef struct CUtoolsMemcpyExtent_st
{
    uint32_t struct_size;
    uint32_t reserved0;

    uint64_t widthInBytes;
    uint64_t height;
    uint64_t depth;
} CUtoolsMemcpyExtent;

typedef struct CUtoolsMemcpyExOptions_st
{
    uint32_t struct_size;
    uint32_t reserved0;

    uint32_t type;      // Takes one of the copy types from CUtoolsMemcpyType
    uint32_t reserved1;
} CUtoolsMemcpyExOptions;

// Parameter type for MemGetStatus
typedef struct CUtoolsMemStatus_st
{
    uint32_t struct_size;

    float vmfrag;     // Device virtual address space fragmentation

    uint64_t vmtotal; // Total device virtual memory
    uint64_t vmfree;  // Free device virtual memory
    uint64_t smtotal; // Total host virtual memory
    uint64_t smfree;  // Free host virtual memory
    uint64_t cmtotal; // Total cached system memory
    uint64_t cmfree;  // Free cached system memory
    uint64_t pmtotal; // Total device physical memory
    uint64_t pmfree;  // Free device physical memory
} CUtoolsMemStatus;


// Forward declaration for struct used in parameter below
struct NvBlockLinearImageInfoRec;

/// \brief ToolsMemory provides functions to deal with any kind of memory -- global, array, etc.
CU_DEFINE_UUID(CU_ETID_ToolsMemory,
    0x2d43dbbf, 0x3cbf, 0x4a5a, 0x94, 0x5e, 0xb3, 0x40, 0x29, 0xe8, 0x1e, 0x75);

typedef struct CUetblToolsMemory_st {
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;

    /// \brief Allocate memory within CUDA driver's instruction-RAM heap.
    CUresult (CUDAAPI *DriverAllocInstructionRam)(
        CUcontext ctx,
        CUtoolsMemBlockHandle hMemBlock, // NO LONGER SUPPORTED, MUST BE ZERO
        uint64_t bytesize,
        CUtoolsMemObjHandle *phMemObj,
        uint64_t *pPc);
    void (CUDAAPI *DriverFreeInstructionRam)(CUcontext ctx, CUtoolsMemObjHandle hMemObj);

    CUresult (CUDAAPI *MemHostGetDevicePointer)(CUcontext ctx, void *pHostAllocation, void **pDevPtr);

    /// \brief Get underlying handles for the current version of an array allocation.
    CUresult (CUDAAPI *ArrayGetMemObj)(
        CUtoolsMemObjHandle *phMemObj,
        size_t *pStartOffset,
        CUarray hArray);

    /// \brief Get block-linear info for CUarray.
    /// Include $DRIVER_BRANCH/drivers/common/inc/nvBlockLinear.h
    /// to get the definition for struct NvBlockLinearImageInfoRec.
    CUresult (CUDAAPI *ArrayGetBlockLinearImageInfo)(
        CUcontext ctx,
        CUarray hArray,
        struct NvBlockLinearImageInfoRec **ppBlockLinearImageInfo);

    /// \brief Get a detailed description of a texture-reference.
    CUresult (CUDAAPI *GetTexRefDescriptor)(
        CUtexref texref, 
        CUtoolsTexRefDescriptor* pDescriptor);

    /// \brief Get the memblock for this memobj
    CUresult (CUDAAPI *MemObjGetMemBlock)(
        CUtoolsMemObjHandle hMemObj,
        CUtoolsMemBlockHandle *phMemBlock);

    /// \brief Allocate host-memory that is also mapped to a device-pointer.
    ///     This is memory that can be accessed by both CPU and GPU threads concurrently.
    ///     The allocation is not 'portable' -- it's only mapped within the specified ctx.
    /// \param ppHostMem (*ppHostMem) will be assigned the host-pointer
    /// \param ppDeviceMem May be zero.  (*ppDeviceMem) will be assigned the device-pointer
    /// \param phMemObj May be zero.  (*phMemObj) will be assigned the allocation's MemObjHandle.
    CUresult (CUDAAPI *MemHostAllocDeviceMapped)(
        CUcontext ctx,
        size_t sizeInBytes,
        void **ppHostMem,
        void **ppDeviceMem,
        CUtoolsMemObjHandle *phMemObj);

    /// \brief Free an allocation made through MemHostAllocDeviceMapped. 
    /// Call this on the CUcontext's thread to avoid threading issues.
    void (CUDAAPI *MemHostFree)(
        CUcontext ctx, 
        void *p);

    /// \brief Allocate device memory and get a device-pointer.
    ///     This memory is accessible by GPU threads.
    /// Call this on the CUcontext's thread to avoid threading issues.
    /// \param ppHostMem (*ppHostMem) will be assigned the host-pointer
    /// \param ppDeviceMem (*ppDeviceMem) will be assigned the device-pointer
    /// \param phMemObj (*phMemObj) will be assigned the allocation's MemObjHandle.
    CUresult (CUDAAPI *MemDeviceAlloc)(
        CUcontext ctx,
        size_t sizeInBytes,
        void **ppDeviceMem,
        CUtoolsMemObjHandle *phMemObj);

    /// \brief Free an allocation made through MemDeviceAlloc.
    void (CUDAAPI *MemDeviceFree)(
        CUcontext ctx,
        void *pDeviceMem);

    /// \brief When enabled memory alloc will be in sysmem instead of the default,
    /// which is usually vidmem.  These functions apply to all future CUcontexts,
    /// and the values are only read once (at CUcontext creation time).
    /// Threading: Not thread-safe by design.  Only call these at startup.
    void (CUDAAPI *ForceInstructionAllocInSysMem)(uint32_t bEnabled);
    void (CUDAAPI *ForceLMemAllocInSysMem)       (uint32_t bEnabled);
    void (CUDAAPI *ForceCrsStackAllocInSysMem)   (uint32_t bEnabled);

    /// Return host pointer for the queried memobj.
    /// If there is no host pointer, *pp = NULL.
    void (CUDAAPI *MemObjGetHostPointer)(
        CUtoolsMemObjHandle hMemObj,
        void **pp);

    /// Return host pointer for the queried memobj.
    /// If there is no host pointer, *pp = NULL.
    void (CUDAAPI *MemObjGetBlockHostPointer)(
        CUtoolsMemObjHandle hMemObj,
        void **pp);

    /// Free an allocation directly in memobj.
    void (CUDAAPI *MemObjFree)(
        CUtoolsMemObjHandle hMemObj);

    /// Find the memobj for a device ptr.
    CUresult (CUDAAPI *MemObjFindByDevicePtr)(
        CUtoolsMemObjHandle *phMemObj,
        CUcontext ctx,
        void *devptr);

    /// Get the memobj's offset with its parent memblock.
    /// The result should always be a positive offset.
    CUresult (CUDAAPI *MemObjGetBlockOffset)(
        size_t *pOffset,
        CUcontext ctx,
        CUtoolsMemObjHandle hMemObj);

    /// Changes the location of the context's chip-wide LMem allocation.
    /// This function always forces LMem to be realloc'd, even if the
    /// new location happens to match the previous location.
    void (CUDAAPI *CtxForceLMemAllocInSysMem)(
        CUcontext ctx, 
        uint32_t bEnabled);
    
    /// Query a context current local memory configuration.
    CUresult (CUDAAPI *CtxQueryLMemAlloc)(
        CUcontext ctx,
        CUtoolsMemObjHandle *phMemObj,
        CUtools_memory_location *pLocation,
        size_t *pSizeInBytes);

    /// Get the minimum size in bytes of a malloc block
    CUresult (CUDAAPI *GetMallocMinBlockSize)(
        CUcontext ctx,
        size_t *pMallocMinBlockSize);

    /// Translate a virtual address to a device pointer address
    CUresult (CUDAAPI *TranslateVAtoDevicePtr)(
        CUcontext ctx,
        uint64_t vaddr,
        uint64_t *pDptr);

    /// Query the PC for a memobj.
    /// It is the caller's responsibility to specify a memobj
    /// that is allocated in the program region.
    /// For example, a valid memobj may come from a CUfunction
    /// or from DriverAllocInstructionRam().
    CUresult (CUDAAPI *MemObjGetPc)(
        CUcontext ctx,
        CUtoolsMemObjHandle hMemObj,
        uint64_t *pc);

    /// Get a memobj's size.
    CUresult (CUDAAPI *MemObjGetSize)(
        CUtoolsMemObjHandle hMemObj,
        uint64_t *pSize);

    /// Call a callback for all memblocks in a context.
    /// No locks are taken in this function, so be sure
    /// to block other threads from concurrently calling
    /// any memory management functions for this context.
    /// The callbackContext pointer is passed through to
    /// the callbacks unmodified.  If the callback ever
    /// returns a result other than CUDA_SUCCESS, the
    /// loop will break and the result will be returned.
    /// The first callback will provide a null handle and
    /// the number of memblocks, and the remaining callbacks
    /// will provide the handle and an incrementing count
    /// starting from zero.
    CUresult (CUDAAPI *CtxEnumerateMemBlocks)(
        CUcontext ctx,
        CUtoolsMemBlockEnumerateCallback visitMemBlock,
        void *callbackContext);

    /// Call a callback for all memobjs in a memblock.
    /// No locks are taken in this function.  It is safe
    /// to call from a memblock enumerate callback (see
    /// previous function).  The callbackContext pointer
    /// is passed through to the callbacks unmodified.
    /// If the callback ever returns a result other than
    /// CUDA_SUCCESS, the loop will break and the result
    /// will be returned.  The first callback will provide
    /// a null handle and the number of memblocks, and the
    /// remaining callbacks will provide the handle and an
    /// incrementing count starting from zero.
    CUresult (CUDAAPI *CtxEnumerateMemObjs)(
        CUtoolsMemBlockHandle hMemBlock,
        CUtoolsMemObjEnumerateCallback visitMemObj,
        void *callbackContext);

    /// Copy data from anywhere to anywhere.  See info for
    /// CUtoolsMemcpyOperand for supported src/dst types.
    /// Supports 3D block copying where appropriate.  After
    /// the function returns, it is safe to modify pageable
    /// host memory locations used as parameters, but the
    /// stream must be synchronized to guarantee any device-
    /// visible memory writes (including to pinned host) are
    /// complete.  Implementation is up to the driver.
    CUresult (CUDAAPI *Memcpy)(
        const CUtoolsMemcpyOperand *dst,
        const CUtoolsMemcpyOperand *src,
        const CUtoolsMemcpyExtent *extent,
        CUtoolsStreamHandle stream);

    /// Get info about the current status of the memory
    /// manager, including estimates of free space.
    CUresult (CUDAAPI *MemGetStatus)(
        CUcontext ctx,
        CUtoolsMemStatus *pMemStatus);

    /// Get the CUtexref corresponding to a CUtexObject.
    CUresult (CUDAAPI *GetTexrefForTexObject)(
        CUcontext ctx,
        CUtexObject texObject,
        CUtexref *pTexref);

    /// Get a detailed description of a surface reference.
    CUresult (CUDAAPI *GetSurfRefDescriptor)(
        CUsurfref surfref,
        CUtoolsSurfRefDescriptor *pDescriptor);

    /// Get the CUsurfref corresponding to a CUsurfObject.
    CUresult (CUDAAPI *GetSurfrefForSurfObject)(
        CUcontext ctx,
        CUsurfObject surfObject,
        CUsurfref *pSurfref);

    /// Find memobj by virtual address.
    CUresult (CUDAAPI *MemObjFindByDeviceVAddr)(
        CUtoolsMemObjHandle *phMemObj,
        CUcontext ctx,
        uint64_t vaddr);

    /// Get device virtual address for memobj.
    CUresult (CUDAAPI *MemObjGetDeviceVAddr)(
        CUtoolsMemObjHandle hMemObj,
        uint64_t *vaddr);

    /// Ask the driver whether a memblock needs to be
    /// saved/restored for kernel replay.
    CUresult (CUDAAPI *SaveMemBlockForReplay)(
        CUtoolsMemBlockHandle hMemBlock,
        uint8_t *pShouldSave);

    /// Find source ctx by memobj.
    CUresult (CUDAAPI *MemObjGetSourceCtx)(
        CUtoolsMemObjHandle hMemObj,
        CUcontext *ctx);

    /// Fill out an allocation descriptor for either
    CUresult (CUDAAPI *MemObjGetAllocationDescriptor)(
        CUtoolsMemObjHandle hMemObj,
        CUarray hArray,
        uint32_t name,
        CUtoolsMemoryAllocationDescriptor *desc);

    // Force local memory to be mapped in device pointer space
    void (CUDAAPI *ForceLMemMapInDevicePtr)   (uint32_t bEnabled);

    /// \brief Allocate host-memory that is also mapped to a device-pointer.
    ///     This is memory that can be accessed by both CPU and GPU threads concurrently.
    ///     The allocation is not 'portable' -- it's only mapped within the specified ctx.
    /// \param bCacheDeviceEnabled Set if the memory should be cached by the GPU. If set, the user
    ///     has to ensure the coherency of accesses.
    ///     The GPU associated with ctx can coherently write to and read from that pointer.
    ///     The CPU can read coherently that pointer, except if the GPU has a write-back
    ///     L2 cache strategy (GP10b for example).
    ///     Other GPUs can't access coherently that pointer before UVM-Full and can read it
    ///     coherently after (except if the main GPU has a write-back L2 cache strategy).
    /// \param ppHostMem (*ppHostMem) will be assigned the host-pointer
    /// \param ppDeviceMem May be zero.  (*ppDeviceMem) will be assigned the device-pointer
    /// \param phMemObj May be zero.  (*phMemObj) will be assigned the allocation's MemObjHandle.
    CUresult (CUDAAPI *MemHostAllocDeviceMappedWithCaching)(
        CUcontext ctx,
        size_t sizeInBytes,
        uint32_t bCacheDeviceEnabled,
        void **ppHostMem,
        void **ppDeviceMem,
        CUtoolsMemObjHandle *phMemObj);

    /// \brief When enabled, CNP data structures will be allocated in sysmem
    /// instead of the default, which is usually vidmem. These functions apply
    /// to all future CUcontexts, and the values are only read once (at CUcontext
    /// creation time).
    /// Threading: Not thread-safe by design. Only call these at startup
    void (CUDAAPI *ForceCnpDataInSysmem)(uint32_t bEnabled);

    /// Find source device of memobj.
    CUresult (CUDAAPI *MemObjGetSourceDevice)(
        CUtoolsMemObjHandle hMemObj,
        uint32_t *device);

    /// \brief Generic memory allocation API. Location and mapping of the memory
    ///     allocation depends on the options structure provided through pOptions.
    ///     This API is backwards compatible and will use default value for newer fields.
    ///     This API is not forward compatible and will return CUDA_ERROR_NOT_SUPPORTED
    ///     if called with a tool newer than the driver. It's up to the caller to re-call
    ///     this API with an older format or ask the user to update its driver.
    /// \param ctx Context the allocation will be mapped in.
    /// \param sizeInBytes Requested size for the allocation.
    /// \param pOptions Describe the required type of allocation.
    /// \param ppHostMem (optional) Host pointer of the allocation.
    /// \param ppDeviceMem (optional) Device pointer of the allocation.
    /// \param phMemObj (optional) Allocation's MemObjHandle
    CUresult (CUDAAPI *MemAlloc)(
        CUcontext ctx,
        size_t sizeInBytes,
        CUtoolsMemAllocOptions const *pOptions,
        void **ppHostMem,
        void **ppDeviceMem,
        CUtoolsMemObjHandle *phMemObj);

    /// \brief When enabled preemption buffer will be in sysmem instead of vidmem.
    /// These functions apply to all future CUcontexts,
    /// and the values are only read once (at CUcontext creation time).
    /// Threading: Not thread-safe by design.  Only call these at startup.
    void (CUDAAPI *ForcePreemptionBufferAllocInSysMem)(uint32_t bEnabled);

    /// Memcpy with CUtoolsMemcpyExOptions. Supports all
    /// functionalities of Memcpy API with extra options.
    /// See info for CUtoolsMemcpyExOptions for available options.
    CUresult (CUDAAPI *MemcpyEx)(
        const CUtoolsMemcpyOperand *dst,
        const CUtoolsMemcpyOperand *src,
        const CUtoolsMemcpyExtent *extent,
        CUtoolsStreamHandle stream,
        const CUtoolsMemcpyExOptions *options);

} CUetblToolsMemory;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
