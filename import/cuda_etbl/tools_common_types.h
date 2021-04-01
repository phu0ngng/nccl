/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_common_types_h__
#define __cuda_etbl_tools_common_types_h__

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus


//  Types used by callbacks and export tables.  This file is for defining
//  types that are used in multiple Tools API headers.

typedef struct CUtoolsStreamHandle_st         *CUtoolsStreamHandle;
typedef struct CUtoolsTaskHandle_st           *CUtoolsTaskHandle;
typedef struct CUtoolsMemObjHandle_st         *CUtoolsMemObjHandle;
typedef struct CUtoolsMemBlockHandle_st       *CUtoolsMemBlockHandle;
typedef struct CUtoolsChannelHandle_st        *CUtoolsChannelHandle;
typedef struct CUtoolsMem2memBufferHandle_st  *CUtoolsMem2memBufferHandle;

typedef struct CUtoolsNvCurrent_st *CUtoolsNvCurrent;

typedef enum CUtools_channel_use_type_enum {
    CU_TOOLS_CHANNEL_USE_COMPUTE             = 0,
    CU_TOOLS_CHANNEL_USE_ASYNC_MEMORY_H_TO_D = 1,
    CU_TOOLS_CHANNEL_USE_ASYNC_MEMORY_D_TO_H = 2,
    // --- always add new constants to the end here ---
    CU_TOOLS_CHANNEL_USE_SIZE,
    CU_TOOLS_CHANNEL_USE_INT    = 0x7fffffff
} CUtools_channel_use_type;

typedef enum CUtools_channel_type_enum {
    CU_TOOLS_CHANNEL_COMPUTE        = 0,
    CU_TOOLS_CHANNEL_ASYNC_MEMCPY_0 = 1,
    CU_TOOLS_CHANNEL_ASYNC_MEMCPY_1 = 2,
    // --- always add new constants to the end here ---
    CU_TOOLS_CHANNEL_SIZE,
    CU_TOOLS_CHANNEL_INT    = 0x7fffffff
} CUtools_channel_type;

typedef enum CUtools_engine_type_enum {
    CU_TOOLS_ENGINE_COMPUTE      = 0,
    CU_TOOLS_ENGINE_TWOD         = 1,
    CU_TOOLS_ENGINE_MEM2MEM      = 2,
    CU_TOOLS_ENGINE_ASYNC_MEMCPY = 3,
    CU_TOOLS_ENGINE_SW           = 4,
    // --- always add new constants to the end here ---
    CU_TOOLS_ENGINE_SIZE,
    CU_TOOLS_ENGINE_FORCE_INT    = 0x7fffffff
} CUtools_engine_type;

/// Type of driver active on a device.
typedef enum CUtools_driver_type_enum {
    CU_TOOLS_DRIVER_TYPE_INVALID         = 0,
    CU_TOOLS_DRIVER_TYPE_RM              = 1,
    CU_TOOLS_DRIVER_TYPE_WDDM            = 2,
    CU_TOOLS_DRIVER_TYPE_GLK             = 3,
    CU_TOOLS_DRIVER_TYPE_AMODEL          = 4,
    CU_TOOLS_DRIVER_TYPE_MPS             = 5,
    CU_TOOLS_DRIVER_TYPE_MRM             = 6,
    CU_TOOLS_DRIVER_TYPE_REMOVED_1       = 7,
    // --- always add new constants to the end here ---
    CU_TOOLS_DRIVER_TYPE_SIZE,
    CU_TOOLS_DRIVER_TYPE_FORCE_INT       = 0x7fffffff
} CUtools_driver_type;

typedef enum CUtools_gr_engine_type_enum {
    CU_TOOLS_GR_ENGINE_TYPE_INVALID         = 0,
    CU_TOOLS_GR_ENGINE_TYPE_GRAPHICS        = 1,
    CU_TOOLS_GR_ENGINE_TYPE_COMPUTE         = 2,
    CU_TOOLS_GR_ENGINE_TYPE_SIZE,
    CU_TOOLS_GR_ENGINE_TYPE_forceint        = 0x7fffffff
} CUtools_gr_engine_type;

/// HW Engine class.
typedef enum CUtools_gr_engine_class_enum {
    CU_TOOLS_GR_ENGINE_CLASS_INVALID                        = 0x0000,
    CU_TOOLS_GR_ENGINE_CLASS_FERMI_A                        = 0x0003,
    CU_TOOLS_GR_ENGINE_CLASS_FERMI_B                        = 0x0004,
    CU_TOOLS_GR_ENGINE_CLASS_FERMI_C                        = 0x0005,
    CU_TOOLS_GR_ENGINE_CLASS_FERMI_COMPUTE_A                = 0x0006,
    CU_TOOLS_GR_ENGINE_CLASS_FERMI_COMPUTE_B                = 0x0007,
    CU_TOOLS_GR_ENGINE_CLASS_KEPLER_A                       = 0x0008,
    CU_TOOLS_GR_ENGINE_CLASS_KEPLER_B                       = 0x0009,
    CU_TOOLS_GR_ENGINE_CLASS_KEPLER_COMPUTE_A               = 0x000A,
    CU_TOOLS_GR_ENGINE_CLASS_KEPLER_COMPUTE_B               = 0x000B,
    CU_TOOLS_GR_ENGINE_CLASS_KEPLER_C                       = 0x0011,
    CU_TOOLS_GR_ENGINE_CLASS_MAXWELL_A                      = 0x0012,
    CU_TOOLS_GR_ENGINE_CLASS_MAXWELL_COMPUTE_A              = 0x0013,
    CU_TOOLS_GR_ENGINE_CLASS_MAXWELL_COMPUTE_B              = 0x0014,
    CU_TOOLS_GR_ENGINE_CLASS_MAXWELL_B                      = 0x0015,
    // --- always add new constants to the end here ---
    CU_TOOLS_GR_ENGINE_CLASS_SIZE,
    CU_TOOLS_GR_ENGINE_CLASS_forceint                       = 0x7fffffff
} CUtools_gr_engine_class;

typedef enum CUtools_debug_event_notify_index_enum {
    CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_INVALID               = 0x0000,
    /// NV9097_NOTIFIERS_DEBUG_INTR
    CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_FERMI_A               = 0x0003,
    /// NV9197_NOTIFIERS_DEBUG_INTR
    CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_FERMI_B               = 0x0004,
    /// NV9297_NOTIFIERS_DEBUG_INTR
    CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_FERMI_C               = 0x0005,
    /// NV90C0_NOTIFIERS_DEBUG_INTR
    CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_FERMI_COMPUTE_A       = 0x0006,
    /// NV91C0_NOTIFIERS_DEBUG_INTR
    CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_FERMI_COMPUTE_B       = 0x0007,
    /// NVA097_NOTIFIERS_DEBUG_INTR
    CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_KEPLER_A              = 0x0008,
    /// NVA197_NOTIFIERS_DEBUG_INTR
    CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_KEPLER_B              = 0x0009,
    /// NVA0C0_NOTIFIERS_DEBUG_INTR
    CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_KEPLER_COMPUTE_A      = 0x000A,
    /// NVA1C0_NOTIFIERS_DEBUG_INTR
    CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_KEPLER_COMPUTE_B      = 0x000B,
    // NVA297_NOTIFIERS_DEBUG_INTR
    CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_KEPLER_C              = 0x0011,
    // NVB097_NOTIFIERS_DEBUG_INTR
    CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_MAXWELL_A             = 0x0012,
    // NVB0C0_NOTIFIERS_DEBUG_INTR
    CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_MAXWELL_COMPUTE_A     = 0x0013,
    // NVB1C0_NOTIFIERS_DEBUG_INTR
    CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_MAXWELL_COMPUTE_B     = 0x0014,
    // --- always add new constants to the end here ---
    CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_SIZE,
    CU_TOOLS_DEBUG_EVENT_NOTIFY_INDEX_forceint              = 0x7fffffff
} CUtools_debug_event_notify_index;

typedef enum CUtoolsModuleOwner_enum
{
    CU_TOOLS_MODULE_OWNER_INVALID                      = 0x000,
    CU_TOOLS_MODULE_OWNER_DRIVER                       = 0x001,
    CU_TOOLS_MODULE_OWNER_USER                         = 0x002,
    // --- always add new constants to the end here ---
    CU_TOOLS_MODULE_OWNER_SIZE,
    CU_TOOLS_MODULE_OWNER_FORCE_INT                    = 0x7fffffff
} CUtoolsModuleOwner;

typedef enum CUtoolsModuleVisibility_enum
{
    CU_TOOLS_MODULE_VISIBILITY_VISIBLE                 = 0x000,
    CU_TOOLS_MODULE_VISIBILITY_HIDDEN_SYSCALL          = 0x001,
    CU_TOOLS_MODULE_VISIBILITY_HIDDEN_TRAPHANDLER      = 0x002,
    CU_TOOLS_MODULE_VISIBILITY_HIDDEN_TRAMPOLINE       = 0x003,
    CU_TOOLS_MODULE_VISIBILITY_HIDDEN_ATENTRY          = 0x004,
    CU_TOOLS_MODULE_VISIBILITY_HIDDEN_TOOLS            = 0x005,
    CU_TOOLS_MODULE_VISIBILITY_HIDDEN_MEMBAR_WAR       = 0x006,
    CU_TOOLS_MODULE_VISIBILITY_HIDDEN_EXITFUNCTION     = 0x007,
    CU_TOOLS_MODULE_VISIBILITY_HIDDEN_USER             = 0x008,
    // --- always add new constants to the end here ---
    CU_TOOLS_MODULE_VISIBILITY_SIZE,
    CU_TOOLS_MODULE_VISIBILITY_HIDDEN_UNKNOWN          = 0x7fffffff,
    CU_TOOLS_MODULE_VISIBILITY_FORCE_INT               = 0x7fffffff
} CUtoolsModuleVisibility;

typedef enum CUtools_cnp_support_enum {
    CU_TOOLS_CNP_NOT_SUPPORTED                     = 0x00,
    CU_TOOLS_CNP_SUPPORTED                         = 0x01,
    // --- always add new constants to the end here ---
    CU_TOOLS_CNP_SIZE,
    CU_TOOLS_CNP_FORCE_INT                         = 0x7fffffff
} CUtools_cnp_support;

typedef enum CUtoolsMemType_enum
{
    CU_TOOLS_MEM_TYPE_INVALID           = 0x0000,
    CU_TOOLS_MEM_TYPE_GENERIC           = 0x0001,
    CU_TOOLS_MEM_TYPE_IMAGE             = 0x0002,
    CU_TOOLS_MEM_TYPE_PUSHBUFFER        = 0x0003,
    CU_TOOLS_MEM_TYPE_GPFIFOBUFFER      = 0x0004,
    CU_TOOLS_MEM_TYPE_FUNCTION          = 0x0005,
    CU_TOOLS_MEM_TYPE_CONTEXT_SAVE      = 0x0006,
    CU_TOOLS_MEM_TYPE_TEXTURE           = 0x0007,
    CU_TOOLS_MEM_TYPE_CLH_OOL           = 0x0008,
    CU_TOOLS_MEM_TYPE_CONSTANT          = 0x0009,
    CU_TOOLS_MEM_TYPE_VIRTUAL_CHANNEL   = 0x000a,
    CU_TOOLS_MEM_TYPE_NOTIFIER          = 0x000b,
    CU_TOOLS_MEM_TYPE_SKED_REFLECTED    = 0x000c,
    CU_TOOLS_MEM_TYPE_SHARED_SEMAPHORE  = 0x000d,
    CU_TOOLS_MEM_TYPE_QMD               = 0x000e,
    CU_TOOLS_MEM_TYPE_MANAGED           = 0x000f,

    // --- always add new constants to the end here ---
    CU_TOOLS_MEM_TYPE_SIZE,
} CUtoolsMemType;

typedef enum CUtoolsMemOwner_enum {
    CU_TOOLS_MEM_OWNER_NONE           = 0x0,
    CU_TOOLS_MEM_OWNER_DRIVER         = 0x1,
    CU_TOOLS_MEM_OWNER_USER           = 0x2,

    // --- always add new constants to the end here ---
    CU_TOOLS_MEM_OWNER_SIZE,
} CUtoolsMemOwner;

typedef enum CUtoolsMemApi_enum {
    // driver-internal allocation
    CU_TOOLS_MEM_API_NOT_VISIBLE            = 0x0,

    // user-visible user allocation
    CU_TOOLS_MEM_API_VISIBLE                = 0x1,

    //
    // allocations that can be shared out using P2P or portable host memory
    //

    // user allocation created using cuMemAlloc
    // - can be freed using cuMemFree
    CU_TOOLS_MEM_API_MEM_ALLOC              = 0x2,

    // user array allocation using cuArrayCreate
    // - can be freed using cuArrayDestroy
    // - freeing will remove all portable instances
    CU_TOOLS_MEM_API_ARRAY_CREATE           = 0x3,

    // user host allocation created using cuMemAllocHost
    // or cuMemHostAlloc
    // - can be freed using cuMemFreeHost
    // - freeing will remove all portable instances
    CU_TOOLS_MEM_API_HOST_ALLOC             = 0x4,

    // user host allocation created using cuMemHostRegister
    // - can be freed using cuMemHostUnregister
    // - freeing will remove all portable instances
    CU_TOOLS_MEM_API_HOST_REGISTER          = 0x5,

    //
    // allocations created from other allocations via P2P or portable host memory
    //

    // user allocation created in another context, shared
    // with this context via P2P automatically
    CU_TOOLS_MEM_API_MEM_ALLOC_PORTABLE     = 0x6,

    // user array allocation using cuArrayCreate
    // - can be freed using cuArrayDestroy
    // - freeing will remove all portable instances
    CU_TOOLS_MEM_API_ARRAY_CREATE_PORTABLE  = 0x7,

    // user host allocation created as portable in another
    // context using cuMemHostAlloc, shared to this
    // context
    // - can be freed using cuMemFreeHost
    // - freeing will remove all portable instances
    //   (including in the owning context)
    CU_TOOLS_MEM_API_HOST_ALLOC_PORTABLE    = 0x8,

    // user host allocation created as portable in another
    // context using cuMemHostRegister, shared to this
    // context
    // - can be freed using cuMemHostUnregister
    // - freeing will remove all portable instances
    //   (including in the owning context)
    CU_TOOLS_MEM_API_HOST_REGISTER_PORTABLE = 0x9,


    // user allocation created in another process, shared
    // with this process via P2P if necessary. Created with
    // cuMemSharedOpen and freed with cuMemSharedClose
    CU_TOOLS_MEM_API_IPC                    = 0xa,

    // user managed allocation created using cuMemAllocManaged
    // - can be freed using cuMemFree
    CU_TOOLS_MEM_API_MANAGED_ALLOC          = 0xb,

    // user managed allocation created in another context, shared
    // with this context via P2P automatically
    CU_TOOLS_MEM_API_MANAGED_ALLOC_PORTABLE = 0xc,

    // --- always add new constants to the end here ---
    CU_TOOLS_MEM_API_SIZE,
} CUtoolsMemApi;

typedef enum CUtoolsMemMapHost_enum
{
    CU_TOOLS_MEM_MAP_HOST_NONE        = 0x0,
    CU_TOOLS_MEM_MAP_HOST_VA          = 0x1,

    // --- always add new constants to the end here ---
    CU_TOOLS_MEM_MAP_HOST_SIZE,
} CUtoolsMemMapHost;

typedef enum CUtoolsMemMapDevice_enum
{
    CU_TOOLS_MEM_MAP_DEVICE_NONE             = 0x0,
    CU_TOOLS_MEM_MAP_DEVICE_VA               = 0x1,
    CU_TOOLS_MEM_MAP_DEVICE_PTR_FORCE_32_BIT = 0x2, // Force to the lower 4GB of the address space, even in 64-bit contexts
    CU_TOOLS_MEM_MAP_DEVICE_PTR_FORCE_64_BIT = 0x3, // Force dptr mapping to 64 bit address space, even on 32 bit build. This is needed for mods.
    CU_TOOLS_MEM_MAP_DEVICE_PTR              = 0x4, // Allow to be anywhere in context's device pointer space (which may be 32-bit or 64-bit)
    CU_TOOLS_MEM_MAP_DEVICE_RANGE            = 0x5,
    CU_TOOLS_MEM_MAP_DEVICE_SPECIAL_NO_VA    = 0x6, // Special allocations that run through most of the "map" code, but don't get a VADDR (WDDM context-save/virtual-channel only)

    // --- always add new constants to the end here ---
    CU_TOOLS_MEM_MAP_DEVICE_SIZE,
} CUtoolsMemMapDevice;

typedef enum CUtoolsMemLocation_enum
{
    CU_TOOLS_MEM_LOCATION_INVALID     = 0x0,
    CU_TOOLS_MEM_LOCATION_HOST        = 0x1,
    CU_TOOLS_MEM_LOCATION_DEVICE      = 0x2,
} CUtoolsMemLocation;

typedef enum CUtoolsUvmLiteOwnerType_enum
{
    CU_TOOLS_UVM_LITE_OWNER_TYPE_INVALID                = 0,
    CU_TOOLS_UVM_LITE_OWNER_TYPE_ALL_STREAMS            = 1,
    CU_TOOLS_UVM_LITE_OWNER_TYPE_ONE_STREAM             = 2,
    CU_TOOLS_UVM_LITE_OWNER_TYPE_NO_STREAM              = 3,

    // --- always add new constants to the end here ---
    CU_TOOLS_UVM_LITE_OWNER_TYPE_SIZE,
    CU_TOOLS_UVM_LITE_OWNER_TYPE_FORCE_INT              = 0x7fffffff
} CUtoolsUvmLiteOwnerType;

typedef enum CUtools_allocation_format_enum
{
    CU_TOOLS_ALLOCATION_FORMAT_INVALID                  = 0,
    CU_TOOLS_ALLOCATION_FORMAT_PITCHED_LINEAR           = 1,
    CU_TOOLS_ALLOCATION_FORMAT_BLOCK_LINEAR             = 2,
    // --- always add new constants to the end here ---
    CU_TOOLS_ALLOCATION_FORMAT_SIZE,
    CU_TOOLS_ALLOCATION_FORMAT_FORCE_INT                = 0x7fffffff
} CUtools_allocation_format;

typedef enum CUtools_allocation_name_enum
{
    CU_TOOLS_ALLOCATION_NAME_INVALID                    = 0,
    CU_TOOLS_ALLOCATION_NAME_USER                       = 1,
    CU_TOOLS_ALLOCATION_NAME_DRIVER_INTERNAL            = 2,
    CU_TOOLS_ALLOCATION_NAME_DEVICE_HEAP                = 3,
    CU_TOOLS_ALLOCATION_NAME_SURFACE_POOL               = 4,
    CU_TOOLS_ALLOCATION_NAME_CNP_LAUNCH_QUEUE_HEAD      = 5,
    CU_TOOLS_ALLOCATION_NAME_GL_INTEROP                 = 6,

    // --- always add new constants to the end here ---
    CU_TOOLS_ALLOCATION_NAME_SIZE,
    CU_TOOLS_ALLOCATION_NAME_FORCE_INT                  = 0x7fffffff
} CUtools_allocation_name;

typedef struct CUtoolsMemoryAllocationDescriptor_st
{
    uint32_t struct_size;

    uint8_t allocationLocation;    // one of CU_MEMORYTYPE_* defined in cuda.h
    uint8_t allocationFormat;      // one of CU_TOOLS_ALLOCATION_FORMAT_*
    uint16_t reserved0;
    uint32_t memHostAllocFlags;    // flags: CU_MEMHOSTALLOC_* defined in cuda.h

    //  3D allocation parameters -- only a subset may be active depending on the allocation
    uint32_t dimensionality;       // {1, 2, 3}
    uint64_t pitch;                // the 'width' dimension
    uint64_t height;
    uint64_t depth;

    //  Array-specific attributes
    uint32_t arrayFormat;          // one of CUarray_format
    uint32_t numChannels;          // channels per array element
    uint32_t ownedByDriver;        // boolean, non-zero means internal driver allocation
    uint32_t reserved1;
    uint32_t isDeviceHeap;         // boolean, non-zero means this is device heap
    uint32_t reserved2;
    uint32_t isUVMManaged;         // boolean, non zero means this is a UVM managed allocation
    uint32_t reserved3;
    uint32_t allocationName;       // one of CU_TOOLS_ALLOCATION_NAME_*
    uint32_t reserved4;
    uint32_t uvmLiteOwnerType;     // one of CU_TOOLS_UVM_LITE_OWNER_TYPE
    uint32_t reserved5;
} CUtoolsMemoryAllocationDescriptor;

typedef enum CUtoolsMembarType_enum {
    CU_TOOLS_MEMBAR_TYPE_NONE = 0,
    CU_TOOLS_MEMBAR_TYPE_GL   = 1,
    CU_TOOLS_MEMBAR_TYPE_SYS  = 2,
} CUtoolsMembarType;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
