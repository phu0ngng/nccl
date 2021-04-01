/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_rm_h__
#define __cuda_etbl_tools_rm_h__

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"
#include "nvtypes.h"

#include "cuda_etbl/tools_common_types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus


/// For CUetblToolsRm::GetContextHandles()
typedef struct CUtoolsContextHandlesRm_st {
    /// The struct_size field will always be set to the size in bytes of
    /// the entire structure.
    uint32_t struct_size;

    /// RM NV01_DEVICE ordinal
    NvU32 rmDeviceInstance;
    /// RM NV20_SUBDEVICE ordinal
    NvU32 rmSubDeviceInstance;

    /// RM rmClient
    NvU32 rmClient;
    /// RM rmDevice (NV01_DEVICE handle)
    NvU32 rmDevice;
    /// RM rmSubDevice (NV20_SUBDEVICE handle)
    NvU32 rmSubDevice;
} CUtoolsContextHandlesRm;

//  For CUetblToolsRm::GetMemObjHandles().
typedef struct CUtoolsMemObjHandlesRm_st {
    /// The struct_size field will always be set to the size in bytes of
    /// the entire structure.
    uint32_t struct_size;
    uint32_t reserved0;

    /// RM hMemory
    NvU32 rmMemory;
    /// RM context DMA handle.
    /// Context DMA is used for memory segmentation on Tesla and earlier,
    /// and is a required parameter to NvRmMapMemoryDma.
    NvU32 rmCtxDma;

    /// memBlockBaseAddress is the base-address of the rmMemory.
    /// memblock and rmMemory are 1:1.
    void *memBlockBaseAddress;
    size_t memBlockSizeInBytes;

    /// Virtual address of the underlying memobj.
    void *memObjBaseAddress;
    size_t memObjSizeInBytes;
} CUtoolsMemObjHandlesRm;

// This specifies how to route a call to the appropriate kernel driver.
typedef struct CUtoolsRmRoute_v2_st {
    uint32_t version;          // should be set to CU_TOOLS_RM_ROUTE_VERSION.
    CUtools_driver_type driverModel;
    union {
        struct {
            // CU_TOOLS_DRIVER_TYPE_WDDM
            // This driver model supports two ways of contacting the KMD.
            // Most code can pass a D3DKMT_HANDLE in hAdapter, as in v1 of this structure.
            //   This will use D3DKMTEscape to contact the driver.
            // Alternately, call CUetblToolsWddm::TranslateWddmHandles to convert the
            //   D3DKMT_HANDLEs for adapter, context, and device into opaque kernel handles.
            //   Setting hAdapter to 0 will allow you to pass hkmdAdapter instead, and use
            //   a direct IOCTL instead of the standard WDDM path.
            // It is an error for both hAdapter and hkmdAdapter to be non-zero.
            uint32_t hAdapter;
            uint32_t reserved1;
            uint64_t hkmdAdapter;
        } wddm;
        struct {
            // CU_TOOLS_DRIVER_TYPE_RM (for TCC, Windows 2000/XP)
            uint32_t gpuId;
        } rm;
        struct {
            // CU_TOOLS_DRIVER_TYPE_GLK (Mac OSX)
            CUcontext ctx;
        } glk;
    } dm;
} CUtoolsRmRoute_v2;

typedef CUtoolsRmRoute_v2 CUtoolsRmRoute;
#define CU_TOOLS_RM_ROUTE_V2 (2U)
#define CU_TOOLS_RM_ROUTE_VERSION \
    ((CU_TOOLS_RM_ROUTE_V2 << 24) | sizeof(CUtoolsRmRoute))

// Previous version for compatibility purposes
typedef struct CUtoolsRmRoute_v1_st {
    uint32_t version;          // should be set to CU_TOOLS_RM_ROUTE_VERSION_V1.
    CUtools_driver_type driverModel;
    union {
        struct {
            uint32_t hAdapter; // CU_TOOLS_DRIVER_TYPE_WDDM
        } wddm;
        struct {
            uint32_t gpuId;    // CU_TOOLS_DRIVER_TYPE_RM (for TCC)
        } rm;
    } dm;
    // v1 is a 12 byte structure.  Tools API rules imposed after
    // this version shipped require structures to be multiples of
    // 8 bytes.  Please be sure to force any new versions of this
    // structure to be a multple of 8 on all architectures.
} CUtoolsRmRoute_v1;
#define CU_TOOLS_RM_ROUTE_V1 (1U)
#define CU_TOOLS_RM_ROUTE_VERSION_V1 \
    ((CU_TOOLS_RM_ROUTE_V1 << 24) | sizeof(CUtoolsRmRoute_v1))

// RmControl NV0000_CTRL_CMD_GPU_GET_ID_INFO
#define CU_TOOLS_RM_GPU_INFO_NAMELEN (128U)  // This must never be changed from 128!
typedef struct CUtoolsRmGpuInfo_st {
    size_t struct_size;
    uint32_t deviceInstance;
    uint32_t subDeviceInstance;
    char name[CU_TOOLS_RM_GPU_INFO_NAMELEN];
    CU_32_BIT_PAD_ON_32_BIT_BUILDS(reserved1)
} CUtoolsRmGpuInfo;

// RmControl NV2080_CTRL_CMD_MC_GET_ARCH_INFO
// Clients should include nvcm.h for the #defines for these fields.
typedef struct CUtoolsRmGpuArchInfo_st {
    size_t struct_size;
    uint32_t architecture;
    uint32_t implementation;
    uint32_t revision;
    CU_32_BIT_PAD_ON_64_BIT_BUILDS(reserved1)
} CUtoolsRmGpuArchInfo;

// RM_MAP_MEMORY of NVOS33_PARAMETERS
typedef enum CUtools_rm_map_memory_flags_enum {
    CU_TOOLS_RM_MAP_MEMORY_FLAGS_INVALID            = 0x00000000,
    CU_TOOLS_RM_MAP_MEMORY_FLAGS_READ               = 0x00000001,
    CU_TOOLS_RM_MAP_MEMORY_FLAGS_WRITE              = 0x00000002,
    CU_TOOLS_RM_MAP_MEMORY_FLAGS_READ_WRITE         = 0x00000003,
    CU_TOOLS_RM_MAP_MEMORY_FLAGS_PERSISTENT         = 0x00000004,
    CU_TOOLS_RM_MAP_MEMORY_FLAGS_SKIP_SIZE_CHECK    = 0x00000008,
    CU_TOOLS_RM_MAP_MEMORY_FLAGS_MEM_SPACE_USER     = 0x00004000,
    CU_TOOLS_RM_MAP_MEMORY_FLAGS_forceint           = 0x7fffffff
} CUtools_rm_map_memory_flags;

typedef enum CUtools_rm_create_fb_segment_flags_enum {
    CU_TOOLS_RM_CREATE_FB_SEGMENT_FLAGS_FIXED_OFFSET      = 0x01,  // 0:0
    CU_TOOLS_RM_CREATE_FB_SEGMENT_FLAGS_MAP_CPUVA         = 0x02,  // 1:1
    CU_TOOLS_RM_CREATE_FB_SEGMENT_FLAGS_APERTURE_VIDMEM   = 0x00,  // 3:2
    CU_TOOLS_RM_CREATE_FB_SEGMENT_FLAGS_APERTURE_COH_SYS  = 0x04,  // 3:2
    CU_TOOLS_RM_CREATE_FB_SEGMENT_FLAGS_APERTURE_NCOH_SYS = 0x08,  // 3:2
    CU_TOOLS_RM_CREATE_FB_SEGMENT_FLAGS_CONTIGUOUS        = 0x10,  // 4:4
    CU_TOOLS_RM_CREATE_FB_SEGMENT_FLAGS_GPU_CACHED        = 0x20,  // 5:5
    CU_TOOLS_RM_CREATE_FB_SEGMENT_FLAGS_PAGE_SIZE_4K      = 0x00,  // 6:6
    CU_TOOLS_RM_CREATE_FB_SEGMENT_FLAGS_PAGE_SIZE_BIG     = 0x40,  // 6:6
    CU_TOOLS_RM_CREATE_FB_SEGMENT_FLAGS_forceint          = 0x7fffffff
} CUtools_rm_create_fb_segment_flags;

// NV0080_CTRL_FB_CREATE_FB_SEGMENT_PARAMS
typedef struct CUtoolsCreateFBSegmentParams_st {
    size_t struct_size;
    uint32_t hCtxDma;
    CU_32_BIT_PAD_ON_64_BIT_BUILDS(reserved1)
    uint64_t dmaOffset;
    uint64_t vidOffset;
    uint64_t offset;
    uint64_t length;
    uint64_t validLength;
    uint32_t allocHintHandle;
    CUtools_rm_create_fb_segment_flags flags;
    uint32_t hMemory;
    CU_32_BIT_PAD_ON_64_BIT_BUILDS(reserved2)
    void* cpuAddressOut;
    uint64_t gpuAddressOut;
} CUtoolsCreateFBSegmentParams;

// NV2080_CTRL_CMD_GPU_EXEC_REG_OPS
typedef enum CUtools_gpu_reg_op_command_enum {
    CU_TOOLS_GPU_REG_OP_COMMAND_INVALID  = 0,
    CU_TOOLS_GPU_REG_OP_COMMAND_READ_32  = 1,
    CU_TOOLS_GPU_REG_OP_COMMAND_WRITE_32 = 2,
    CU_TOOLS_GPU_REG_OP_COMMAND_READ_64  = 3,
    CU_TOOLS_GPU_REG_OP_COMMAND_WRITE_64 = 4
} CUtools_gpu_reg_op_command;

// NV2080_CTRL_CMD_GPU_EXEC_REG_OPS
typedef enum CUtools_gpu_reg_op_type_enum {
    CU_TOOLS_GPU_REG_OP_TYPE_INVALID     = 0,
    // Caller specifies GLOBAL to force direct BAR0 access.
    // This can access truly chip-global state, and it can access
    // context-specific state for the GPU context that happens to be
    // swapped in when the call is made.
    CU_TOOLS_GPU_REG_OP_TYPE_GLOBAL      = 1,
    // Caller specifies GR_CTX to request RM to access context-specific
    // state. The state is possibly swapped out or virtualized; RM handles
    // these cases correctly.
    CU_TOOLS_GPU_REG_OP_TYPE_GR_CTX      = 2,
    // Caller specifies GR_CTX_QUAD to request RM to access context-specific
    // state for a specific GPU quadrant. Note that this was introduced with
    // Kepler, and may not make sense on other GPUs. The state is possibly
    // swapped out or virtualized; RM handles these cases correctly.
    CU_TOOLS_GPU_REG_OP_TYPE_GR_CTX_QUAD = 3
} CUtools_gpu_reg_op_type;

// Each REG_OP represents a single BAR0 read or write.
// BAR0 is the name of the 16MiB region in the CPU's physical address space
// where NVIDIA GPU's memory-mapped registers reside.  These registers are
// also called PRI registers.  Caller is generally advised to only do a set
// of reads, or only do a set of writes, per call.
// The *effective* operation of a single REG_OP is described below.
//      Assume uint32_t *pBAR0;
//      READ_32 :
//          regValueLo = *(pBAR0 + regOffset);
//          regValueHi = 0;
//      READ_64 :
//          regValueLo = *(pBAR0 + regOffset + 0);
//          regValueHi = *(pBar0 + regOffset + 4);
//          // Yes, this is defined as two serial 32-bit reads.
//          // This behavior is required by the chip.
//      WRITE_32:
//          // really it's a read-modify-write
//          uint32_t loValue = *(pBAR0 + regOffset);
//          loValue = (loValue & ~regAndNMaskLo) | regValueLo;
//          *(pBAR0 + regOffset) = loValue;
//      WRITE_64:
//          // really it's a read-modify-write
//          uint32_t loValue = *(pBAR0 + regOffset + 0);
//          uint32_t hiValue = *(pBAR0 + regOffset + 4);
//          loValue = (loValue & ~regAndNMaskLo) | regValueLo;
//          hiValue = (hiValue & ~regAndNMaskHi) | regValueHi;
//          *(pBAR0 + regOffset + 0) = loValue;
//          *(pBAR0 + regOffset + 4) = hiValue;
//          // WARNING: RM will corrupt or hang the chip if you do
//          // Read-Modify-Write of only the low or only the high 32-bits.
//          // These are the problematic scenarios:
//          //  #1: regAndNMaskLo==0xffffffff, regAndNMaskHi!=0xffffffff
//          //  #2: regAndNMaskLo!=0xffffffff, regAndNMaskHi==0xffffffff
//          // Caller is generally advised to always write all 64 bits.
typedef struct CUtoolsGpuRegOp_st{
    size_t   struct_size;
    uint8_t  regOp;          // one of CUtools_gpu_reg_op_command
    uint8_t  regType;        // one of CUtools_gpu_reg_op_type
    uint8_t  regStatus;      // zero means success, non-zero means failure
    uint8_t  regInstance;    // selector for regs that alias to the same address (e.g. per-quadrant regs)
    uint32_t regOffset;      // BAR0 offset
    uint32_t regValueHi;     // set-mask high
    uint32_t regValueLo;     // set-mask low
    uint32_t regAndNMaskHi;  // clear-mask high
    uint32_t regAndNMaskLo;  // clear-mask low
    CU_32_BIT_PAD_ON_32_BIT_BUILDS(reserved1)
} CUtoolsGpuRegOp;

// Note: Unlike the RM defines, it is not safe to assume copy engines are indexed in this enum.
// A hypothetical CU_TOOLS_ENGINE_RESET_COPY3 *must* be added to the end of the list, even though
// it would not be equal to 3+CU_TOOLS_ENGINE_RESET_COPY0.
typedef enum CUtools_engine_reset_type_enum {
    CU_TOOLS_ENGINE_RESET_INVALID   = 0,
    CU_TOOLS_ENGINE_RESET_GRAPHICS  = 1,
    CU_TOOLS_ENGINE_RESET_COPY0     = 2,
    CU_TOOLS_ENGINE_RESET_COPY1     = 3,
    CU_TOOLS_ENGINE_RESET_COPY2     = 4,
    CU_TOOLS_ENGINE_RESET_VP        = 5,
    CU_TOOLS_ENGINE_RESET_ME        = 6,
    CU_TOOLS_ENGINE_RESET_PPP       = 7,
    CU_TOOLS_ENGINE_RESET_BSP       = 8,
    CU_TOOLS_ENGINE_RESET_MPEG      = 9,
    CU_TOOLS_ENGINE_RESET_SW        = 10,
    CU_TOOLS_ENGINE_RESET_CIPHER    = 11,
    CU_TOOLS_ENGINE_RESET_VIC       = 12,
    CU_TOOLS_ENGINE_RESET_MSENC     = 13,
    // --- always add new constants to the end here ---
    CU_TOOLS_ENGINE_RESET_SIZE,
    CU_TOOLS_ENGINE_RESET_forceint  = 0x7fffffff
} CUtools_engine_reset_type;

//  The GPC and TPC address of a single virtual SM.
//  See NV2080_CTRL_GR_GET_SM_TO_GPC_TPC_MAPPINGS_PARAMS
typedef struct CUtoolsVsmMapping_st {
     uint32_t gpcId;
     uint32_t tpcId;
} CUtoolsVsmMapping;

typedef struct CUtoolsVsmMappings_st  {
    uint32_t struct_size;   // sizeof(CUtoolsVsmMappings)  size of this struct
    uint32_t entry_size;    // sizeof(CUtoolsVsmMapping)  size of each entry
    uint32_t numSm;         // [out]
    uint32_t entryCount;    // [in] count of CUtoolsVsmMapping's allocated
    CUtoolsVsmMapping* entries;
    void *rsvd1;
} CUtoolsVsmMappings;

/// For CUetblToolsRm: RmAllocOsEvent, RmFreeOsEvent, RmClearOsEvent
typedef struct CUtoolsRmOsEventHandle_st {
    /// The struct_size field will always be set to the size in bytes of
    /// the entire structure.
    uint32_t struct_size;
    uint32_t reserved;
    union {
        struct {
            int32_t fd;
        } linux_descriptor;
    } os;
} CUtoolsRmOsEventHandle;

/// This table allows tools like debuggers to override RM behavior.
CU_DEFINE_UUID(CU_ETID_ToolsRm,
    0x0d180614, 0xd672, 0x47cb, 0xab, 0x6e, 0xb9, 0x63, 0xe5, 0x1b, 0xf4, 0xc6);

typedef struct CUetblToolsRm_st {
    // This export table supports versioning by adding to the end without
    // changing the ETID.  The struct_size field will always be set to the
    // size in bytes of the entire export table structure.
    size_t struct_size;

    /// Get RM handles for the CUcontext.
    CUresult (CUDAAPI *GetContextHandles)(
        CUtoolsContextHandlesRm *pContextHandlesRm,
        CUcontext ctx);

    /// Get the RM rmChannel where the compute engine is bound.
    /// On Tesla and Fermi, this rmChannel has a single underlying GPU channel
    /// where all compute work is pushed; there is only one such rmChannel per
    /// CUcontext on these GPUs.
    CUresult (CUDAAPI *GetComputeEngineChannel)(
        NvU32 *pRmClient,
        NvU32 *pRmChannel,
        CUcontext ctx);

    /// Get RM allocation handles and offset information for a memobj.
    CUresult (CUDAAPI *MemObjGetRmHandles)(
        CUtoolsMemObjHandlesRm *pMemObjHandlesRm,
        CUcontext ctx,
        CUtoolsMemObjHandle hMemObj);

    /// NVIDIA_IOCTL_ENUMERATE_GPUS
    /// Returns the number of physical GPUs that are available through the
    /// requested driverModel.
    size_t (CUDAAPI *RmGetGpuCount)(CUtools_driver_type driverModel);

    /// NVIDIA_IOCTL_ENUMERATE_GPUS
    /// Retrieve up to count GPUIDs for the specified driverModel.
    CUresult (CUDAAPI *RmEnumerateGpus)(
        CUtools_driver_type driverModel,
        uint32_t *pGpuIds,
        size_t count);

    // RmAllocRoot
    CUresult (CUDAAPI *RmAllocRoot)(
        const CUtoolsRmRoute *route,
        NvU32 *pRmClient);

    // RmFree
    CUresult (CUDAAPI *RmFree)(
        const CUtoolsRmRoute *route,
        NvU32 rmClient,
        NvU32 rmParent,
        NvU32 rmObject);

    // RM_ALLOC of NV01_DEVICE_0 + deviceInstance
    CUresult (CUDAAPI *RmAllocDevice)(
        const CUtoolsRmRoute *route,
        NvU32 rmClient,
        NvU32 rmDevice,
        NvU32 deviceInstance);

    // RM_ALLOC of NV20_SUBDEVICE_0 + subDeviceInstance
    CUresult (CUDAAPI *RmAllocSubDevice)(
        const CUtoolsRmRoute *route,
        NvU32 rmClient,
        NvU32 rmDevice,
        NvU32 rmSubDevice,
        NvU32 subDeviceInstance);

    /// RM_ALLOC of NVOS21_PARAMETERS + NV0005_ALLOC_PARAMETERS
    /// On Windows:
    ///     hClass = NV01_EVENT_OS_EVENT
    ///     hIndex = NV2080_NOTIFIERS_GR_DEBUG_INTR
    ///     hEvent = caller-allocated HANDLE returned from CreateEvent()
    CUresult (CUDAAPI *RmAllocDebugEventForSubDevice)(
        const CUtoolsRmRoute *route,
        NvU32 rmClient,
        NvU32 rmSubDevice,
        NvU32 rmDebugEvent,
        void *hEvent);

    /// RmControl NV0000_CTRL_CMD_GPU_GET_ID_INFO
    CUresult (CUDAAPI *RmCtrlGetGpuInfo)(
        const CUtoolsRmRoute *route,
        NvU32 rmClient,
        NvU32 rmGpuId,
        CUtoolsRmGpuInfo *pInfo);

    /// RmControl NV2080_CTRL_CMD_MC_GET_ARCH_INFO
    CUresult (CUDAAPI *RmCtrlGetArchInfo)(
        const CUtoolsRmRoute *route,
        NvU32 rmClient,
        NvU32 rmSubDevice,
        CUtoolsRmGpuArchInfo *pArchInfo);

    /// RmControl NV2080_CTRL_CMD_EVENT_SET_NOTIFICATION for NV2080_NOTIFIERS_GR_DEBUG_INTR
    /// if enable != 0 : NV2080_CTRL_EVENT_SET_NOTIFICATION_ACTION_REPEAT
    /// if enable == 0 : NV2080_CTRL_EVENT_SET_NOTIFICATION_ACTION_DISABLE
    CUresult (CUDAAPI *RmCtrlSetGrDebugNotification)(
        const CUtoolsRmRoute *route,
        NvU32 rmClient,
        NvU32 rmSubDevice,
        uint32_t enable);

    /// offset : On Tesla architecture, offset may be at byte granularity.
    ///          On Fermi+, it must be at page-granularity.
    ///   GPU page size on Fermi is either 64kiB or 128kiB.
    ///   A conservative caller should always align to 128kiB.
    CUresult (CUDAAPI *RmMapMemory)(
        const CUtoolsRmRoute *route,
        NvU32 rmClient,
        NvU32 rmDevice,
        NvU32 rmMemory,
        uint64_t offset,
        uint64_t length,
        CUtools_rm_map_memory_flags flags,
        void **ppMapping);

    // RM_UNMAP_MEMORY of NVOS34_PARAMETERS
    CUresult (CUDAAPI *RmUnmapMemory)(
        const CUtoolsRmRoute *route,
        NvU32 rmClient,
        NvU32 rmDevice,
        NvU32 rmMemory,
        CUtools_rm_map_memory_flags flags,
        void *pMapping);

    // RmControl NV0080_CTRL_CMD_FB_CREATE_FB_SEGMENT
    // Binds hMemory to a GPU physical address range.
    // The hMemory can then be mapped via RmMapMemory.
    CUresult (CUDAAPI *RmCtrlCreateFBSegment)(
        const CUtoolsRmRoute *route,
        NvU32 rmClient,
        NvU32 rmDevice,
        CUtoolsCreateFBSegmentParams *params);

    // RmControl NV0080_CTRL_CMD_FB_DESTROY_FB_SEGMENT
    CUresult (CUDAAPI *RmCtrlDestroyFBSegment)(
        const CUtoolsRmRoute *route,
        NvU32 rmClient,
        NvU32 rmDevice,
        NvU32 rmMemory);

    // RmControl NV2080_CTRL_GR_GET_TESLA_TPC_INFO_PARAMS
    CUresult (CUDAAPI *RmCtrlGetTeslaTpcEnabledMask)(
        const CUtoolsRmRoute *route,
        NvU32 rmClient,
        NvU32 rmSubDevice,
        uint32_t *pTpcEnabledMask);

    // RmControl NV2080_CTRL_CMD_GR_GET_TESLA_SM_INFO
    CUresult (CUDAAPI *RmCtrlGetTeslaSmEnabledMask)(
        const CUtoolsRmRoute *route,
        NvU32 rmClient,
        NvU32 rmSubDevice,
        uint32_t tpcID,
        uint32_t *pSmEnabledMask);

    // RmControl NV2080_CTRL_CMD_GPU_GET_FERMI_GPC_INFO
    CUresult (CUDAAPI *RmCtrlGetFermiGpcEnabledMask)(
        const CUtoolsRmRoute *route,
        NvU32 rmClient,
        NvU32 rmSubDevice,
        uint32_t *pGpcEnabledMask);

    // RmControl NV2080_CTRL_CMD_GPU_GET_FERMI_TPC_INFO
    CUresult (CUDAAPI *RmCtrlGetFermiTpcEnabledMask)(
        const CUtoolsRmRoute *route,
        NvU32 rmClient,
        NvU32 rmSubDevice,
        uint32_t gpcID,
        uint32_t *pTpcEnabledMask);

    // RmControl NV2080_CTRL_CMD_GR_GET_INFO, NV2080_CTRL_GR_INFO_INDEX_MAX_WARPS_PER_SM
    CUresult (CUDAAPI *RmCtrlGetMaxWarpsPerSM)(
        const CUtoolsRmRoute *route,
        NvU32 rmClient,
        NvU32 rmSubDevice,
        uint32_t *pWarpsPerSM);

    // RmControl NV2080_CTRL_CMD_GR_GET_INFO, NV2080_CTRL_GR_INFO_INDEX_SM_REG_BANK_REG_COUNT
    CUresult (CUDAAPI *RmCtrlGetTeslaLRFRowsPerBank)(
        const CUtoolsRmRoute *route,
        NvU32 rmClient,
        NvU32 rmSubDevice,
        uint32_t *pRowsPerBank);

    // NV506F_CTRL_CMD_RESET_CHANNEL
    CUresult (CUDAAPI *RmCtrlResetChannelTesla)(
        const CUtoolsRmRoute *route,
        NvU32 rmClient,
        NvU32 rmChannel,
        CUtools_engine_reset_type engineType);

    // NV906F_CTRL_CMD_RESET_CHANNEL
    CUresult (CUDAAPI *RmCtrlResetChannelFermi)(
        const CUtoolsRmRoute *route,
        NvU32 rmClient,
        NvU32 rmChannel,
        CUtools_engine_reset_type engineType);

    // RmControl NV2080_CTRL_CMD_GPU_EXEC_REG_OPS
    // Execute an array of REG_OPs via the RM.
    // Beware the semantics of this command; the operations are not executed
    // strictly in-order.
    // First all the WRITE_32 and WRITE_64 operations are done (in-order),
    // then all the READ_32 and READ_64 operations are done (in-order).
    // This means that a 'script' containing writes and reads will not work
    // in general, and the caller must invoke this function multiple times
    // to accomplish such tasks.
    CUresult (CUDAAPI *RmCtrlExecRegOps)(
        const CUtoolsRmRoute *route,
        NvU32 rmClient,
        NvU32 rmSubDevice,
        NvU32 rmClientTarget,
        NvU32 rmChannelTarget,
        CUtoolsGpuRegOp *pRegOps,
        size_t regOpCount);

    /// Translate an engine class to a corresponding notifyIndex.
    /// The resultant notifyIndex can be used with RmAllocDebugEventForObject().
    CUresult (CUDAAPI *GetDebugEventNotifyIndexForGrEngineClass)(
        CUtools_gr_engine_class engineClass,
        CUtools_debug_event_notify_index *pNotifyIndex);

    /// Get the RM class instance handle for the compute engine.
    /// This can be used as the parent object of a debug event object.
    /// May only be called if ctx's device is using the RM driver model.
    CUresult (CUDAAPI *GetComputeEngineClassInstanceHandle)(
        CUcontext ctx,
        NvU32 *pRmEngineClass);

    /// RM_ALLOC of NVOS21_PARAMETERS + NV0005_ALLOC_PARAMETERS
    /// On Windows:
    ///     hClass = NV01_EVENT_OS_EVENT
    ///     hEvent = caller-allocated HANDLE returned from CreateEvent()
    /// If the notifyIndex is for a GR engine class, then rmParent
    /// should be a handle to a corresponding class instance.
    /// On Linux: Pass a pointer to the  fd returned from the RmAllocOsEvent function.
    CUresult (CUDAAPI *RmAllocDebugEventForObject)(
        CUtoolsRmRoute *route,
        NvU32 rmClient,
        NvU32 rmParentClient,
        NvU32 rmParent,
        NvU32 rmDebugEvent,
        CUtools_debug_event_notify_index notifyIndex,
        void *hEvent);

    // RmControl NV2080_CTRL_CMD_GPU_GET_NAME_STRING
    // Retrieves a GPU name in Unicode (UTF-16) format.
    // pNameString is supplied by the caller, and is considered to be at least nameStringChars wide.
    // Inadequate buffer sizes will result in truncated output, but pNameString will still be null-terminated.
    CUresult (CUDAAPI *RmCtrlGetGpuNameString)(
        const CUtoolsRmRoute *route,
        NvU32 rmClient,
        NvU32 rmSubDevice,
        wchar_t *pNameString,
        size_t nameStringChars);

    // NVA06F_CTRL_CMD_RESET_CHANNEL
    CUresult (CUDAAPI *RmCtrlResetChannelGK100)(
        const CUtoolsRmRoute *route,
        NvU32 rmClient,
        NvU32 rmChannel,
        CUtools_engine_reset_type engineType);

    // RmControl NV2080_CTRL_CMD_GR_GET_SM_TO_GPC_TPC_MAPPINGS
    // Returns an array that maps VsmId -> { GpcId, TpcId }
    CUresult (CUDAAPI *RmCtrlGetVsmMappings)(
        const CUtoolsRmRoute *route,
        NvU32 rmClient,
        NvU32 rmSubDevice,
        CUtoolsVsmMappings *pParams);

    // Linux: Allocate a handle for use with RmAllocDebugEventForObject.
    // Free the event with RmFreeOsEvent.
    // Unsupported on other platforms.
    CUresult (CUDAAPI *RmAllocOsEvent)(
        const CUtoolsRmRoute *route,
        NvU32 rmClient,
        NvU32 rmDevice,
        CUtoolsRmOsEventHandle *pOsEventHandle);

    // Linux: Free the OS event handle  allocated in RmAllocOsEvent.
    // Unsupported on other platforms.
    CUresult (CUDAAPI *RmFreeOsEvent)(
        const CUtoolsRmRoute *route,
        NvU32 rmClient,
        NvU32 rmDevice,
        const CUtoolsRmOsEventHandle *pOsEventHandle);

    // Linux: Changes file descriptor to a non-signaled state.
    // This will call RmGetEventData to clear the descriptor.
    // Unsupported on other platforms.
    CUresult (CUDAAPI *RmClearOsEvent)(
        const CUtoolsRmRoute *route,
        NvU32 rmClient,
        const CUtoolsRmOsEventHandle *pOsEventHandle);

    // Enable/Disable channel
    CUresult (CUDAAPI *RmChannelEnable)(const CUcontext ctx, NvBool bEnable);

} CUetblToolsRm;


#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
