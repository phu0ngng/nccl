/*
 * Copyright 1993-2015 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_link_h__
#define __cuda_etbl_tools_link_h__

#include "cuda.h"
#include "cuda_uuid.h"
#include "cuda_stdint.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef uint32_t CUendpoint;

typedef enum CUtoolsLinkType_enum
{
    CU_TOOLS_LINK_TYPE_UNKNOWN      = 0,
    CU_TOOLS_LINK_TYPE_PCI          = 0x10,
    CU_TOOLS_LINK_TYPE_PCIE_GEN1    = 0x11,
    CU_TOOLS_LINK_TYPE_PCIE_GEN2    = 0x12,
    CU_TOOLS_LINK_TYPE_PCIE_GEN3    = 0x13,
    CU_TOOLS_LINK_TYPE_PCIE_GEN4    = 0x14,
    CU_TOOLS_LINK_TYPE_NVLINK       = 0x20,
    CU_TOOLS_LINK_TYPE_NVLINK2      = 0x21,
    CU_TOOLS_LINK_TYPE_NVLINK2_2    = 0x22,
    CU_TOOLS_LINK_TYPE_NVLINK3      = 0x23,

    // --- always add new constants to the end here ---
    CU_TOOLS_LINK_TYPE_SIZE
} CUtoolsLinkType;

typedef struct CUtoolsLinkDesc_st {
    uint32_t            struct_size;
    // Ensure that the size of the struct is a multiple of 64Bits
    uint32_t            padding0;

    uint32_t            type; // one of CUtoolsLinkType
    uint32_t            count;
    uint32_t            bandwidth; // the bandwidth is provided in MBPS
    uint32_t            perfRank;
    uint32_t            p2pSupported;
    uint32_t            atomicSupported;

    CUendpoint          source;
    CUendpoint          destination;

    // When the link is made of several pairs/sublinks sourcePort and
    // destination Port indicate the slot number used for each pairs/sublinks.
    // For instance if NvLink is used between GPU1 and GPU2 and if the link
    // is using two pairs sourcePort will contain the slot ID for the two pairs
    // on GPU1 and destinationPort will contain the slot ID for the same two
    // pairs on GPU2.
    uint32_t            *sourcePort;
    uint32_t            *destinationPort;
    uint32_t            nvswitchConnected;
    // Ensure that the size of the struct is a multiple of 64Bits
    uint32_t            padding2;

    uint32_t            blockLinearAccessSupported;
    // Ensure that the size of the struct is a multiple of 64Bits
    uint32_t            padding3;

    uint32_t            isDirectLink;
    uint32_t            isIbmNpuRelaxedOrderingEnabled;
} CUtoolsLinkDesc;

typedef struct CUtoolsIbmNpuRegistersInfo_st {
    uint32_t            struct_size;

    // Ensure that the size of the struct is a multiple of 64Bits
    uint32_t            padding0;

    uint64_t            currentGeneration64BitsRegistersHostPtr;
    uint64_t            lastCompletedGeneration64BitsRegistersHostPtr;


    uint64_t            currentGeneration64BitsRegistersDevicePtr;
    uint64_t            lastCompletedGeneration64BitsRegistersDevicePtr;
} CUtoolsIbmNpuRegistersInfo;

/// \brief interface for Link information
CU_DEFINE_UUID(CU_ETID_ToolsLink,
    0x794bbd6f, 0x8caa, 0x468e, 0xba, 0xf3, 0xc1, 0x3, 0xb6, 0xd1, 0x8e, 0x3c);

typedef struct CUetblToolsLink_st {
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;

    void* legacy_0;
    void* legacy_1;

    /// \brief Get a descriptor that stores information on the link between two devices
    /// \param source the id of the source device
    /// \param destination the id of the destination device
    /// \param link (out) the link descriptor between source and destination.
    CUresult (CUDAAPI *GetGpuToGpuLinkDescriptor)(
        CUdevice source,
        CUdevice destination,
        CUtoolsLinkDesc *link);

    /// \brief Get a descriptor that stores information on the link between one device and the host.
    /// \param source the id of the source device.
    /// \param link (out) the link descriptor between source and the host.
    CUresult (CUDAAPI *GetGpuToSysLinkDescriptor)(
        CUdevice source,
        CUtoolsLinkDesc *link);

    /// \brief For IBM Power 9 Get the NPU registers informations associated with the specified context.
    /// \param context the cuda context that should be used
    /// \param ibmNpuRegistersInfo (out) a pointer to a structure that will be filled with the registers info
    CUresult(CUDAAPI *GetIbmNpuRegistersInfo)(
        CUcontext context,
        CUtoolsIbmNpuRegistersInfo *info);

} CUetblToolsLink;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif
