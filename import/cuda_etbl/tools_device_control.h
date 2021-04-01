/*
 * Copyright 1993-2014 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_device_control_h__
#define __cuda_etbl_tools_device_control_h__

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"

#include "cuda_etbl/tools_common_types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus


/// Modes for work distribution on a device.
typedef enum CUtools_device_work_distribution_mode_enum {
    CU_TOOLS_DEVICE_WORK_DISTRIBUTION_MODE_DEFAULT     = 0,
    CU_TOOLS_DEVICE_WORK_DISTRIBUTION_MODE_ROUND_ROBIN = 1,
    CU_TOOLS_DEVICE_WORK_DISTRIBUTION_MODE_DYNAMIC     = 2,
    // --- always add new constants to the end here ---
    CU_TOOLS_DEVICE_WORK_DISTRIBUTION_MODE_SIZE,
    CU_TOOLS_DEVICE_WORK_DISTRIBUTION_MODE_INT         = 0x7fffffff
} CUtools_device_work_distribution_mode;

/// Clock types for a device.
typedef enum CUtools_device_clock_type_enum {
    CU_TOOLS_DEVICE_CLOCK_TYPE_MAX     = 0,
    CU_TOOLS_DEVICE_CLOCK_TYPE_CURRENT = 1,
    // --- always add new constants to the end here ---
    CU_TOOLS_DEVICE_CLOCK_TYPE_SIZE,
    CU_TOOLS_DEVICE_CLOCK_TYPE_INT     = 0x7fffffff
} CUtools_device_clock_type;

/// Clock domains for a device.
typedef enum CUtools_device_clock_domain_enum {
    CU_TOOLS_DEVICE_CLOCK_DOMAIN_GRAPHICS  = 0,
    CU_TOOLS_DEVICE_CLOCK_DOMAIN_PROCESSOR = 1,
    CU_TOOLS_DEVICE_CLOCK_DOMAIN_MEMORY    = 2,
    // --- always add new constants to the end here ---
    CU_TOOLS_DEVICE_CLOCK_DOMAIN_SIZE,
    CU_TOOLS_DEVICE_CLOCK_DOMAIN_INT       = 0x7fffffff
} CUtools_device_clock_domain;

/// \brief The device table provides functions to retrieve state from CUdevice
CU_DEFINE_UUID(CU_ETID_ToolsDeviceControl,
    0xe2e763b9, 0x5d17, 0x4cab, 0xa5, 0x68, 0x97, 0x1c, 0x49, 0xd5, 0xe8, 0x51);

typedef struct CUetblToolsDeviceControl_st {
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;

    /// \brief Get the current clock speeds for the specified CUdevice.
    CUresult (CUDAAPI *GetClock)(
        CUdevice dev,
        CUtools_device_clock_type type,
        CUtools_device_clock_domain domain,
        uint32_t* clock);
        
    /// \brief Set the clock speeds for the specified CUdevice.
    CUresult (CUDAAPI *SetClock)(
        CUdevice dev,
        CUtools_device_clock_domain domain,
        uint32_t clock);
        
    /// \brief Get the current work distibution mode for the specified CUdevice.
    CUresult (CUDAAPI *GetWorkDistributionMode)(
        CUdevice dev,
        CUtools_device_work_distribution_mode *mode);
    
    /// \brief Set the work distibution mode for the specified CUdevice.
    CUresult (CUDAAPI *SetWorkDistributionMode)(
        CUtoolsNvCurrent *pnvCurrent,
        CUdevice dev,
        CUtools_device_work_distribution_mode mode);

    /// \brief Set the maximum number of SMs the scheduler can use for launches
    /// on the specified CUcontext.  NOTE:  Only implemented for Tesla and Fermi
    /// architectures.  For QMD-based architectures, this must be done using a
    /// combination of custom pushbuffer methods and modifying fields in the QMD.
    CUresult (CUDAAPI *SetMaxSmCount)(
        CUtoolsNvCurrent *pnvCurrent,
        CUcontext ctx,
        int32_t maxSmCount);

} CUetblToolsDeviceControl;


#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
