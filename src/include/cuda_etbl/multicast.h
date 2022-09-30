/*
 * Copyright 2020-2023 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_multicast_h__
#define __cuda_etbl_multicast_h__

#include "cuda.h"
#include "cuda_uuid.h"
#include "nvtypes.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

CU_DEFINE_UUID(CU_ETID_Multicast,
    0x9ae64b3c, 0x1588, 0x40ed, 0xbb, 0x8d, 0x57, 0x38, 0x5b, 0x3c, 0x35, 0x9a);

/**
 * Specifies the properties for a multicast object.
 */
struct multicastObjectProp_st {
    /** Size of multicast object */
    size_t size;
    /**
     * Multicast team size. The number of devices that will participate in
     * backing the multicast space with physical memory
     */
    unsigned int numDevices;
    /** requested ::CUmemAllocationHandleType */
    CUmemAllocationHandleType requestedHandleTypes;
    /** flags for future use, must be zero now */
    unsigned long long flags;
};
typedef struct multicastObjectProp_st multicastObjectProp;

typedef struct CUetblMulticast_st {
    // This export table supports versioning by adding to the end without
    // changing the ETID.  The struct_size field will always be set to the
    // size in bytes of the entire export table structure.
    size_t struct_size;

    /**
     * \brief Indicate if multicast is supported by the given device.
     *
     * \param[out] multicastSupported     A returned value of 1 indicates that
     *                                    multicast is supported by the device;
     *                                    whereas 0 indicates that multicast is
     *                                    is not supported by the device.
     * \param[in]  dev                    The device on which multicast support
     *                                    is queried.
     *
     * \return
     * ::CUDA_SUCCESS,
     * ::CUDA_ERROR_INVALID_VALUE
     */
    CUresult (CUDAAPI *DeviceSupportsMulticast)(int *multicastSupported, CUdevice dev);

    /**
     * \brief Create a generic allocation handle representing a multicast object described by the given properties.
     *
     * This does not create a physical memory allocation. The generic allocation
     * \p handle for the object can be mapped to the address space of calling
     * process via ::cuMemMap. This handle cannot be transmitted directly to
     * other processes (see ::cuMemExportToShareableHandle). The size (see
     * multicastObjectProp::size) of this multicast object must be a multiple of
     * the value given via ::cuMemGetAllocationGranularity flag
     * ::CU_MEM_ALLOC_GRANULARITY_RECOMMENDED.
     *
     * Multicast team size is specified by multicastObjectProp::numDevices. The
     * multicast object must be backed by the specified number of physical mem
     * allocations created via ::cuMemCreate one on each of devices specified by
     * multicastObjectProp::numDevices in order to make the multicast object
     * accessible using a virtual address (\sa etiMulticastBindMem).
     *
     * \param[out] handle       Value of handle returned.
     * \param[in]  prop         Properties of the multicast object to create.
     *
     * \return
     * ::CUDA_SUCCESS,
     * ::CUDA_ERROR_INVALID_VALUE,
     * ::CUDA_ERROR_OUT_OF_MEMORY,
     * ::CUDA_ERROR_INVALID_DEVICE,
     * ::CUDA_ERROR_NOT_INITIALIZED,
     * ::CUDA_ERROR_DEINITIALIZED,
     * ::CUDA_ERROR_NOT_PERMITTED,
     * ::CUDA_ERROR_NOT_SUPPORTED
     *
     * \sa etiMulticastBindMem, ::cuMemCreate, ::cuMemRelease, ::cuMemExportToShareableHandle, ::cuMemImportFromShareableHandle
     */
    CUresult (CUDAAPI *MulticastCreate)(CUmemGenericAllocationHandle *handle, const multicastObjectProp *prop);

    /**
     * \brief Bind a physical memory allocation to a multicast object.
     *
     * Binds a physical memory allocation specified by \p memHandle and created
     * via \sa ::cuMemCreate to a multicast object represented by \p mcHandle
     * and created via etiMulticastCreate. The intended \p size of the bind
     * and the offset in physical memory ( \p memOffset ) must be a multiple of
     * the value given by ::cuMemGetAllocationGranularity flag
     * ::CU_MEM_ALLOC_GRANULARITY_RECOMMENDED.
     * The \p size + \p memOffset must be smaller than the size of the allocated
     * physical memory. Similarly the \p size + \p mcOffset must be smaller than
     * the size of multicast object.
     *
     * \param[in]  mcHandle     Handle representing a multicast object.
     * \param[in]  mcOffset     Offset into multicast va range for attachment,
     *                          must be zero now.
     * \param[in]  memHandle    Handle representing physical memory allocation.
     * \param[in]  memOffset    Offset into the physical memory for attachment.
     * \param[in]  size         size of physical memory that will be bound to
     *                          multicast object.
     * \param[in]  flags        flags for future use, must be zero now.
     *
     * \return
     * ::CUDA_SUCCESS,
     * ::CUDA_ERROR_INVALID_VALUE,
     * ::CUDA_ERROR_INVALID_DEVICE,
     * ::CUDA_ERROR_NOT_INITIALIZED,
     * ::CUDA_ERROR_DEINITIALIZED,
     * ::CUDA_ERROR_NOT_PERMITTED,
     * ::CUDA_ERROR_NOT_SUPPORTED
     *
     * \sa etiMulticastCreate, ::cuMemCreate
     */
    CUresult (CUDAAPI *MulticastBindMem)(CUmemGenericAllocationHandle mcHandle, size_t mcOffset, CUmemGenericAllocationHandle memHandle, size_t memOffset, size_t size, unsigned long long flags);
} CUetblMulticast;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
