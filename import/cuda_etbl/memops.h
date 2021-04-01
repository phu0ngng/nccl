/*
 * Copyright 2018 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_memops_h__
#define __cuda_etbl_memops_h__

#include "cuda_uuid.h"
#include "cuda_packing.h"
#include "cuda_stdint.h"
#include "cuda.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

CU_DEFINE_UUID(CU_ETID_Memops,
               0x2b8eb874, 0xf4c5, 0x484f, 0x90, 0xe4, 0xc6, 0x7c, 0xe8, 0x3f, 0x69, 0xe4);  

/**
 * Flags for ::cuStreamWriteMemory
 */
typedef enum CUstreamWriteMemory_flags_enum {
    CU_STREAM_WRITE_MEMORY_FENCE_SYS         = 0x0, /**< A heavyweight trailing barrier
                                                         which would prevent reordering of egress traffic
                                                         across all interconnects, including P2P over PCIe and NVLink. */
    CU_STREAM_WRITE_MEMORY_FENCE_GPU         = 0x1, /**< A lightweight trailing barrier, at least including
                                                         traffic towards local device memory. */
    CU_STREAM_WRITE_MEMORY_FENCE_NONE        = 0x2  /**< Omits the memory barrier which would normally
                                                         occur after the write. */
} CUstreamWriteMemory_flags;

/**
 * Scopes for CU_STREAM_MEM_OP_MEMORY_BARRIER
 */
typedef enum CUstreamMemoryBarrier_scope_enum {
    CU_STREAM_MEMORY_BARRIER_SCOPE_SYS     = 0x0, /**< heavyweight barrier, prevent reordering of
                                                       egress traffic across all interconnects,
                                                       including P2P over PCIe and NVLink. */
    CU_STREAM_MEMORY_BARRIER_SCOPE_GPU     = 0x1  /**< a lightweight barrier, at least fencing
                                                       traffic towards local device memory. */
} CUstreamMemoryBarrier_scope;

/**
 * Operation types for ::CU_STREAM_MEM_OP_MEMORY_BARRIER
 */
typedef enum CUstreamMemoryBarrier_op_enum {
    CU_STREAM_MEMORY_BARRIER_OP_WRITE_32     = (1<<0), /**< Fences ::CU_STREAM_MEM_OP_WRITE_VALUE_32 operations */
    CU_STREAM_MEMORY_BARRIER_OP_WRITE_64     = (1<<1), /**< Fences ::CU_STREAM_MEM_OP_WRITE_VALUE_64 operations */
    CU_STREAM_MEMORY_BARRIER_OP_WRITE_MEMORY = (1<<2)  /**< Fences ::CU_STREAM_MEM_OP_WRITE_MEMORY operations */
} CUstreamMemoryBarrier_op;

/**
 * A mask representing the set of store operations supported by ::CU_STREAM_MEM_OP_MEMORY_BARRIER
 */
#define CU_STREAM_MEMORY_BARRIER_OP_ALL (CU_STREAM_MEMORY_BARRIER_OP_WRITE_32 | CU_STREAM_MEMORY_BARRIER_OP_WRITE_64 | CU_STREAM_MEMORY_BARRIER_OP_WRITE_MEMORY)

/**
 * New extra operations for ::cueStreamBatchMemOp
 */
#define CU_STREAM_MEM_OP_WRITE_MEMORY ((CUstreamBatchMemOpType)6) /**< Similar to ::cuStreamWriteValue32, but with variable-size payload.
                                                                       Copies the provided source buffer to the pre-registered destination buffer.
                                                                       The source buffer is consumed at the time of the API call and can
                                                                       be reused when the call returns.
                                                                       The destination buffer is written asynchronously in stream order.
                                                                       The maximum byte count can be queried via CU_DEVICE_ATTRIBUTE_MAXIMUM_STREAM_WRITE_MEMORY_SIZE.
                                                                       Optimizes performance of the write portion, and may use an
                                                                       intermediate staging buffer. */
#define CU_STREAM_MEM_OP_MEMORY_BARRIER ((CUstreamBatchMemOpType)7) /**< Issues a standalone memory barrier. It stops
                                                                         operations belonging to the set "after" and issued
                                                                         after the barrier, to be reordered with
                                                                         operations belonging to the set "before" and issued
                                                                         before it, within the distance defined by "scope".
                                                                         See ::CUstreamMemoryBarrier_op and
                                                                         ::CUstreamMemoryBarrier_scope respectively for
                                                                         the available operation types and scopes. */


typedef struct CUstreamMemOpWriteMemoryParams_st {
    CUstreamBatchMemOpType operation;
    CUdeviceptr dst;
    void *src;
    size_t byteCount;
    unsigned int flags;
    CUdeviceptr alias;                 /**< For driver internal use. Initial value is unimportant. */
} CUstreamMemOpWriteMemoryParams;

typedef struct CUstreamMemOpMemoryBarrierParams_st {
    CUstreamBatchMemOpType operation;
    CUstreamMemoryBarrier_scope scope; /**< The barrier orders operations to at least the distance
                                          defined by this scope. See ::CUstreamMemoryBarrier_scope */
    unsigned int set_before;           /**< a bitwise OR mask of ::CUstreamMemoryBarrier_op,
                                          represents the set of operations issued before
                                          the barrier. */
    unsigned int set_after;            /**< a bitwise OR mask of ::CUstreamMemoryBarrier_op,
                                                represents the set of operations issued after
                                                the barrier. */
} CUstreamMemOpMemoryBarrierParams;

#define CU_STREAM_BATCH_MEM_OP_OMIT_TRAILING_MEMBAR (1U<<29) /**< Omit the membar which happens at the end of the
                                                                  batch before the CTS release. This is a breach 
                                                                  of the weak consistency contract which is useful
                                                                  to extend the coverage of the unit tests */
#define CU_STREAM_BATCH_MEM_OP_RELAXED_ORDERING (1U<<30) /**< Submit the memops observing the consistency 
                                                              guarantees as described in the
                                                              "Weak semantics" comment section of
                                                              cuimemops.c */

typedef enum CUmemopAttribute_enum {
    CU_MEMOP_ATTRIBUTE_MAXIMUM_STREAM_WRITE_MEMORY_SIZE = 1, /**< Maximum byte transfer count for ::CU_STREAM_MEM_OP_WRITE_MEMORY. See ::cueStreamBatchMemOp*/
    CU_MEMOP_ATTRIBUTE_CAN_USE_STREAM_WRITE_MEMORY = 2,      /**< ::CU_STREAM_MEM_OP_WRITE_MEMORY is supported in ::cueStreamBatchMemOp. */
    CU_MEMOP_ATTRIBUTE_CAN_USE_STREAM_MEMORY_BARRIER = 3,    /**< ::CU_STREAM_MEM_OP_MEMORY_BARRIER is supported in ::cueStreamBatchMemOp. */
    CU_MEMOP_ATTRIBUTE_MAXIMUM_BATCH_COUNT = 4,              /**< Maximum number of memops which can be passed down to ::cueStreamBatchMemOp and ::cueStreamBatchMemOp */
    CU_MEMOP_ATTRIBUTE_API_VERSION = 5,                      /**< API version implemented in this ETBL. Use the CU_MEMOP_VERSION_COMPATIBLE() predicate to check for compatibility. */
    CU_MEMOP_ATTRIBUTE_MAX
} CUmemopAttribute;

/**
 * Memops ETBL Versioning
 *
 * In general, version numbers will be incremented as follows:
 * - When a backwards-compatible change is made to the structure layout, the
 *   minor version for that structure will be incremented. Applications
 *   built against an older minor version will continue to work with the newer
 *   minor version of the APIs without recompilation.
 * - When a breaking change is made to the APIs, the major version
 *   will be incremented. Applications built against an older major
 *   version require at least recompilation and potentially additional updates
 *   to use the new API.
 */

#define CU_MEMOP_VERSION 0x00010000
#define CU_MEMOP_MAJOR_VERSION_MASK   0xffff0000
#define CU_MEMOP_MINOR_VERSION_MASK   0x0000ffff

#define CU_MEMOP_MAJOR_VERSION(v) \
    (((v) & CU_MEMOP_MAJOR_VERSION_MASK) >> 16)

#define CU_MEMOP_MINOR_VERSION(v) \
    (((v) & CU_MEMOP_MINOR_VERSION_MASK))

#define CU_MEMOP_MAJOR_VERSION_MATCHES(v) \
    (CU_MEMOP_MAJOR_VERSION(v) == CU_MEMOP_MAJOR_VERSION(CU_MEMOP_VERSION))

#define CU_MEMOP_VERSION_COMPATIBLE(v)    \
    (CU_MEMOP_MAJOR_VERSION_MATCHES(v) && \
    (CU_MEMOP_MINOR_VERSION(v) >= (CU_MEMOP_MINOR_VERSION(CU_MEMOP_VERSION))))

typedef struct CUetblMemops_st {
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;

    /**
     * \brief Returns information about the CUDA memory operation attributes for the device
     *
     * Returns in \p *pi the integer value of the attribute \p attrib on device
     * \p dev.
     *
     * The supported attributes are:
     * - ::CU_MEMOP_ATTRIBUTE_MAXIMUM_STREAM_WRITE_MEMORY_SIZE: maximum byte transfer count for ::CU_STREAM_MEM_OP_WRITE_MEMORY;
     * - ::CU_MEMOP_ATTRIBUTE_CAN_USE_STREAM_WRITE_MEMORY: device supports ::CU_STREAM_MEM_OP_WRITE_MEMORY in ::cueStream(Deferred)BatchMemOp;
     * - ::CU_MEMOP_ATTRIBUTE_CAN_USE_STREAM_MEMORY_BARRIER: device supports ::CU_STREAM_MEM_OP_MEMORY_BARRIER in ::cueStream(Deferred)BatchMemOp;
     * - ::CU_MEMOP_ATTRIBUTE_MAXIMUM_BATCH_COUNT: maximum number of memops which can be passed down to ::cueStreamDeferredBatchMemOp and;
     *
     * \param pi     - Returned attribute value
     * \param attrib - Attribute to query
     * \param dev    - Device handle
     *
     * \return
     * ::CUDA_SUCCESS,
     * ::CUDA_ERROR_DEINITIALIZED,
     * ::CUDA_ERROR_NOT_INITIALIZED,
     * ::CUDA_ERROR_INVALID_CONTEXT,
     * ::CUDA_ERROR_INVALID_VALUE,
     * ::CUDA_ERROR_INVALID_DEVICE
     * \notefnerr
     *
     * \sa ::cueStreamBatchMemOp
     */
    CUresult (CUDAAPI *cueMemopGetAttribute)(int *pi, CUmemopAttribute attrib, CUdevice dev);
    
    /**
     * \brief Batch operations to synchronize a stream via memory operations
     *
     * This is an extended version of cuStreamBatchMemOp, accepting CUDA memory operations, extended memory operations and extended flags.
     * The extended operations are:
     * - ::CU_STREAM_MEM_OP_WRITE_MEMORY
     * - ::CU_STREAM_MEM_OP_MEMORY_BARRIER
     *
     * The extended flags are:
     * - ::CU_STREAM_BATCH_MEM_OP_OMIT_TRAILING_MEMBAR: The batch will not have the MEMBAR.SYS that would be normally proceeding the CTS semaphore release.
     * - ::CU_STREAM_BATCH_MEM_OP_RELAXED_ORDERING: The operations will execute respecting the weak semantics instead of the default strong semantics.
     * 
     * \param stream     - The stream to enqueue the operations in.
     * \param count      - The number of operations in \p paramArray. Must be less or equal to CU_MEMOP_ATTRIBUTE_MAXIMUM_BATCH_COUNT.
     * \param paramArray - The types and parameters of the set of operations.
     * \param flags      - Flags for the batch of operations.
     *
     * \return
     * ::CUDA_SUCCESS,
     * ::CUDA_ERROR_INVALID_VALUE,
     * ::CUDA_ERROR_NOT_SUPPORTED
     * \notefnerr
     *
     * \sa ::cuStreamatchMemoOp
     */
    CUresult (CUDAAPI *cueStreamBatchMemOp)(CUstream stream, unsigned int count, CUstreamBatchMemOpParams *paramArray, unsigned int flags);
    CUresult (CUDAAPI *cueStreamBatchMemOp_ptsz)(CUstream stream, unsigned int count, CUstreamBatchMemOpParams *paramArray, unsigned int flags);

    /**
     * \brief Batch operations to synchronize a stream both to a second stream and via memory operations
     *
     *  This is an extended version of cueStreamBatchMemOp, accepting a dependency stream as a parameter.
     *  It reduces CPU overhead by combining multiple CUDA API calls into one:
     *    cuEventCreate(&ev);
     *    cuEventRecord(ev, dependencyStream);
     *    cuStreamWaitEvent(stream, ev, 0);
     *    cueStreamBatchMemOp(stream, count, paramArray, flags);
     *    cuEventDestroy(ev);
     *
     * See ::cueStreamBatchMemOp for the full set of supported operations.
     *
     * \param stream     - The stream to enqueue the operations in.
     * \param dependencyStream - The stream with which \p stream is synchronized. 
     * \param count      - The number of operations in \p paramArray. Must be less or equal to CU_MEMOP_ATTRIBUTE_MAXIMUM_BATCH_COUNT.
     * \param paramArray - The types and parameters of the set of operations.
     * \param flags      - Flags for the batch of operations.
     *
     * \return
     * ::CUDA_SUCCESS,
     * ::CUDA_ERROR_INVALID_VALUE,
     * ::CUDA_ERROR_NOT_SUPPORTED
     * \notefnerr
     *
     * \sa ::cuStreamatchMemoOp, ::cueStreamatchMemoOp
     */
    CUresult (CUDAAPI *cueStreamDeferredBatchMemOp)(CUstream stream, CUstream dependencyStream, unsigned int count, CUstreamBatchMemOpParams *paramArray, unsigned int flags);
    CUresult (CUDAAPI *cueStreamDeferredBatchMemOp_ptsz)(CUstream stream, CUstream dependencyStream, unsigned int count, CUstreamBatchMemOpParams *paramArray, unsigned int flags);
} CUetblMemops;

#if defined(__CUDA_API_PER_THREAD_DEFAULT_STREAM)
    #define cueStreamBatchMemOp            __CUDA_API_PTSZ(cueStreamBatchMemOp)
    #define cueStreamDeferredBatchMemOp    __CUDA_API_PTSZ(cueStreamDeferredBatchMemOp)
#endif


#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
