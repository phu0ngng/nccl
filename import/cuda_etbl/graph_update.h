/*
 * Copyright 1993-2019 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_graph_update__
#define __cuda_etbl_graph_update__

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_etbl/tools_common_types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

CU_DEFINE_UUID(CU_ETID_GraphExecUpdate,
    0x56cecf70, 0x5f45, 0x4de8, 0x9b, 0xfc, 0x30, 0x77, 0x30, 0x82, 0xf1, 0xad);

typedef struct CUetblGraphExecUpdate_st {
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;

    /// \brief Determine if an instantiated graph can be updated with a given graph.
    ///
    /// Determine if the given graph is similar enough to allow the parameters in
    /// the instantiated graph to be updated.
    ///
    /// What is supported:
    ///  - Changing kernel parameters.
    ///
    /// What is not supported:
    /// - Changing anything except what is explicitly supported(see above).
    /// - Changing from using normal kernel parameters to extra kernel parameters.
    /// - Child graphs(CU_GRAPH_NODE_TYPE_GRAPH).
    ///
    /// \param hGraphExec The instantiated graph
    /// \param hGraph The graph containing the updated parameters
    /// \param bCanUpdate_out non-zero(true) if the graphs are compatible, otherwise zero(false).
    /// \return CUDA_ERROR_INVALID_VALUE if any of the parameters are NULL.
    ///     CUDA_ERROR_OUT_OF_MEMORY if memory is unavailable. Otherwise, returns CUDA_SUCCESS.
    CUresult (CUDAAPI *GraphExecCanUpdateNodesWithGraph)(
        CUgraphExec hGraphExec,
        CUgraph hGraph,
        int *bCanUpdate_out);

    /// \brief Update an instantiated graph with the given graph
    ///
    /// Allow updating an instantiated graph with another graph which
    /// differs only in kernel parameters.
    ///
    /// Calling this function with arguments for which GraphExecCanUpdateNodesWithGraph()
    /// would have yielded bCanUpdate_out == 0 results in undefined behavior.
    ///
    /// \param hGraphExec The instantiated graph to be updated
    /// \param hGraph The graph containing the updated parameters
    /// \return CUDA_ERROR_INVALID_VALUE if any of the parameters are NULL.
    ///     CUDA_ERROR_OUT_OF_MEMORY if memory is unavailable. Otherwise, returns CUDA_SUCCESS.
    CUresult (CUDAAPI *GraphExecUpdateNodesWithGraph)(
        CUgraphExec hGraphExec,
        CUgraph hGraph);

    /// \brief Try to update an instantiated graph with a given graph.
    ///
    /// Updates the instantiated graph only if a call to cuGraphExecCanUpdateNodesWithGraph()
    /// would have set *bCanUpdate_out to true.  cuGraphExecTryUpdateNodesWithGraph() sets 
    /// *bDidUpdate_out to non-zero (true) if it performed an update or zero (false) otherwise.
    ///
    /// \param hGraphExec The instantiated graph to be updated
    /// \param hGraph The graph containing the updated parameters
    /// \param bDidUpdate_out non-zero(true) if the graph was updated, otherwise zero(false).
    /// \return CUDA_ERROR_INVALID_VALUE if any of the parameters are NULL.
    ///     CUDA_ERROR_OUT_OF_MEMORY if memory is unavailable. Otherwise, returns CUDA_SUCCESS.
    CUresult (CUDAAPI *GraphExecTryUpdateNodesWithGraph)(
        CUgraphExec hGraphExec,
        CUgraph hGraph,
        int *bDidUpdate_out);

} CUetblGraphExecUpdate;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
