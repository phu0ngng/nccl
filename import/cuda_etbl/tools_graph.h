/*
 * Copyright 1993-2018 by NVIDIA Corporation.  All rights reserved.  All
 * information contained herein is proprietary and confidential to NVIDIA
 * Corporation.  Any use, reproduction, or disclosure without the written
 * permission of NVIDIA Corporation is prohibited.
 */

#ifndef __cuda_etbl_tools_graph_h__
#define __cuda_etbl_tools_graph_h__

#include "cuda.h"

#include "cuda_uuid.h"
#include "cuda_etbl/tools_common_types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

CU_DEFINE_UUID(CU_ETID_ToolsGraph,
    0xe337dbc2, 0x75c4, 0x4742, 0xb3, 0xa8, 0xa0, 0x79, 0xf3, 0x72, 0x2, 0xec);

// Flags for reporting supported graph update features. See GraphUpdateGetCapabilities().
#define CUI_GRAPH_FLAG_SUPPORTS_MEMSET_UPDATE (1 << 0)
#define CUI_GRAPH_FLAG_SUPPORTS_MEMCPY_UPDATE (1 << 1)

typedef struct CUtoolsKernelNodeInfo_st {
    uint32_t struct_size;

    CUtoolsStreamHandle stream;
    uint64_t gridId;
} CUtoolsKernelNodeInfo;

typedef struct CUetblToolsGraph_st {
    // This export table supports versioning by adding to the end without changing
    // the ETID.  The struct_size field will always be set to the size in bytes of
    // the entire export table structure.
    size_t struct_size;

    CUresult (CUDAAPI *GraphNodeFindInClone)(
        CUgraphNode *phNode,
        CUgraphNode hOriginalNode,
        CUgraph hClonedGraph);

    CUresult (CUDAAPI *GraphKernelNodeSetParams)(
        CUgraphNode hNode,
        const CUDA_KERNEL_NODE_PARAMS *nodeParams);

    CUresult (CUDAAPI *GraphMemcpyNodeSetParams)(
        CUgraphNode hNode,
        const CUDA_MEMCPY3D *nodeParams);

    CUresult (CUDAAPI *GraphMemsetNodeSetParams)(
        CUgraphNode hNode,
        CUDA_MEMSET_NODE_PARAMS *params);

    CUresult (CUDAAPI *GraphHostNodeSetParams)(
        CUgraphNode hNode,
        CUDA_HOST_NODE_PARAMS *params);

    /// \brief Get all the dependencies of a node
    ///
    /// \p dependencies is filled with at most \p bufferSize nodes. One way to
    /// use this function is to call it a first time with a NULL buffer and
    /// \p bufferSize set to 0. Once the number of dependencies is known,
    /// one can allocate a buffer to the right size and finally call this
    /// function a second time to fill the buffer.
    ///
    /// \param hNode The graph node
    /// \param bufferSize The number of elements of the pre-allocated buffer
    /// \param pDependencies The pre-allocated buffer to be filled
    /// \param pNumDependencies The number of dependencies
    /// \return CUDA_ERROR_INVALID_VALUE if \p bufferSize is not zero and
    ///     \p pDependencies is NULL, or if pNumDependenciesUsed is NULL.
    ///     Otherwise, returns CUDA_SUCCESS.
    CUresult (CUDAAPI *GraphNodeGetDependencies)(
        CUgraphNode hNode,
        size_t bufferSize,
        CUgraphNode *pDependencies,
        size_t *pNumDependencies);

    /// \brief Get all the dependent nodes of a node
    ///
    /// \param hNode The graph node
    /// \param bufferSize The number of elements of the pre-allocated buffer
    /// \param pDependentNodes The pre-allocated buffer to be filled
    /// \param pNumDependentNodes The number of dependent nodes 
    /// \return CUDA_ERROR_INVALID_VALUE if \p bufferSize is not zero and
    ///     \p pDependentNodes is NULL, or if pNumDependentNodes is NULL.
    ///     Otherwise, returns CUDA_SUCCESS.
    CUresult (CUDAAPI *GraphNodeGetDependentNodes)(
        CUgraphNode hNode,
        size_t bufferSize,
        CUgraphNode *pDependentNodes,
        size_t *pNumDependentNodes);

    CUresult (CUDAAPI *GraphNodeGetType)(
        CUgraphNode hNode,
        CUgraphNodeType *type);

    CUresult (CUDAAPI *GraphKernelNodeGetParams)(
        CUgraphNode hNode,
        CUDA_KERNEL_NODE_PARAMS *params);

    CUresult (CUDAAPI *GraphMemcpyNodeGetParams)(
        CUgraphNode hNode,
        CUDA_MEMCPY3D *params);

    CUresult (CUDAAPI *GraphMemsetNodeGetParams)(
        CUgraphNode hNode,
        CUDA_MEMSET_NODE_PARAMS *params);

    CUresult (CUDAAPI *GraphHostNodeGetParams)(
        CUgraphNode hNode,
        CUDA_HOST_NODE_PARAMS *nodeParams);

    /// \brief Get all the nodes of a graph
    ///
    /// \param hGraph The graph
    /// \param bufferSize The number of elements of the pre-allocated buffer
    /// \param pNodes The pre-allocated buffer to be filled
    /// \param pNumNodes The number of nodes
    /// \param onlyRoots Whether only root nodes are needed
    /// \return CUDA_ERROR_INVALID_VALUE if \p bufferSize is not zero and
    ///     \p pNodes is NULL, or if pNumNodes is NULL.
    ///     Otherwise, returns CUDA_SUCCESS.
    CUresult (CUDAAPI *GraphGetNodes)(
        CUgraph hGraph,
        size_t bufferSize,
        CUgraphNode *pNodes,
        size_t *pNumNodes,
        uint8_t onlyRoots);

    CUresult (CUDAAPI *GraphLaunch)(
        CUgraphExec hGraphExec,
        CUtoolsStreamHandle hStream);

    CUresult (CUDAAPI *GraphClone)(
        CUgraph *phGraphClone,
        CUgraph hOriginalGraph);

    CUresult (CUDAAPI *GraphInstantiate)(
        CUgraphExec *phGraphExec,
        CUgraph hGraph,
        CUgraphNode *phErrorNode,
        char *logBuffer,
        size_t bufferSize);

    CUresult (CUDAAPI *GraphDestroy)(
        CUgraph hGraph);

    CUresult (CUDAAPI *GraphExecDestroy)(
        CUgraphExec hGraphExec);

    CUresult (CUDAAPI *GraphGetId)(
        CUgraph graph,
        uint64_t *pId);

    /// \brief Get all the contexts used by an executable graph
    ///
    /// \param graphExec The executable graph
    /// \param bufferSize The number of elements of the pre-allocated buffer
    /// \param pCtxs The pre-allocated buffer of contexts to be filled
    /// \param pNumContextsUsed The number of used contexts
    /// \return CUDA_ERROR_INVALID_VALUE if \p bufferSize is not zero and
    ///     \p pCtxs is NULL, or if \p pNumContextsUsed is NULL.
    ///     Otherwise, returns CUDA_SUCCESS.
    CUresult (CUDAAPI *GraphExecGetCtxs)(
        CUgraphExec graphExec,
        size_t bufferSize,
        CUcontext *pCtxs,
        size_t *pNumContextsUsed);

    /// \brief Get all the streams used by an executable graph
    ///
    /// \param graphExec The executable graph
    /// \param bufferSize The number of elements of the pre-allocated buffer
    /// \param pStreams The pre-allocated buffer of streams to be filled
    /// \param pNumStreamsUsed The number of used streams
    /// \return CUDA_ERROR_INVALID_VALUE if \p bufferSize is not zero and
    ///     \p streams is NULL, or if pNumStreamsUsed is NULL.
    ///     Otherwise, returns CUDA_SUCCESS.
    CUresult (CUDAAPI *GraphExecGetStreams)(
        CUgraphExec graphExec,
        size_t bufferSize,
        CUtoolsStreamHandle *pStreams,
        size_t *pNumStreamsUsed);

    CUresult (CUDAAPI *GraphNodeGetId)(
        CUgraphNode node,
        uint64_t *pId);

    /// \brief Get information related to a kernel graph node
    ///
    /// \p The graph node must have a kernel node type
    ///
    /// \param hNode The graph node
    /// \param pKernelNodeInfo The graph node information
    /// \return CUDA_ERROR_INVALID_VALUE if \p hNode is not a valid
    ///     kernel graph node or if \p pKernelNodeInfo is NULL.
    ///     Otherwise, returns CUDA_SUCCESS.
    CUresult (CUDAAPI *GraphKernelNodeGetInfo)(
        CUgraphNode hNode,
        CUtoolsKernelNodeInfo *pKernelNodeInfo);

    CUresult (CUDAAPI *GraphMemcpyNodeGetStreams)(
        CUgraphNode hNode,
        CUtoolsStreamHandle *phPushStream,
        CUtoolsStreamHandle *phPullStream);

    CUresult (CUDAAPI *GraphMemsetNodeGetStream)(
        CUgraphNode hNode,
        CUtoolsStreamHandle *phStream);

    CUresult (CUDAAPI *GraphChildGraphNodeGetGraph)(
        CUgraphNode hNode,
        CUgraph *phGraph);

    CUresult (CUDAAPI *GraphHostNodeGetStream)(
        CUgraphNode hNode,
        CUtoolsStreamHandle *phStream);

    CUresult (CUDAAPI *GraphEmptyNodeGetStream)(
        CUgraphNode hNode,
        CUtoolsStreamHandle *phStream);

    CUresult (CUDAAPI *GraphSetIsQmdChaining)(
        CUgraphExec graphExec,
        uint8_t isChaining);

    CUresult (CUDAAPI *GraphGetIsQmdChaining)(
        CUgraphExec graphExec,
        uint8_t *isChaining);

    CUresult (CUDAAPI *GraphNodeGetGraph)(
        CUgraphNode hNode,
        CUgraph *phGraph);

    /// \brief Get all the edges of a graph
    ///
    /// \param hGraph The graph
    /// \param bufferSize The number of elements of each pre-allocated buffer
    /// \param pFrom The pre-allocated buffer of edges' source to be filled
    /// \param pTo The pre-allocated buffer of edges' destination to be filled
    /// \param pNumEdges The number of edges
    /// \return CUDA_ERROR_INVALID_VALUE if \p bufferSize is not zero and
    ///     one of the buffer is NULL, or if pNumEdges is NULL.
    ///     Otherwise, returns CUDA_SUCCESS.
    CUresult (CUDAAPI *GraphGetEdges)(
        CUgraph hGraph,
        size_t bufferSize,
        CUgraphNode *pFrom,
        CUgraphNode *pTo,
        size_t *pNumEdges);

    /// \brief Get original node and original graph
    ///
    /// This outputs NULL, if there is no original node
    CUresult (CUDAAPI *GraphNodeFindOrigin)(
        CUgraphNode hClonedNode,
        CUgraph *phOriginalGraph,
        CUgraphNode *phOriginalNode);

    /// \brief Check if graph is executable
    ///
    /// \param hGraph The graph
    /// \param isExecutable return 1 if graph is executable, otherwise 0
    /// \return CUDA_ERROR_INVALID_VALUE if \p hGraph is NULL.
    ///  Otherwise, returns CUDA_SUCCESS
    CUresult (CUDAAPI *GraphIsExecutable)(
        CUgraph hGraph,
        uint8_t *isExecutable);

    CUresult (CUDAAPI *GraphSetIsMemsetToKernelNodeConversion)(
        CUgraphExec graphExec,
        uint8_t isMemsetToKernelNodeConversion);

    /// \brief Report supported capabilites of graph update
    ///
    /// \param flags Bitwise OR of supported feature flags.
    /// \return Always returns CUDA_SUCCESS
    CUresult (CUDAAPI *GraphUpdateGetCapabilities)(int *flags);
} CUetblToolsGraph;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // file guard
