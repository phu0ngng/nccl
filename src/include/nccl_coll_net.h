/*************************************************************************
 * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COLL_NET_H_
#define NCCL_COLL_NET_H_

#include "nccl.h"

#define NCCL_COLL_NET_HANDLE_MAXSIZE 64

#define NCCL_PTR_HOST 0x1
#define NCCL_PTR_CUDA 0x2

typedef struct {
  // Name of the network (mainly for logs)
  const char* name;
  // Initialize the network.
  ncclResult_t (*init)(/*ncclDebugLogger_t logFunction*/);  //TODO
  // Return the number of adapters.
  ncclResult_t (*devices)(int* ndev);
  // Return the device path in /sys. NCCL will call free on this path.
  ncclResult_t (*pciPath)(int dev, char** path);
  // Return whether this device supports host pointers and/or CUDA pointers
  // as data from the current GPU. Supported types should be composed with
  // NCCL_PTR_HOST and NCCL_PTR_CUDA.
  ncclResult_t (*ptrSupport)(int dev, int* supportedTypes);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to NCCL_COLL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create connections.
  ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
  // Create a group for collective operations. handles have been created
  // using listen() above.
  ncclResult_t (*connect)(int dev, void* handles[], int nranks, void* listenComm, void** collComm);
  // Returns combinations of reduction operations and data types supported by the collective operations.
  // The support array (of type int) must be of size ncclNumOps*ncclNumTypes.
  // The index for an (operation, type) combination is ncclRedOp*ncclNumTypes+ncclDataType.
  ncclResult_t (*reduceSupport)(int* support);
  // Performs an asynchronous allreduce operation on the collective group.
  // May return request == NULL if the call cannot be performed (or would block).
  // Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
  ncclResult_t (*iallreduce)(void* collComm, void* sendData, void* recvData, int size,
      ncclDataType_t dtype, ncclRedOp_t redOp, int type, void** request);
  // Perform a flush/fence to make sure all data received with NCCL_PTR_CUDA is
  // visible to the GPU
  ncclResult_t (*flush)(void* collComm, void* data, int size);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  ncclResult_t (*test)(void* request, int* done, int* size);
  // Close and free collective comm objects
  ncclResult_t (*closeColl)(void* collComm);
  ncclResult_t (*closeListen)(void* listenComm);
} ncclCollNet_v1_t;


typedef ncclCollNet_v1_t ncclCollNet_t;

#define NCCL_COLLNET_PLUGIN_SYMBOL ncclCollNetPlugin_v1

extern
#ifdef __cplusplus
"C"
#endif
ncclCollNet_t* collNet;

#endif // end include guard
