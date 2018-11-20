/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "broadcast.h"

void GetBuffSize(size_t *sendcount, size_t *recvcount, size_t *procSharedCount, int *sameExpected, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  BroadcastGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, procSharedCount, sameExpected, count, nranks);
}

void RunTest(struct threadArgs_t* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  // set coll
  args->coll = &broadcast;

  ncclDataType_t *run_types;
  const char **run_typenames;
  int type_count;
  int begin_root, end_root; 

  if ((int)type != -1) { 
    type_count = 1;
    run_types = &type;
    run_typenames = &typeName;
  } else { 
    type_count = ncclNumTypes;
    run_types = test_types;
    run_typenames = test_typenames;
  }

  if (root != -1) { 
     begin_root = end_root = root;
  } else { 
     begin_root = 0;
     end_root = args->nProcs*args->nThreads*args->nGpus-1;
  }

  for (int i=0; i<type_count; i++) { 
       for (int j=begin_root; j<=end_root; j++) {
          TimeTest(args, run_types[i], run_typenames[i], (ncclRedOp_t)0, "", j);
       }
  }   
}
