/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "reduce.h"

void GetBuffSize(size_t *sendcount, size_t *recvcount, size_t *procSharedCount, int *sameExpected, size_t count, int nranks) {
  size_t paramcount, sendInplaceOffset, recvInplaceOffset;
  ReduceGetCollByteCount(sendcount, recvcount, &paramcount, &sendInplaceOffset, &recvInplaceOffset, procSharedCount, sameExpected, count, nranks);
}

void RunTest(struct threadArgs_t* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  // set coll
  args->coll = &reduce;

  ncclDataType_t *run_types;
  ncclRedOp_t *run_ops;
  const char **run_typenames, **run_opnames;
  int type_count, op_count;
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

  if ((int)op != -1) { 
    op_count = 1;
    run_ops = &op;
    run_opnames = &opName;
  } else { 
    op_count = ncclNumOps;
    run_ops = test_ops;
    run_opnames = test_opnames;
  }

  if (root != -1) { 
     begin_root = end_root = root;
  } else { 
     begin_root = 0;
     end_root = args->nProcs*args->nThreads*args->nGpus-1;
  }

  for (int i=0; i<type_count; i++) { 
      for (int j=0; j<op_count; j++) { 
         for (int k=begin_root; k<=end_root; k++) { 
             TimeTest(args, run_types[i], run_typenames[i], run_ops[j], run_opnames[j], k);
         }
      }
  }   
}
