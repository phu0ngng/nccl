/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common.h"
#include "nccl_coll.h"
#include "all_reduce.h"
#include "all_gather.h"
#include "reduce_scatter.h"
#include "reduce.h"
#include "broadcast.h"

#define NCOLLS 5
struct ncclColl_t ncclColls[NCOLLS] = {allReduce, allGather, reduceScatter, reduce, broadcast};

int ncclstringtocoll(char *str) {
  for (int t = 0; t < NCOLLS; t++) {
    if (strcmp(str, ncclColls[t].name) == 0) {
      return t;
    }
  }
  PRINT("invalid collective %s, defaulting to %s .. \n", str, ncclColls[0].name);
  return 0;
}

void GetBuffSize(size_t *sendcount, size_t *recvcount, size_t *procSharedCount, int *sameExpected, size_t count, int nranks) {
  *sendcount = *recvcount = *procSharedCount = 1024*1024*1024;
  *sameExpected = 0;
}

void RunTest(struct threadArgs_t* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  FILE* ptr = fopen(args->replayFile, "r");
  if (ptr == NULL) {
    PRINT("Cannot open %s\n", args->replayFile);
    return;
  }

  int lines = 0;
  char collName[20], run_typename[20], run_opname[20];
  size_t count;
  int run_root;
  int nRanks = args->nProcs * args->nThreads * args->nGpus;

  // read file
  while(fscanf(ptr, "%s %zu %s %s %d", collName, &count, run_typename, run_opname, &run_root) != EOF) {
    // set collective
    int collIndex = ncclstringtocoll(collName);
    args->coll = ncclColls+collIndex;

    // set data type and op
    ncclDataType_t run_type = (ncclDataType_t)ncclstringtotype(run_typename);
    ncclRedOp_t run_op = (ncclRedOp_t)ncclstringtoop(run_opname);

    // set size
    if (strcmp(collName, "ReduceScatter") == 0 || strcmp(collName, "AllGather") == 0) {
      count *= nRanks;
    }
    args->minbytes = args->maxbytes = count * wordSize(run_type);

    // run
    TimeTest(args, run_type, run_typename, run_op, run_opname, run_root);
    lines++;
  }

  PRINT("# Ran a total of %d collectives\n", lines);
  fclose(ptr);
}
