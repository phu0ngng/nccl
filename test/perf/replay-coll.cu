/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 *
 * replay-coll - This test will replay collectives from the specified format found in replay-coll.log
 * This is NOT to be confused with the NCCL diagnostic replay tool
 ************************************************************************/

#include "common.h"
#define PRINT if (is_main_thread) printf

#define NCOLLS 5
struct testColl testColls[NCOLLS] = {
  allReduceTest,
  allGatherTest,
  reduceScatterTest,
  reduceTest,
  broadcastTest
};

int ncclstringtocoll(char *str) {
  for (int t = 0; t < NCOLLS; t++) {
    if (strcmp(str, testColls[t].name) == 0) {
      return t;
    }
  }
  PRINT("invalid collective %s, defaulting to %s .. \n", str, testColls[0].name);
  return 0;
}

void ReplayGetBuffSize(size_t *sendcount, size_t *recvcount, size_t count, int nranks) {
  *sendcount = *recvcount = 1024*1024*1024;
}

testResult_t ReplayRunTest(struct threadArgs* args, int root, ncclDataType_t type, const char* typeName, ncclRedOp_t op, const char* opName) {
  if (args->replayFile == NULL) {
    PRINT("Replay : no input file provided\n");
    return testInternalError;
  }
  FILE* ptr = fopen(args->replayFile, "r");
  if (ptr == NULL) {
    PRINT("Replay : cannot open input file %s\n", args->replayFile);
    return testInternalError;
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
    args->collTest = testColls+collIndex;

    // set data type and op
    ncclDataType_t run_type = (ncclDataType_t)ncclstringtotype(run_typename);
    ncclRedOp_t run_op = (ncclRedOp_t)ncclstringtoop(run_opname);

    // set size
    if (strcmp(collName, "ReduceScatter") == 0 || strcmp(collName, "AllGather") == 0) {
      count *= nRanks;
    }
    args->minbytes = args->maxbytes = count * wordSize(run_type);

    // run
    TESTCHECK(TimeTest(args, run_type, run_typename, run_op, run_opname, run_root));
    lines++;
  }

  PRINT("# Ran a total of %d collectives\n", lines);
  fclose(ptr);
  return testSuccess;
}

struct testEngine ncclTestEngine = {
  ReplayGetBuffSize,
  ReplayRunTest
};

#undef PRINT
