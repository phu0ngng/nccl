/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "param.h"

/* Parse user defined rings. Format is like :
 * "0 1|1 0|0 1 2 3|3 2 1 0|0 2 3 1|1 3 2 0|0 1 2 3 4 5 6 7|7 6 5 4 3 2 1 0"
 * Rings with a non-matching number of ranks are ignored so we can provide
 * rings for multiple cases.
 */
#define MAX_ENV_RANKS 512
static ncclResult_t parseRings(const char* str, int* nringsRet, int nranks, int* prev, int* next) {
  int ranks[MAX_ENV_RANKS];
  int nrings = 0;
  int rank = 0;
  int offset = 0;
  int status = 0; // 0 : between numbers, 1 : inside number
  do {
    int digit = str[offset] - '0';
    if (digit >= 0 && digit <= 9) {
      if (status == 0) {
        ranks[rank] = digit;
        status = 1;
      } else {
        ranks[rank] = ranks[rank]*10+digit;
      }
    } else {
      if (status == 1) {
        rank++;
        if (rank == MAX_ENV_RANKS) goto end;
      }
      status = 0;
      if (str[offset] == '|' || str[offset] == '\0') {
        int prevRank = ranks[rank-1];
        // Ignore rings if nranks doesn't match
        if (rank != nranks) goto newring;

        for (int r=0; r<nranks; r++) {
          int rank = ranks[r];
          // Ignore rings with ranks out of bounds
          if (rank < 0 || rank >= nranks) goto newring;
          // Ignore rings with duplicate ranks
          for (int i=0; i<r; i++)
            if (ranks[i] == rank) goto newring;

          next[nrings*nranks+prevRank] = rank;
          prev[nrings*nranks+rank] = prevRank;
          prevRank = rank;
        }
        nrings++;
newring:
        rank = 0;
      }
    }
  } while (str[offset++] != 0);
end:
  *nringsRet = nrings;
  return ncclSuccess;
}

#ifdef __PPC__
// Make the default NCCL_MIN_NRINGS=4 for IBM/Power nodes
#define DEFAULT_MIN_NRINGS 4
#else
#define DEFAULT_MIN_NRINGS 0
#endif
NCCL_PARAM(MinNrings, "MIN_NRINGS", DEFAULT_MIN_NRINGS);
NCCL_PARAM(MaxNrings, "MAX_NRINGS", 0);

/* Users can force the number of threads with an environment variable */
NCCL_PARAM(Nthreads, "NTHREADS", -2);
ncclResult_t getEnvThreads(int* nthreads) {
  int64_t nt = ncclParamNthreads();
  if (nt != -2)
    *nthreads = nt;
  if (*nthreads > NCCL_MAX_NTHREADS) *nthreads = NCCL_MAX_NTHREADS;
  return ncclSuccess;
}
