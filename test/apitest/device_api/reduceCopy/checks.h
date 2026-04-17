/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _REDUCE_COPY_TEST_CHECKS_H_
#define _REDUCE_COPY_TEST_CHECKS_H_

#include <cuda_runtime.h>
#include "nccl.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <gtest/gtest.h>
#include <sstream>
#include <stdexcept>

// Error checking macros for ReduceCopy tests
struct TestAbortException : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

template <typename Fn>
inline void runTestWithAbortHandling(Fn&& fn) {
  try {
    fn();
  } catch (const TestAbortException&) {
    return;
  } catch (const std::exception& e) {
    ADD_FAILURE() << "Unhandled exception: " << e.what();
  } catch (...) {
    ADD_FAILURE() << "Unhandled non-standard exception";
  }
}
#define NCCLCHECK(cmd) do {                               \
  /* Use gtest failure + exception so the test fails but the suite continues. */ \
  ncclResult_t res = cmd;                               \
  if (res != ncclSuccess) {                             \
    std::ostringstream oss;                           \
    oss << "NCCL error " << __FILE__ << ":" << __LINE__ \
      << ": " << ncclGetErrorString(res);           \
    ADD_FAILURE() << oss.str();                       \
    throw TestAbortException(oss.str());              \
  }                                                     \
} while(0)

#define CUDACHECK(cmd) do {                               \
  /* Use gtest failure + exception so the test fails but the suite continues. */ \
  cudaError_t err = cmd;                                \
  if (err != cudaSuccess) {                             \
    std::ostringstream oss;                           \
    oss << "CUDA error " << __FILE__ << ":" << __LINE__ \
      << ": " << cudaGetErrorString(err);           \
    ADD_FAILURE() << oss.str();                       \
    throw TestAbortException(oss.str());              \
  }                                                     \
} while(0)

// Helper functions to wait for communicator state (for non-blocking communicators)
static inline void waitCommState(ncclComm_t comm) {
  ncclResult_t state;
  do {
    NCCLCHECK(ncclCommGetAsyncError(comm, &state));
    if (state == ncclInProgress) {
      usleep(10);
    }
  } while (state == ncclInProgress);
  if (state != ncclSuccess) {
    std::ostringstream oss;
    oss << "NCCL error: communicator failed with " << ncclGetErrorString(state);
    ADD_FAILURE() << oss.str();
    throw TestAbortException(oss.str());
  }
}

static inline void waitCommStateBatch(ncclComm_t* comms, int num) {
  ncclResult_t state;
  int complete;
  do {
    complete = 1;
    for (int i = 0; i < num; ++i) {
      NCCLCHECK(ncclCommGetAsyncError(comms[i], &state));
      if (state == ncclInProgress) {
        complete = 0;
        break;
      } else if (state != ncclSuccess) {
        std::ostringstream oss;
        oss << "NCCL error: communicator " << i << " failed with "
          << ncclGetErrorString(state);
        ADD_FAILURE() << oss.str();
        throw TestAbortException(oss.str());
      }
    }
    if (!complete) {
      usleep(10);
    }
  } while (!complete);
}

// Macro to check NCCL command and wait for completion if ncclInProgress
#define NCCLCHECK_COMM_WAIT(cmd, comm) do {           \
  ncclResult_t res = cmd;                             \
  if (res == ncclInProgress) {                        \
    waitCommState(comm);                           \
  } else if (res != ncclSuccess) {                    \
    std::ostringstream oss;                         \
    oss << "NCCL error " << __FILE__ << ":" << __LINE__ \
      << ": " << ncclGetErrorString(res);        \
    ADD_FAILURE() << oss.str();                     \
    throw TestAbortException(oss.str());            \
  }                                                   \
} while(0)

#define NCCLCHECK_COMM_WAITBATCH(cmd, comms, num) do {        \
  ncclResult_t res = cmd;                                     \
  if (res == ncclInProgress) {                                \
    waitCommStateBatch(comms, num);                         \
  } else if (res != ncclSuccess) {                            \
    std::ostringstream oss;                                 \
    oss << "NCCL error " << __FILE__ << ":" << __LINE__     \
      << ": " << ncclGetErrorString(res);                \
    ADD_FAILURE() << oss.str();                             \
    throw TestAbortException(oss.str());                    \
  }                                                           \
} while(0)

// Non-throwing variants for cleanup paths. Use in destructors and cleanup only.
// Throwing during stack unwinding (e.g. after a prior CUDA/NCCL error) causes
// std::terminate(); these macros never throw so cleanup can run without aborting.
#define CUDACHECK_NO_THROW(cmd) do {                            \
  cudaError_t err = (cmd);                                    \
  if (err != cudaSuccess) {                                    \
    (void)fprintf(stderr, "CUDA (cleanup, non-fatal) %s:%d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
  }                                                           \
} while(0)

#define NCCLCHECK_NO_THROW(cmd) do {                            \
  ncclResult_t res = (cmd);                                   \
  if (res != ncclSuccess) {                                   \
    (void)fprintf(stderr, "NCCL (cleanup, non-fatal) %s:%d: %s\n", \
            __FILE__, __LINE__, ncclGetErrorString(res)); \
  }                                                           \
} while(0)

#endif // _REDUCE_COPY_TEST_CHECKS_H_

