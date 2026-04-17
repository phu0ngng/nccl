/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_REDUCE_COPY_TEST_FUNCTIONS_H
#define NCCL_REDUCE_COPY_TEST_FUNCTIONS_H

#include "config.h"
#include "api_function_traits.h"

// Helpers delegate to central api_function_traits.h (single source of truth).

inline bool usesCustomReduceOp(ApiFunctionId funcId) {
  return apiTraitsUsesCustomReduceOp(funcId);
}

inline bool isReduceSumVariant(ApiFunctionId funcId) {
  return apiTraitsIsReduceSumVariant(funcId);
}

inline bool isMultimemOperation(ApiFunctionId funcId) {
  return apiTraitsHasMultimemSource(funcId) || apiTraitsHasMultimemDestination(funcId);
}

inline bool isAllGatherVariant(ApiFunctionId funcId) {
  return apiTraitsIsAllGatherVariant(funcId);
}

inline bool isLambdaVariantFunction(ApiFunctionId funcId) {
  return apiTraitsIsLambda(funcId);
}

inline bool isMulVariant(ApiFunctionId funcId) {
  return apiTraitsIsMulVariant(funcId);
}

inline bool isReduceSumCopyVariant(ApiFunctionId funcId) {
  return apiTraitsIsReduceSumCopyVariant(funcId);
}

#endif // NCCL_REDUCE_COPY_TEST_FUNCTIONS_H
