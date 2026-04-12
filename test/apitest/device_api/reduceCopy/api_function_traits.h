/*************************************************************************
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _REDUCE_COPY_API_FUNCTION_TRAITS_H_
#define _REDUCE_COPY_API_FUNCTION_TRAITS_H_

#include <array>
#include <cstddef>
#if !defined(__CUDA_ARCH__)
#include <string>
#include <vector>
#endif

// Semantic category for dispatch and param adaptation (nSrc/nDst).
enum class ApiFunctionCategory {
  GenericReduceCopy,  // Series 1.x — generic RedOp (LSA/Multimem)
  ReduceSumCopy,     // Series 2.x — sum-specific reduce-copy
  ReduceSum,         // Series 3.x — N->1
  AllGather,         // Series 4.x — Copy/Broadcast 1->N
  ReduceSumCopyNtoM  // Series 5.x — N->M
};

// Single source of truth for each ApiFunctionId: traits and human-readable name.
struct ApiFunctionTraitsEntry {
  bool lambda;           // Supports lambda-based access (offset testing)
  bool local;            // Single-device / local variant
  bool multimemSource;   // Has multimem source
  bool multimemDest;     // Has multimem destination
  bool usesCustomRedOp;  // Uses custom reduce op (Generic/Mul/MultimemCopy_Generic)
  bool isMulVariant;     // Multiplication variant (LsaReduceLsaCopy_Generic_Mul)
  ApiFunctionCategory category;
  int nSrcDefault;       // Default nSrc placeholder (2 for Copy/ReduceSumCopy, etc.)
  int nDstDefault;       // Default nDst (1 for ReduceSum, 2 for others)
  int testingGroup;      // 1=all types, 2=copyOnly types, 3=float only (see X-macro comment)
  const char* name;
};

// Single list: enum and traits table are generated from this (order enforced, co-located).
// X(name, lambda, local, multimemSrc, multimemDest, usesCustomRedOp, isMulVariant, category, nSrcDefault, nDstDefault, testingGroup)
//
// testingGroup — controls which types are tested (matches generate_tests.py / test_matrix.h):
//   1 = all types    (Group 1: owns alignment computation — sizeof(T) affects pack selection)
//   2 = copyOnly types (Group 2: thin wrapper delegating to Group 1; uses explicit-copy type set)
//   3 = float only   (Group 3: pure handle/window translation, no type-specific logic in wrapper)
#define NCCL_REDUCE_COPY_API_FUNC_COUNT 39
#define NCCL_REDUCE_COPY_API_FUNC_LIST \
  /* Series 1.x — Generic ReduceCopy with RedOp (related: Generic, Mul variant, Multimem) */ \
  X(LsaReduceLsaCopy_Generic,           true,  false, false, false, true,  false, GenericReduceCopy, 2, 2, 1) \
  X(LsaReduceLsaCopy_Generic_Mul,       true,  false, false, false, true,  true,  GenericReduceCopy, 2, 2, 1) \
  X(LsaReduceMultimemCopy_Generic,      true,  false, false, true,  true,  false, GenericReduceCopy, 2, 2, 1) \
  /* Series 2.x — Sum-Specific ReduceCopy (thin OpSum wrappers; alignment computed by Series 1) */ \
  X(LsaReduceSumLsaCopy,                true,  false, false, false, false, false, ReduceSumCopy, 2, 2, 2) \
  X(LsaReduceSumMultimemCopy,           true,  false, false, true,  false, false, ReduceSumCopy, 2, 2, 2) \
  X(MultimemReduceSumLsaCopy,            true,  false, true,  false, false, false, ReduceSumCopy, 2, 2, 1) \
  X(MultimemReduceSumMultimemCopy,       true,  false, true,  true,  false, false, ReduceSumCopy, 2, 2, 1) \
  /* Series 3.x — ReduceSum (N->1) */ \
  X(LsaReduceSum_Lambda,                true,  false, false, false, false, false, ReduceSum, 2, 1, 2) \
  X(LsaReduceSum_SymPtr_Team,           false, false, false, false, false, false, ReduceSum, 2, 1, 1) \
  X(LsaReduceSum_SymPtr_DevComm,        false, false, false, false, false, false, ReduceSum, 2, 1, 3) \
  X(LsaReduceSum_Window_Team,           false, false, false, false, false, false, ReduceSum, 2, 1, 3) \
  X(LsaReduceSum_Window_DevComm,        false, false, false, false, false, false, ReduceSum, 2, 1, 3) \
  X(MultimemReduceSum_SymPtr,            false, false, true,  false, false, false, ReduceSum, 2, 1, 2) \
  X(MultimemReduceSum_RawPtr,            false, false, true,  false, false, false, ReduceSum, 2, 1, 2) \
  X(MultimemReduceSum_Window,            false, false, true,  false, false, false, ReduceSum, 2, 1, 3) \
  X(LocalReduceSum_Lambda,              true,  true,  false, false, false, false, ReduceSum, 2, 1, 2) \
  X(LocalReduceSum_Strided,             false, true,  false, false, false, false, ReduceSum, 2, 1, 1) \
  /* Series 4.x — Copy/Broadcast (1->N) */ \
  X(LsaCopy_Lambda,                     true,  false, false, false, false, false, AllGather, 2, 2, 2) \
  X(LsaCopy_SymPtr_Team,                false, false, false, false, false, false, AllGather, 2, 2, 1) \
  X(LsaCopy_SymPtr_DevComm,             false, false, false, false, false, false, AllGather, 2, 2, 3) \
  X(LsaCopy_Window_Team,                false, false, false, false, false, false, AllGather, 2, 2, 3) \
  X(LsaCopy_Window_DevComm,             false, false, false, false, false, false, AllGather, 2, 2, 3) \
  X(MultimemCopy_SymPtr,                 false, false, false, true,  false, false, AllGather, 2, 2, 2) \
  X(MultimemCopy_RawPtr,                 false, false, false, true,  false, false, AllGather, 2, 2, 2) \
  X(MultimemCopy_Window,                 false, false, false, true,  false, false, AllGather, 2, 2, 3) \
  X(LocalCopy_Lambda,                   true,  true,  false, false, false, false, AllGather, 2, 2, 2) \
  X(LocalCopy_Strided,                  false, true,  false, false, false, false, AllGather, 2, 2, 1) \
  /* Series 5.x — ReduceSumCopy (N->M) */ \
  X(LsaReduceSumCopy_SameTeam,          false, false, false, false, false, false, ReduceSumCopyNtoM, 2, 2, 1) \
  X(LsaReduceSumCopy_DevComm,           false, false, false, false, false, false, ReduceSumCopyNtoM, 2, 2, 3) \
  X(LsaReduceSumCopy_Windows,           false, false, false, false, false, false, ReduceSumCopyNtoM, 2, 2, 3) \
  X(LsaReduceSumCopy_DifferentTeams,    false, false, false, false, false, false, ReduceSumCopyNtoM, 2, 2, 1) \
  X(MultimemReduceSumCopy_SymPtr,        false, false, true,  true,  false, false, ReduceSumCopyNtoM, 2, 2, 1) \
  X(MultimemReduceSumCopy_RawPtr,        false, false, true,  true,  false, false, ReduceSumCopyNtoM, 2, 2, 1) \
  X(MultimemReduceSumCopy_Windows,       false, false, true,  true,  false, false, ReduceSumCopyNtoM, 2, 2, 3) \
  X(LsaReduceSumMultimemCopy_SymPtr,    false, false, false, true,  false, false, ReduceSumCopyNtoM, 2, 2, 1) \
  X(LsaReduceSumMultimemCopy_RawPtr,    false, false, false, true,  false, false, ReduceSumCopyNtoM, 2, 2, 1) \
  X(MultimemReduceSumLsaCopy_SymPtr,     false, false, true,  false, false, false, ReduceSumCopyNtoM, 2, 2, 1) \
  X(MultimemReduceSumLsaCopy_RawPtr,     false, false, true,  false, false, false, ReduceSumCopyNtoM, 2, 2, 1) \
  X(LocalReduceSumCopy_Strided,         false, true,  false, false, false, false, ReduceSumCopyNtoM, 2, 2, 1)

// Enum and table generated from the same list (order enforced).
#define X(name, lam, loc, mms, mmd, cust, mul, cat, nsrc, ndst, grp) name,
enum class ApiFunctionId { NCCL_REDUCE_COPY_API_FUNC_LIST };
#undef X

#define X(name, lam, loc, mms, mmd, cust, mul, cat, nsrc, ndst, grp) \
  { lam, loc, mms, mmd, cust, mul, ApiFunctionCategory::cat, nsrc, ndst, grp, #name },
constexpr std::array<ApiFunctionTraitsEntry, NCCL_REDUCE_COPY_API_FUNC_COUNT> kApiFunctionTraitsTable = {{
  NCCL_REDUCE_COPY_API_FUNC_LIST
}};
#undef X

static_assert(static_cast<size_t>(ApiFunctionId::LocalReduceSumCopy_Strided) + 1 == NCCL_REDUCE_COPY_API_FUNC_COUNT,
        "ApiFunctionId count must match NCCL_REDUCE_COPY_API_FUNC_COUNT");

// Device- and host-safe constexpr accessors (used in kernels).
// Implemented as switch statements over the X-macro list so they evaluate purely
// from the enum value at compile time, without needing device-accessible storage
// for kApiFunctionTraitsTable (a host-only constexpr std::array).
__host__ __device__ constexpr bool hasMultimemSourceConstexpr(ApiFunctionId id) {
  switch (id) {
#define X(name, lam, loc, mms, mmd, cust, mul, cat, nsrc, ndst, grp) \
    case ApiFunctionId::name: return mms;
    NCCL_REDUCE_COPY_API_FUNC_LIST
#undef X
  }
  return false;
}
__host__ __device__ constexpr bool hasMultimemDestinationConstexpr(ApiFunctionId id) {
  switch (id) {
#define X(name, lam, loc, mms, mmd, cust, mul, cat, nsrc, ndst, grp) \
    case ApiFunctionId::name: return mmd;
    NCCL_REDUCE_COPY_API_FUNC_LIST
#undef X
  }
  return false;
}
__host__ __device__ constexpr bool isMultimemVariantConstexpr(ApiFunctionId id) {
  return hasMultimemSourceConstexpr(id) || hasMultimemDestinationConstexpr(id);
}
__host__ __device__ constexpr bool isReduceSumVariantConstexpr(ApiFunctionId id) {
  switch (id) {
#define X(name, lam, loc, mms, mmd, cust, mul, cat, nsrc, ndst, grp) \
    case ApiFunctionId::name: return ApiFunctionCategory::cat == ApiFunctionCategory::ReduceSum;
    NCCL_REDUCE_COPY_API_FUNC_LIST
#undef X
  }
  return false;
}
__host__ __device__ constexpr bool isReduceSumCopyVariantConstexpr(ApiFunctionId id) {
  switch (id) {
#define X(name, lam, loc, mms, mmd, cust, mul, cat, nsrc, ndst, grp) \
    case ApiFunctionId::name: \
      return ApiFunctionCategory::cat == ApiFunctionCategory::GenericReduceCopy || \
           ApiFunctionCategory::cat == ApiFunctionCategory::ReduceSumCopy || \
           ApiFunctionCategory::cat == ApiFunctionCategory::ReduceSumCopyNtoM;
    NCCL_REDUCE_COPY_API_FUNC_LIST
#undef X
  }
  return false;
}

// Classification accessors (host and device).
// Switch-based so they evaluate purely from the enum value without needing
// device-accessible storage for kApiFunctionTraitsTable.
__host__ __device__ inline bool apiTraitsIsLambda(ApiFunctionId id) {
  switch (id) {
#define X(name, lam, loc, mms, mmd, cust, mul, cat, nsrc, ndst, grp) \
    case ApiFunctionId::name: return lam;
    NCCL_REDUCE_COPY_API_FUNC_LIST
#undef X
  }
  return false;
}
__host__ __device__ inline bool apiTraitsIsLocal(ApiFunctionId id) {
  switch (id) {
#define X(name, lam, loc, mms, mmd, cust, mul, cat, nsrc, ndst, grp) \
    case ApiFunctionId::name: return loc;
    NCCL_REDUCE_COPY_API_FUNC_LIST
#undef X
  }
  return false;
}
__host__ __device__ inline bool apiTraitsHasMultimemSource(ApiFunctionId id) {
  return hasMultimemSourceConstexpr(id);
}
__host__ __device__ inline bool apiTraitsHasMultimemDestination(ApiFunctionId id) {
  return hasMultimemDestinationConstexpr(id);
}
__host__ __device__ inline bool apiTraitsUsesCustomReduceOp(ApiFunctionId id) {
  switch (id) {
#define X(name, lam, loc, mms, mmd, cust, mul, cat, nsrc, ndst, grp) \
    case ApiFunctionId::name: return cust;
    NCCL_REDUCE_COPY_API_FUNC_LIST
#undef X
  }
  return false;
}
__host__ __device__ inline bool apiTraitsIsMulVariant(ApiFunctionId id) {
  switch (id) {
#define X(name, lam, loc, mms, mmd, cust, mul, cat, nsrc, ndst, grp) \
    case ApiFunctionId::name: return mul;
    NCCL_REDUCE_COPY_API_FUNC_LIST
#undef X
  }
  return false;
}
__host__ __device__ inline ApiFunctionCategory apiTraitsCategory(ApiFunctionId id) {
  switch (id) {
#define X(name, lam, loc, mms, mmd, cust, mul, cat, nsrc, ndst, grp) \
    case ApiFunctionId::name: return ApiFunctionCategory::cat;
    NCCL_REDUCE_COPY_API_FUNC_LIST
#undef X
  }
  return ApiFunctionCategory::GenericReduceCopy;
}
__host__ __device__ inline int apiTraitsNSrcDefault(ApiFunctionId id) {
  switch (id) {
#define X(name, lam, loc, mms, mmd, cust, mul, cat, nsrc, ndst, grp) \
    case ApiFunctionId::name: return nsrc;
    NCCL_REDUCE_COPY_API_FUNC_LIST
#undef X
  }
  return 0;
}
__host__ __device__ inline int apiTraitsNDstDefault(ApiFunctionId id) {
  switch (id) {
#define X(name, lam, loc, mms, mmd, cust, mul, cat, nsrc, ndst, grp) \
    case ApiFunctionId::name: return ndst;
    NCCL_REDUCE_COPY_API_FUNC_LIST
#undef X
  }
  return 0;
}
// Returns the testing group for a function (1=all types, 2=copyOnly types, 3=float only).
__host__ __device__ inline int apiTraitsTestingGroup(ApiFunctionId id) {
  switch (id) {
#define X(name, lam, loc, mms, mmd, cust, mul, cat, nsrc, ndst, grp) \
    case ApiFunctionId::name: return grp;
    NCCL_REDUCE_COPY_API_FUNC_LIST
#undef X
  }
  return 1;
}

__host__ __device__ inline bool apiTraitsIsReduceSumVariant(ApiFunctionId id) {
  return apiTraitsCategory(id) == ApiFunctionCategory::ReduceSum;
}
__host__ __device__ inline bool apiTraitsIsAllGatherVariant(ApiFunctionId id) {
  return apiTraitsCategory(id) == ApiFunctionCategory::AllGather;
}
__host__ __device__ inline bool apiTraitsIsReduceSumCopyVariant(ApiFunctionId id) {
  return isReduceSumCopyVariantConstexpr(id);
}

#if !defined(__CUDA_ARCH__)
inline const std::array<ApiFunctionTraitsEntry, NCCL_REDUCE_COPY_API_FUNC_COUNT>& getApiFunctionTraitsTable() {
  return kApiFunctionTraitsTable;
}
inline const ApiFunctionTraitsEntry& getApiFunctionTraits(ApiFunctionId id) {
  return kApiFunctionTraitsTable[static_cast<size_t>(id)];
}

inline std::string getApiFunctionNameFromTraits(ApiFunctionId id) {
  size_t idx = static_cast<size_t>(id);
  if (idx < kApiFunctionTraitsTable.size()) {
    return std::string(getApiFunctionTraits(id).name);
  }
  return "Unknown_" + std::to_string(static_cast<int>(id));
}

// All API function IDs in enum order (single list for iteration).
inline std::vector<ApiFunctionId> getAllApiFunctionIds() {
  std::vector<ApiFunctionId> ids;
  ids.reserve(NCCL_REDUCE_COPY_API_FUNC_COUNT);
  for (int i = 0; i < NCCL_REDUCE_COPY_API_FUNC_COUNT; ++i) {
    ids.push_back(static_cast<ApiFunctionId>(i));
  }
  return ids;
}
#endif // !__CUDA_ARCH__

#endif // _REDUCE_COPY_API_FUNCTION_TRAITS_H_
