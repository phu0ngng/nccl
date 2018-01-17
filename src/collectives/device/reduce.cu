/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common.h"
#include "reduce.h"
#include "collectives.h"

#define UNROLL 4

#if NCCL_OP == 0
IMPL_COLL2(ncclReduce, sum, FuncSum);
#elif NCCL_OP == 1
IMPL_COLL2(ncclReduce, prod, FuncProd);
#elif NCCL_OP == 2
IMPL_COLL2(ncclReduce, min, FuncMin);
#elif NCCL_OP == 3
IMPL_COLL2(ncclReduce, max, FuncMax);
#endif
