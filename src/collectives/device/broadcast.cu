/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common.h"
#include "broadcast.h"
#include "collectives.h"

#define UNROLL 8

#if NCCL_OP == 0
IMPL_COLL3(ncclBcast, copy, FuncSum, i8, int8_t);
#endif
