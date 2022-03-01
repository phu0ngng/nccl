/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "all_reduce.h"
#include "common.h"
#include "collectives.h"

//int L1CarveoutMode = 0;
//NCCL_PARAM(L1CarveoutMode, "L1_CARVEOUT", -2);
//if (L1CarveoutMode>=0) {
//cudaFuncSetAttribute(IMPL_COLL_R(AllReduce), cudaFuncAttributePreferredSharedMemoryCarveout, 90);
//}

IMPL_COLL_R(AllReduce);
