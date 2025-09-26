/*************************************************************************
 * Copyright (c) 2016-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NET_IB_CONNECT_H_
#define NET_IB_CONNECT_H_

#include "ibvwrap.h"

struct ncclIbQpCreateAttr {
  uint8_t ibPort;
  enum ibv_qp_type type;
  unsigned int accessFlags;
  struct ibv_cq* cq;
  struct ibv_pd* pd;
  uint32_t maxRecvWorkRequest;
  uint32_t maxSendWorkRequest;
};

#endif // NET_IB_CONNECT_H_