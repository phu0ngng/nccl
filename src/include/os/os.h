/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_OS_H_
#define NCCL_OS_H_

#include "nccl.h"

#include <cstdint>

#ifdef NCCL_OS_WINDOWS
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <iphlpapi.h>
#endif

#ifdef NCCL_OS_LINUX
#define NCCL_INVALID_SOCKET -1
#elif defined(NCCL_OS_WINDOWS)
#define NCCL_INVALID_SOCKET INVALID_SOCKET
#endif

#ifdef NCCL_OS_LINUX
typedef int ncclSocketDescriptor;
#elif defined(NCCL_OS_WINDOWS)
typedef SOCKET ncclSocketDescriptor;
#endif

uint64_t ncclOsGetpid();
void ncclOsSleep(unsigned int time_msec);
ncclResult_t ncclOsSetCpuStackSize();

/* Socket functions */
bool ncclOsSocketIsValid(struct ncclSocket* sock);
bool ncclOsSocketDescriptorIsValid(ncclSocketDescriptor sock);
ncclResult_t ncclOsFindInterfaces(const char* prefixList, char* names, union ncclSocketAddress *addrs, int sock_family,
  int maxIfNameSize, int maxIfs, int* found);
void ncclOsPollSocket(ncclSocketDescriptor sock, int op);
ncclResult_t ncclOsSocketPollConnect(struct ncclSocket* sock);
ncclResult_t ncclOsSocketStartConnect(struct ncclSocket* sock);
ncclResult_t ncclOsSocketSetFlags(struct ncclSocket* sock);
ncclResult_t ncclOsSocketProgressOpt(int op, struct ncclSocket* sock, void* ptr, int size, int* offset, int block, int* closed);
ncclResult_t ncclOsSocketResetFd(struct ncclSocket* sock);
void ncclOsSocketResetAccept(struct ncclSocket* sock);
ncclResult_t ncclOsSocketTryAccept(struct ncclSocket* sock);

#endif /* NCCL_OS_H */
