/*************************************************************************
 * Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEBUG_H_
#define NCCL_DEBUG_H_

#include "core.h"
#include <stdio.h>

#include <unistd.h>
#include <sys/syscall.h>
#include <limits.h>
#define gettid() (pid_t) syscall(SYS_gettid)

typedef enum {NONE=0, VERSION=1, WARN=2, INFO=3, ABORT=4, TRACE=5} DebugLevel;
extern DebugLevel ncclDebugLevel;
extern pthread_mutex_t ncclDebugOutputLock;
extern FILE *ncclDebugFile;
extern void getHostName(char* hostname, int maxlen);

#define WARN(...) do {                                           \
  if (ncclDebugLevel >= WARN) {                                  \
    char hostname[1024];                                         \
    getHostName(hostname, 1024);                                 \
    int cudaDev;                                                 \
    cudaGetDevice(&cudaDev);                                     \
    pthread_mutex_lock(&ncclDebugOutputLock);                    \
    fprintf(ncclDebugFile,"\n%s:%d:%d [%d] %s:%d WARN ", hostname, getpid(), gettid(), cudaDev, __FILE__, __LINE__); \
    fprintf(ncclDebugFile,__VA_ARGS__);                          \
    fprintf(ncclDebugFile,"\n");                                 \
    fflush(ncclDebugFile);                                       \
    pthread_mutex_unlock(&ncclDebugOutputLock);                  \
    if (ncclDebugLevel == ABORT) abort();                        \
  }                                                              \
} while(0)

#define INFO(...) do {                                           \
  if (ncclDebugLevel >= INFO) {                                  \
    char hostname[1024];                                         \
    getHostName(hostname, 1024);                                 \
    int cudaDev;                                                 \
    cudaGetDevice(&cudaDev);                                     \
    pthread_mutex_lock(&ncclDebugOutputLock);                    \
    fprintf(ncclDebugFile,"%s:%d:%d [%d] INFO ", hostname, getpid(), gettid(), cudaDev); \
    fprintf(ncclDebugFile,__VA_ARGS__);fprintf(ncclDebugFile,"\n"); \
    fflush(ncclDebugFile);                                       \
    pthread_mutex_unlock(&ncclDebugOutputLock);                  \
  }                                                              \
} while(0)

#ifdef ENABLE_TRACE
#define TRACE(...) do {                                          \
if (ncclDebugLevel == TRACE) {                                   \
    char hostname[1024];                                         \
    getHostName(hostname, 1024);                                 \
    int cudaDev;                                                 \
    cudaGetDevice(&cudaDev);                                     \
    pthread_mutex_lock(&ncclDebugOutputLock);                    \
    fprintf(ncclDebugFile,"%s:%d:%d [%d] %s:%d TRACE ", hostname, getpid(), gettid(), cudaDev, __func__, __LINE__); \
    fprintf(ncclDebugFile,__VA_ARGS__);fprintf(ncclDebugFile,"\n"); \
    fflush(ncclDebugFile);                                       \
    pthread_mutex_unlock(&ncclDebugOutputLock);                  \
  }                                                              \
} while(0)
#else
#define TRACE(...)
#endif

extern int ncclPrintCRCs;
extern int ncclChecks;

static void initDebug() {
  const char* nccl_debug = getenv("NCCL_DEBUG");
  if (nccl_debug == NULL) {
    ncclDebugLevel = NONE;
  } else if (strcmp(nccl_debug, "VERSION") == 0) {
    ncclDebugLevel = VERSION;
  } else if (strcmp(nccl_debug, "WARN") == 0) {
    ncclDebugLevel = WARN;
  } else if (strcmp(nccl_debug, "INFO") == 0) {
    ncclDebugLevel = INFO;
  } else if (strcmp(nccl_debug, "ABORT") == 0) {
    ncclDebugLevel = ABORT;
  } else if (strcmp(nccl_debug, "TRACE") == 0) {
    ncclDebugLevel = TRACE;
  }

  // Parse and expand the NCCL_DEBUG_FILE path
  const char* nccl_debug_file = getenv("NCCL_DEBUG_FILE");
  if (nccl_debug_file != NULL) {
    int c = 0;
    char debug_fn[PATH_MAX+1] = "";
    char *dfn = debug_fn;
    while (nccl_debug_file[c] != '\0' && c < PATH_MAX) {
      if (nccl_debug_file[c++] != '%') {
        *dfn++ = nccl_debug_file[c-1];
        continue;
      }
      switch (nccl_debug_file[c++]) {
        case '%': // Double %
          *dfn++ = '%';
          break;
        case 'h': // %h = hostname
          char hostname[1024];
          getHostName(hostname, 1024);
          dfn += snprintf(dfn, PATH_MAX, "%s", hostname);
          break;
        case 'p': // %p = pid
          dfn += snprintf(dfn, PATH_MAX, "%d", getpid());
          break;
        default: // Echo everything we don't understand
          *dfn++ = '%';
          *dfn++ = nccl_debug_file[c-1];
          break;
      }
    }
    *dfn = '\0';
    if (debug_fn[0] != '\0') {
      FILE *file = fopen(debug_fn, "w");
      if (file != NULL) {
        INFO("DEBUG file is '%s'", debug_fn);
        ncclDebugFile = file;
      }
    }
  }

  const char* nccl_crc = getenv("NCCL_CRC");
  if (nccl_crc != NULL && strcmp(nccl_crc, "PRINT") == 0) {
    ncclPrintCRCs = 1;
  } else {
    ncclPrintCRCs = 0;
  }

  const char* nccl_checks_disable = getenv("NCCL_CHECKS_DISABLE");
  if (nccl_checks_disable && atoi(nccl_checks_disable) > 0) {
    ncclChecks = 0;
  } else {
    ncclChecks = 1;
  }

  pthread_mutex_init(&ncclDebugOutputLock, NULL);
}

#endif
