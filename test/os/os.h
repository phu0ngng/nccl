/*************************************************************************
 * Copyright (c) 2016-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef __NCCL_TEST_OS_H__
#define __NCCL_TEST_OS_H__

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdarg.h>
#include "nccl.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Host identification and hashing
 * Get a unique identifier for the host that combines hostname and
 * system-specific unique ID (boot_id on Linux, MachineGuid on Windows)
 */
uint64_t ncclTestGetHostHash(const char* hostname);

/* Process operations
 * Platform-specific process utilities
 */
typedef enum {
  NCCL_TEST_OS_SUCCESS = 0,
  NCCL_TEST_OS_ERROR = 1,
  NCCL_TEST_OS_NOT_SUPPORTED = 2
} ncclTestOsResult_t;

/* Setup signal handling for child processes (Linux only)
 * Returns NCCL_TEST_OS_SUCCESS on Linux, NCCL_TEST_OS_NOT_SUPPORTED on Windows
 */
ncclTestOsResult_t ncclTestOsSetupSignalHandler();

/* Check if fork/exec operations are supported on this platform
 * Returns 1 if supported, 0 otherwise
 */
int ncclTestOsSupportsProcess();

/* POSIX compatibility functions
 * Cross-platform wrappers for common system functions
 */

/* Get hostname - equivalent to POSIX gethostname() */
int ncclTestGetHostname(char* name, size_t len);

/* Get process ID - equivalent to POSIX getpid() */
int ncclTestGetPid();

/* Case-insensitive string comparison - equivalent to POSIX strcasecmp() */
int ncclTestStrcasecmp(const char* s1, const char* s2);

/* Case-insensitive string comparison with length - equivalent to POSIX strncasecmp() */
int ncclTestStrncasecmp(const char* s1, const char* s2, size_t n);

/* Allocate and format string - equivalent to POSIX asprintf() */
int ncclTestAsprintf(char** strp, const char* fmt, ...);

/* Set environment variable - equivalent to POSIX setenv() */
int ncclTestSetenv(const char* name, const char* value, int overwrite);

/* Set line buffering on stream - equivalent to POSIX setlinebuf() */
void ncclTestSetlinebuf(FILE* stream);

#ifdef __cplusplus
}
#endif

struct ncclComm;
typedef struct ncclComm* ncclComm_t;

/* Command-line option parsing - getopt_long support
 * On Linux: just use system getopt.h
 * On Windows: provide custom implementation
 */
#if defined(NCCL_OS_WINDOWS)

#include <windows.h>
#include <process.h>

#ifdef __cplusplus
extern "C" {
#endif

/* option structure for long options */
struct option {
    const char *name;
    int has_arg;
    int *flag;
    int val;
};

/* Constants for has_arg field */
#define no_argument         0
#define required_argument   1
#define optional_argument   2

/* getopt global variables */
extern char *optarg;
extern int optind, opterr, optopt;

/* Parse command-line options - equivalent to POSIX getopt_long() */
int getopt_long(int argc, char * const argv[], const char *optstring,
                const struct option *longopts, int *longindex);


#ifdef __cplusplus
}
#endif

extern "C" char const* ncclGetLastError(ncclComm_t comm);

#define NCCL_WEAK

#elif defined(NCCL_OS_LINUX)
#include <dlfcn.h>
#include <getopt.h>
#include <pthread.h>
#include <unistd.h>
#include <libgen.h>
#include <signal.h>
#include <sched.h>

#define NCCL_STRINGIFY_(x) #x

extern "C" __attribute__((weak)) char const* ncclGetLastError(ncclComm_t comm) {
  return "";
}

#define NCCL_WEAK __attribute__((weak))

#endif

#endif // __NCCL_TEST_OS_H__
