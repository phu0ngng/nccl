/*************************************************************************
 * Copyright (c) 2016-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "os.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <strings.h>

#define HOSTID_FILE "/proc/sys/kernel/random/boot_id"

/* DJB2a hash function */
static uint64_t getHash(const char* string, size_t n) {
  uint64_t result = 5381;
  for (size_t c = 0; c < n; c++) {
    result = ((result << 5) + result) ^ string[c];
  }
  return result;
}

/* Generate a hash of the unique identifying string for this host
 * that will be unique for both bare-metal and container instances
 * Equivalent of a hash of: $(hostname)$(cat /proc/sys/kernel/random/boot_id)
 */
uint64_t ncclTestGetHostHash(const char* hostname) {
  char hostHash[1024];

  (void) strncpy(hostHash, hostname, sizeof(hostHash));
  int offset = strlen(hostHash);

  FILE *file = fopen(HOSTID_FILE, "r");
  if (file != NULL) {
    char *p;
    if (fscanf(file, "%ms", &p) == 1) {
        strncpy(hostHash+offset, p, sizeof(hostHash)-offset-1);
        free(p);
    }
    fclose(file);
  }

  hostHash[sizeof(hostHash)-1]='\0';

  return getHash(hostHash, strlen(hostHash));
}

ncclTestOsResult_t ncclTestOsSetupSignalHandler() {
  if (signal(SIGCHLD, SIG_IGN) == SIG_ERR) {
    printf("Failed to set up automatic cleanup of zombie processes\n");
    return NCCL_TEST_OS_ERROR;
  }
  return NCCL_TEST_OS_SUCCESS;
}

int ncclTestOsSupportsProcess() {
  return 1;
}

int ncclTestGetHostname(char* name, size_t len) {
  return gethostname(name, len);
}

int ncclTestGetPid() {
  return (int)getpid();
}

int ncclTestStrcasecmp(const char* s1, const char* s2) {
  return strcasecmp(s1, s2);
}

int ncclTestStrncasecmp(const char* s1, const char* s2, size_t n) {
  return strncasecmp(s1, s2, n);
}

int ncclTestAsprintf(char** strp, const char* fmt, ...) {
  va_list args;
  va_start(args, fmt);
  int result = vasprintf(strp, fmt, args);
  va_end(args);
  return result;
}

int ncclTestSetenv(const char* name, const char* value, int overwrite) {
  return setenv(name, value, overwrite);
}

void ncclTestSetlinebuf(FILE* stream) {
  setlinebuf(stream);
}
