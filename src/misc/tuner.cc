/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <dlfcn.h>
#include <errno.h>
#include <stdlib.h>

#include "checks.h"
#include "debug.h"
#include "nccl_tuner.h"

pthread_mutex_t tunerPluginLock = PTHREAD_MUTEX_INITIALIZER;
static int tunerPluginRefCount;
static void* tunerPluginLib = nullptr;
static ncclTuner_v3_t* tunerSymbol = nullptr;
static ncclTuner_v2_t* ncclTuner_v2 = nullptr;
static ncclTuner_v3_t ncclTuner_v2_as_v3;

static ncclResult_t ncclTuner_v2_as_v3_getCollInfo(void* context, ncclFunc_t collType, size_t nBytes, int collNetSupport, int nvlsSupport, int numPipeOps, float** collCostTable, int numAlgo __attribute__((unused)), int numProto __attribute__((unused)), int* nChannels) {
  int algorithm, protocol;
  NCCLCHECK(ncclTuner_v2->getCollInfo(context, collType, nBytes, collNetSupport, nvlsSupport, numPipeOps, &algorithm, &protocol, nChannels));
  // set time to 0 below to make sure this algorithm/protocol is selected later on
  collCostTable[algorithm][protocol] = 0.0;
  return ncclSuccess;
}

static ncclResult_t ncclTuner_v2_as_v3_init(size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction, void** context) {
  NCCLCHECK(ncclTuner_v2->init(nRanks, nNodes, logFunction, context));
  ncclTuner_v2_as_v3.name = ncclTuner_v2->name;
  ncclTuner_v2_as_v3.getCollInfo = ncclTuner_v2_as_v3_getCollInfo;
  ncclTuner_v2_as_v3.destroy = ncclTuner_v2->destroy;
  return ncclSuccess;
}

#define MAX_STR_LEN 255

static void* tryOpenLib(const char* name, int* err, char* errStr) {
  *err = 0;
  if (nullptr == name || strlen(name) == 0) {
    return nullptr;
  }

  if (strncasecmp(name, "STATIC_PLUGIN", strlen(name)) == 0) {
    name = nullptr;
  }

  void *handle = dlopen(name, RTLD_LAZY | RTLD_LOCAL);
  if (nullptr == handle) {
    strncpy(errStr, dlerror(), MAX_STR_LEN);
    errStr[MAX_STR_LEN] = '\0';
    if (strstr(errStr, name) && strstr(errStr, "No such file or directory")) {
      *err = ENOENT;
    }
  }
  return handle;
}

static char* tryOpenLibCheck(int openErr, char* openErrStr, char* nameList, int *nameListLen, char* name) {
  if (openErr == ENOENT) {
    snprintf(nameList, *nameListLen, " %s", name);
    nameList += strlen(name) + 1;
    *nameListLen -= strlen(name) + 1;
    return nameList;
  }
  INFO(NCCL_ENV|NCCL_TUNING, "TUNER/Plugin: %s", openErrStr);
  return nameList;
}

static void* openTunerPluginLib(char* couldNotFindNames, int len) {
  int openErr;
  void *pluginLib;
  char tunerPluginLibName[PATH_MAX];
  char openErrStr[MAX_STR_LEN + 1] = { 0 };
  const char *envTunerPluginName = getenv("NCCL_TUNER_PLUGIN");
  if (envTunerPluginName && strlen(envTunerPluginName)) {
    INFO(NCCL_ENV|NCCL_TUNING, "TUNER/Plugin: NCCL_TUNER_PLUGIN set to %s", envTunerPluginName);
    snprintf(tunerPluginLibName, PATH_MAX, "%s", envTunerPluginName);
    pluginLib = tryOpenLib(tunerPluginLibName, &openErr, openErrStr);
    if (pluginLib) {
      INFO(NCCL_ENV|NCCL_TUNING, "TUNER/Plugin: Plugin name set by env to %s", tunerPluginLibName);
      return pluginLib;
    }
    couldNotFindNames = tryOpenLibCheck(openErr, openErrStr, couldNotFindNames, &len, tunerPluginLibName);

    snprintf(tunerPluginLibName, PATH_MAX, "libnccl-tuner-%s.so", envTunerPluginName);
    pluginLib = tryOpenLib(tunerPluginLibName, &openErr, openErrStr);
    if (pluginLib) {
      INFO(NCCL_ENV|NCCL_TUNING, "TUNER/Plugin: Plugin name set by env to %s", tunerPluginLibName);
      return pluginLib;
    }
    couldNotFindNames = tryOpenLibCheck(openErr, openErrStr, couldNotFindNames, &len, tunerPluginLibName);
  } else {
    snprintf(tunerPluginLibName, PATH_MAX, "libnccl-tuner.so");
    pluginLib = tryOpenLib(tunerPluginLibName, &openErr, openErrStr);
    if (pluginLib) {
      return pluginLib;
    }
    couldNotFindNames = tryOpenLibCheck(openErr, openErrStr, couldNotFindNames, &len, tunerPluginLibName);
  }

  const char *envNetPluginName = getenv("NCCL_NET_PLUGIN");
  if (envNetPluginName && strlen(envNetPluginName)) {
    // Users are allowed to pack tuner into the net plugin
    snprintf(tunerPluginLibName, PATH_MAX, "%s", envNetPluginName);
    pluginLib = tryOpenLib(tunerPluginLibName, &openErr, openErrStr);
    if (pluginLib) {
      INFO(NCCL_ENV|NCCL_TUNING, "TUNER/Plugin: Plugin name set by env to %s", tunerPluginLibName);
      return pluginLib;
    }
    couldNotFindNames = tryOpenLibCheck(openErr, openErrStr, couldNotFindNames, &len, tunerPluginLibName);

    snprintf(tunerPluginLibName, PATH_MAX, "libnccl-net-%s.so", envNetPluginName);
    pluginLib = tryOpenLib(tunerPluginLibName, &openErr, openErrStr);
    if (pluginLib) {
      INFO(NCCL_ENV|NCCL_TUNING, "TUNER/Plugin: Plugin name set by env to %s", tunerPluginLibName);
      return pluginLib;
    }
    couldNotFindNames = tryOpenLibCheck(openErr, openErrStr, couldNotFindNames, &len, tunerPluginLibName);
  } else {
    snprintf(tunerPluginLibName, PATH_MAX, "libnccl-net.so");
    pluginLib = tryOpenLib(tunerPluginLibName, &openErr, openErrStr);
    if (pluginLib) {
      return pluginLib;
    }
    couldNotFindNames = tryOpenLibCheck(openErr, openErrStr, couldNotFindNames, &len, tunerPluginLibName);
  }
  tunerPluginLibName[0] = '\0';
  return nullptr;
}

enum {
  tunerPluginLoadFailed  = -1,
  tunerPluginLoadReady   =  0,
  tunerPluginLoadSuccess =  1,
};

#define MAX_PLUGIN_LOAD 4

ncclResult_t ncclTunerPluginLoad(ncclTuner_t** tuner) {
  // Initialize to nullptr by default if plugin tuner cannot be loaded.
  char couldNotFindNames[MAX_PLUGIN_LOAD * PATH_MAX] = { 0 };
  *tuner = nullptr;
  static int status = tunerPluginLoadReady;
  if (tunerPluginLoadFailed == status) {
    return ncclSuccess;
  }

  pthread_mutex_lock(&tunerPluginLock);
  if (tunerPluginLoadFailed == status) {
    goto exit;
  }

  if (tunerPluginLoadSuccess == status) {
    *tuner = tunerSymbol;
    ++tunerPluginRefCount;
    goto exit;
  }

  tunerPluginLib = openTunerPluginLib(couldNotFindNames, MAX_PLUGIN_LOAD * PATH_MAX);
  if (nullptr == tunerPluginLib) {
    if (strlen(couldNotFindNames)) {
      INFO(NCCL_ENV|NCCL_TUNING, "TUNER/Plugin: Could not find:%s. Using internal tuner plugin.", couldNotFindNames);
    } else {
      INFO(NCCL_ENV|NCCL_TUNING, "TUNER/Plugin: Using internal tuner plugin.");
    }
    goto fail;
  }

  tunerSymbol = (ncclTuner_v3_t*)dlsym(tunerPluginLib, "ncclTunerPlugin_v3");
  if (tunerSymbol == nullptr) {
    INFO(NCCL_ENV|NCCL_TUNING, "TUNER/Plugin: Failed to find ncclTunerPlugin_v3 symbol.");
    ncclTuner_v2 = (ncclTuner_v2_t*)dlsym(tunerPluginLib, "ncclTunerPlugin_v2");
    if (ncclTuner_v2 == nullptr) {
      INFO(NCCL_ENV|NCCL_TUNING, "TUNER/Plugin: Failed to find ncclTunerPlugin_v2 symbol, using internal tuner instead.");
      dlclose(tunerPluginLib);
      goto fail;
    } else {
      ncclTuner_v2_as_v3.init = ncclTuner_v2_as_v3_init;
      ncclTuner_v2_as_v3.name = ncclTuner_v2->name;
      tunerSymbol = &ncclTuner_v2_as_v3;
    }
  }

  INFO(NCCL_ENV|NCCL_TUNING, "TUNER/Plugin: Using tuner plugin %s", tunerSymbol->name);
  *tuner = tunerSymbol;
  ++tunerPluginRefCount;
  status = tunerPluginLoadSuccess;

exit:
  pthread_mutex_unlock(&tunerPluginLock);
  return ncclSuccess;
fail:
  tunerPluginLib = nullptr;
  status = tunerPluginLoadFailed;
  goto exit;
}

ncclResult_t ncclTunerPluginUnload(ncclTuner_t** tuner) {
  if (*tuner == nullptr) return ncclSuccess;
  pthread_mutex_lock(&tunerPluginLock);
  if (0 == (--tunerPluginRefCount)) {
    INFO(NCCL_TUNING, "TUNER/Plugin: Closing tuner: '%s'", tunerSymbol->name);
    dlclose(tunerPluginLib);
    tunerPluginLib = nullptr;
    tunerSymbol = nullptr;
    *tuner = nullptr;
  }
  pthread_mutex_unlock(&tunerPluginLock);
  return ncclSuccess;
}
