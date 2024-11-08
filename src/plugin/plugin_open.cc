/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <dlfcn.h>

#include "debug.h"

#define MAX_STR_LEN 255

static void* tryOpenLib(char* name, int* err, char* errStr) {
  *err = 0;
  if (nullptr == name || strlen(name) == 0) {
    return nullptr;
  }

  if (strncasecmp(name, "STATIC_PLUGIN", strlen(name)) == 0) {
    name = nullptr;
  }

  void *handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
  if (nullptr == handle) {
    strncpy(errStr, dlerror(), MAX_STR_LEN);
    errStr[MAX_STR_LEN] = '\0';
    // "handle" and "name" won't be NULL at the same time.
    // coverity[var_deref_model]
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
  INFO(NCCL_INIT|NCCL_NET, "Plugin: %s", openErrStr);
  return nameList;
}

static void* openNetPluginLib(char* couldNotFindNames, int len) {
  int openErr;
  void *pluginLib;
  char netPluginLibName[PATH_MAX];
  char openErrStr[MAX_STR_LEN + 1] = { 0 };
  const char *envNetPluginName = getenv("NCCL_NET_PLUGIN");
  if (envNetPluginName && strlen(envNetPluginName)) {
    snprintf(netPluginLibName, PATH_MAX, "%s", envNetPluginName);
    pluginLib = tryOpenLib(netPluginLibName, &openErr, openErrStr);
    if (pluginLib) {
      INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Plugin name set by env to %s", netPluginLibName);
      return pluginLib;
    }
    couldNotFindNames = tryOpenLibCheck(openErr, openErrStr, couldNotFindNames, &len, netPluginLibName);

    snprintf(netPluginLibName, PATH_MAX, "libnccl-net-%s.so", envNetPluginName);
    pluginLib = tryOpenLib(netPluginLibName, &openErr, openErrStr);
    if (pluginLib) {
      INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Plugin name set by env to %s", netPluginLibName);
      return pluginLib;
    }
    couldNotFindNames = tryOpenLibCheck(openErr, openErrStr, couldNotFindNames, &len, netPluginLibName);
  } else {
    snprintf(netPluginLibName, PATH_MAX, "libnccl-net.so");
    pluginLib = tryOpenLib(netPluginLibName, &openErr, openErrStr);
    if (pluginLib) {
      return pluginLib;
    }
    couldNotFindNames = tryOpenLibCheck(openErr, openErrStr, couldNotFindNames, &len, netPluginLibName);
  }
  return nullptr;
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

static void* openProfilerPluginLib(char* couldNotFindNames, int len) {
  int openErr;
  void *pluginLib;
  char profilerPluginLibName[PATH_MAX];
  char openErrStr[MAX_STR_LEN + 1] = { 0 };

  const char *envProfilerPluginName = getenv("NCCL_PROFILER_PLUGIN");
  if (envProfilerPluginName && strlen(envProfilerPluginName)) {
    snprintf(profilerPluginLibName, PATH_MAX, "%s", envProfilerPluginName);
    pluginLib = tryOpenLib(profilerPluginLibName, &openErr, openErrStr);
    if (pluginLib) {
      INFO(NCCL_INIT|NCCL_ENV, "PROFILER/Plugin: Plugin name set by env to %s", profilerPluginLibName);
      return pluginLib;
    }

    couldNotFindNames = tryOpenLibCheck(openErr, openErrStr, couldNotFindNames, &len, profilerPluginLibName);
    snprintf(profilerPluginLibName, PATH_MAX, "libnccl-profiler-%s.so", envProfilerPluginName);
    pluginLib = tryOpenLib(profilerPluginLibName, &openErr, openErrStr);
    if (pluginLib) {
      INFO(NCCL_INIT|NCCL_ENV, "PROFILER/Plugin: Plugin name set by env to %s", profilerPluginLibName);
      return pluginLib;
    }
    couldNotFindNames = tryOpenLibCheck(openErr, openErrStr, couldNotFindNames, &len, profilerPluginLibName);
  } else {
    snprintf(profilerPluginLibName, PATH_MAX, "libnccl-profiler.so");
    pluginLib = tryOpenLib(profilerPluginLibName, &openErr, openErrStr);
    if (pluginLib) {
      return pluginLib;
    }
    couldNotFindNames = tryOpenLibCheck(openErr, openErrStr, couldNotFindNames, &len, profilerPluginLibName);
  }

  return nullptr;
}
