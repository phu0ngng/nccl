#include <stdio.h>
#include <dlfcn.h>
#include <sys/types.h>
#include <limits.h>
#include <stdlib.h>
#include <errno.h>
#include "plugin.h"
#include "nccl_net.h"
#include "nccl_tuner.h"
#include "nccl_profiler.h"

#define NCCL_NET_PLUGIN_SYM "ncclNetPlugin_v11"
#define NCCL_TUNER_PLUGIN_SYM "ncclTunerPlugin_v5"
#define NCCL_PROFILER_PLUGIN_SYM "ncclProfiler_v5"

enum test {
  ncclPluginNetRelPathTest,
  ncclPluginTunerRelPathTest,
  ncclPluginProfilerRelPathTest,
  ncclPluginNetAbsPathTest,
  ncclPluginTunerAbsPathTest,
  ncclPluginProfilerAbsPathTest,
  ncclPluginNetSuffixTest,
  ncclPluginTunerSuffixTest,
  ncclPluginProfilerSuffixTest,
  ncclPluginNetStaticTest,
};

const char* libName[] = {
  "libnccl-net-example.so",
  "libnccl-tuner-example.so",
  "libnccl-profiler-example.so",
  "libnccl-net-example.so",
  "libnccl-tuner-example.so",
  "libnccl-profiler-example.so",
  "example",
  "example",
  "example",
  "STATIC_PLUGIN",
};

const char* testName[] = {
  "test_network_plugin_relpath",
  "test_tuner_plugin_relpath",
  "test_profiler_plugin_relpath",
  "test_network_plugin_abspath",
  "test_tuner_plugin_abspath",
  "test_profiler_plugin_abspath",
  "test_network_plugin_suffix",
  "test_tuner_plugin_suffix",
  "test_profiler_plugin_suffix",
  "test_network_plugin_static",
};

static int test_plugin_load(enum test type) {
  char name[PATH_MAX] = {};
  if (type < ncclPluginNetAbsPathTest) {
    sprintf(name, "%s", libName[type]);
  } else if (type < ncclPluginNetSuffixTest) {
    const char* dirname = getenv("NCCL_HOME");
    if (dirname) {
      sprintf(name, "%s/test/unit/plugins/%s", dirname, libName[type]);
    } else {
      fprintf(stderr, "Error: NCCL_HOME not set\n");
      return 1;
    }
  } else { // suffix and static plugin
    sprintf(name, "%s", libName[type]);
  }

  switch (type) {
    case ncclPluginNetRelPathTest:
    case ncclPluginNetAbsPathTest:
    case ncclPluginNetSuffixTest:
      {
        void* handle = ncclOpenNetPluginLib(name);
        if (handle) {
          ncclNet_t* sym = (ncclNet_t*)dlsym(handle, NCCL_NET_PLUGIN_SYM);
          if (sym) {
            if (strncmp(sym->name, "Plugin", strlen("Plugin")) == 0) {
              ncclClosePluginLib(handle, ncclPluginTypeNet);
              fprintf(stdout, "%s: SUCCESS\n", testName[type]);
              return 0;
            }
            ncclClosePluginLib(handle, ncclPluginTypeNet);
            fprintf(stderr, "%s: plugin name not found (path: %s)\n", testName[type], name);
            return 1;
          }
          ncclClosePluginLib(handle, ncclPluginTypeNet);
          fprintf(stderr, "%s: %s (path: %s)\n", testName[type], strerror(errno), name);
          return 1;
        }
        fprintf(stderr, "%s: %s (path: %s)\n", testName[type], strerror(errno), name);
        return 1;
      }
      break;
    case ncclPluginNetStaticTest:
      {
        void* handle = ncclOpenNetPluginLib(name);
        if (handle) {
          return 0;
        }
        fprintf(stderr, "%s: %s (path: %s)\n", testName[type], strerror(errno), name);
        return 1;
      }
      break;
    case ncclPluginTunerAbsPathTest:
    case ncclPluginTunerRelPathTest:
    case ncclPluginTunerSuffixTest:
      {
        void* handle = ncclOpenTunerPluginLib(name);
        if (handle) {
          ncclTuner_t* sym = (ncclTuner_t*)dlsym(handle, NCCL_TUNER_PLUGIN_SYM);
          if (sym) {
            if (strncmp(sym->name, "Example", strlen("Example")) == 0) {
              ncclClosePluginLib(handle, ncclPluginTypeTuner);
              fprintf(stdout, "%s: SUCCESS\n", testName[type]);
              return 0;
            }
            ncclClosePluginLib(handle, ncclPluginTypeTuner);
            fprintf(stderr, "%s: plugin name not found (path: %s)\n", testName[type], name);
            return 1;
          }
          ncclClosePluginLib(handle, ncclPluginTypeTuner);
          fprintf(stderr, "%s: %s (path: %s)\n", testName[type], strerror(errno), name);
          return 1;
        }
        fprintf(stderr, "%s: %s (path: %s)\n", testName[type], strerror(errno), name);
        return 1;
      }
      break;
    case ncclPluginProfilerRelPathTest:
    case ncclPluginProfilerAbsPathTest:
    case ncclPluginProfilerSuffixTest:
      {
        void* handle = ncclOpenProfilerPluginLib(name);
        if (handle) {
          ncclProfiler_t* sym = (ncclProfiler_t*)dlsym(handle, NCCL_PROFILER_PLUGIN_SYM);
          if (sym) {
            if (strncmp(sym->name, "Example", strlen("Example")) == 0) {
              ncclClosePluginLib(handle, ncclPluginTypeProfiler);
              fprintf(stdout, "%s: SUCCESS\n", testName[type]);
              return 0;
            }
            ncclClosePluginLib(handle, ncclPluginTypeProfiler);
            fprintf(stderr, "%s: plugin name not found (path: %s)\n", testName[type], name);
            return 1;
          }
          ncclClosePluginLib(handle, ncclPluginTypeProfiler);
          fprintf(stderr, "%s: %s (path: %s)\n", testName[type], strerror(errno), name);
          return 1;
        }
        fprintf(stderr, "%s: %s (path: %s)\n", testName[type], strerror(errno), name);
        return 1;
      }
      break;
    default:;
  }
  return 1;
}

int main(void) {
  int errors = 0;
  if (test_plugin_load(ncclPluginNetRelPathTest)) errors++;
  if (test_plugin_load(ncclPluginNetAbsPathTest)) errors++;
  if (test_plugin_load(ncclPluginNetSuffixTest)) errors++;
  if (test_plugin_load(ncclPluginNetStaticTest)) errors++; // comment for now as not loading local symbols
  if (test_plugin_load(ncclPluginTunerRelPathTest)) errors++;
  if (test_plugin_load(ncclPluginTunerAbsPathTest)) errors++;
  if (test_plugin_load(ncclPluginTunerSuffixTest)) errors++;
  if (test_plugin_load(ncclPluginProfilerRelPathTest)) errors++;
  if (test_plugin_load(ncclPluginProfilerAbsPathTest)) errors++;
  if (test_plugin_load(ncclPluginProfilerSuffixTest)) errors++;
  if (errors) {
    fprintf(stderr, "%d tests failed!\n", errors);
    return 1;
  }
  fprintf(stdout, "All tests successfull!\n");
  return 0;
}
