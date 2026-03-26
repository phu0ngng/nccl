#include <stdio.h>
#include <dlfcn.h>
#include <sys/types.h>
#include <limits.h>
#include <stdlib.h>
#include <errno.h>
#include <dirent.h>
#include <sys/stat.h>
#include <string.h>
#include <unistd.h>
#include "plugin.h"
#include "nccl_net.h"
#include "nccl_tuner.h"
#include "nccl_profiler.h"
#include "nccl_env.h"

#define NCCL_NET_PLUGIN_SYM "ncclNetPlugin_v11"
#define NCCL_TUNER_PLUGIN_SYM "ncclTunerPlugin_v5"
#define NCCL_PROFILER_PLUGIN_SYM "ncclProfiler_v6"
#define NCCL_ENV_PLUGIN_SYM "ncclEnvPlugin_v2"

enum test {
  ncclPluginNetRelPathTest,
  ncclPluginTunerRelPathTest,
  ncclPluginProfilerRelPathTest,
  ncclPluginEnvRelPathTest,
  ncclPluginNetAbsPathTest,
  ncclPluginTunerAbsPathTest,
  ncclPluginProfilerAbsPathTest,
  ncclPluginEnvAbsPathTest,
  ncclPluginNetSuffixTest,
  ncclPluginTunerSuffixTest,
  ncclPluginProfilerSuffixTest,
  ncclPluginEnvSuffixTest,
  ncclPluginNetStaticTest,
};

const char* libName[] = {
  "libnccl-net-example.so",
  "libnccl-tuner-example.so",
  "libnccl-profiler-example.so",
  "libnccl-env-example.so",
  "libnccl-net-example.so",
  "libnccl-tuner-example.so",
  "libnccl-profiler-example.so",
  "libnccl-env-example.so",
  "example",
  "example",
  "example",
  "example",
  "STATIC_PLUGIN",
};

const char* testName[] = {
  "test_network_plugin_relpath",
  "test_tuner_plugin_relpath",
  "test_profiler_plugin_relpath",
  "test_env_plugin_relpath",
  "test_network_plugin_abspath",
  "test_tuner_plugin_abspath",
  "test_profiler_plugin_abspath",
  "test_env_plugin_abspath",
  "test_network_plugin_suffix",
  "test_tuner_plugin_suffix",
  "test_profiler_plugin_suffix",
  "test_env_plugin_suffix",
  "test_network_plugin_static",
};

static void list_directory_contents(const char* path) {
  fprintf(stdout, "\n=== Listing directory: %s ===\n", path);

  DIR* dir = opendir(path);
  if (dir == NULL) {
    fprintf(stderr, "ERROR: Cannot open directory '%s': %s\n", path, strerror(errno));
    return;
  }

  struct dirent* entry;
  struct stat file_stat;
  char full_path[PATH_MAX];
  int file_count = 0;

  while ((entry = readdir(dir)) != NULL) {
    // Skip . and ..
    if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
      continue;
    }

    snprintf(full_path, sizeof(full_path), "%s/%s", path, entry->d_name);

    if (stat(full_path, &file_stat) == 0) {
      char perms[11];
      snprintf(perms, sizeof(perms), "%c%c%c%c%c%c%c%c%c%c",
        S_ISDIR(file_stat.st_mode) ? 'd' : '-',
        (file_stat.st_mode & S_IRUSR) ? 'r' : '-',
        (file_stat.st_mode & S_IWUSR) ? 'w' : '-',
        (file_stat.st_mode & S_IXUSR) ? 'x' : '-',
        (file_stat.st_mode & S_IRGRP) ? 'r' : '-',
        (file_stat.st_mode & S_IWGRP) ? 'w' : '-',
        (file_stat.st_mode & S_IXGRP) ? 'x' : '-',
        (file_stat.st_mode & S_IROTH) ? 'r' : '-',
        (file_stat.st_mode & S_IWOTH) ? 'w' : '-',
        (file_stat.st_mode & S_IXOTH) ? 'x' : '-');

      fprintf(stdout, "  %s  %8ld  %s\n", perms, (long)file_stat.st_size, entry->d_name);
      file_count++;
    } else {
      fprintf(stdout, "  ??????????  %8s  %s (stat failed: %s)\n", "?", entry->d_name, strerror(errno));
      file_count++;
    }
  }

  closedir(dir);
  fprintf(stdout, "Total files: %d\n", file_count);
  fprintf(stdout, "=== End of directory listing ===\n\n");
}

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
        // Clear any previous dlerror
        dlerror();
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
          const char* err = dlerror();
          ncclClosePluginLib(handle, ncclPluginTypeProfiler);
          fprintf(stderr, "%s: dlsym failed: %s (path: %s)\n", testName[type], err ? err : "unknown error", name);
          return 1;
        }
        // Get detailed error from dlopen
        const char* err = dlerror();
        fprintf(stderr, "%s: dlopen failed: %s (path: %s)\n", testName[type], err ? err : strerror(errno), name);
        return 1;
      }
      break;
    case ncclPluginEnvRelPathTest:
    case ncclPluginEnvAbsPathTest:
    case ncclPluginEnvSuffixTest:
      {
        void* handle = ncclOpenEnvPluginLib(name);
        if (handle) {
          ncclEnv_t* sym = (ncclEnv_t*)dlsym(handle, NCCL_ENV_PLUGIN_SYM);
          if (sym) {
            if (strncmp(sym->name, "ncclEnvExample", strlen("ncclEnvExample")) == 0) {
              ncclClosePluginLib(handle, ncclPluginTypeEnv);
              fprintf(stdout, "%s: SUCCESS\n", testName[type]);
              return 0;
            }
            ncclClosePluginLib(handle, ncclPluginTypeEnv);
            fprintf(stderr, "%s: plugin name not found (path: %s)\n", testName[type], name);
            return 1;
          }
          ncclClosePluginLib(handle, ncclPluginTypeEnv);
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
  if (test_plugin_load(ncclPluginEnvRelPathTest)) errors++;
  if (test_plugin_load(ncclPluginEnvAbsPathTest)) errors++;
  if (test_plugin_load(ncclPluginEnvSuffixTest)) errors++;
  if (errors) {
    fprintf(stderr, "%d tests failed!\n", errors);

    // Print diagnostic information to help debug the failure
    fprintf(stderr, "\n=== DIAGNOSTIC INFORMATION ===\n");

    // Current working directory
    char cwd[PATH_MAX];
    fprintf(stderr, "Current working directory: ");
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
      fprintf(stderr, "%s\n", cwd);
    } else {
      fprintf(stderr, "Unable to get current directory\n");
    }

    // Check LD_LIBRARY_PATH
    const char* ld_path = getenv("LD_LIBRARY_PATH");
    fprintf(stderr, "LD_LIBRARY_PATH: %s\n", ld_path ? ld_path : "(not set)");

    // Check NCCL_HOME
    const char* nccl_home = getenv("NCCL_HOME");
    fprintf(stderr, "NCCL_HOME: %s\n\n", nccl_home ? nccl_home : "(not set)");

    // List the plugins directory where we expect to find the .so files
    if (nccl_home) {
      char plugins_dir[PATH_MAX];
      snprintf(plugins_dir, sizeof(plugins_dir), "%s/test/unit/plugins", nccl_home);
      list_directory_contents(plugins_dir);
    } else {
      fprintf(stderr, "WARNING: NCCL_HOME not set, cannot list plugins directory\n");
    }

    // Also check if LD_LIBRARY_PATH points to a directory
    if (ld_path && strlen(ld_path) > 0) {
      // Parse the first directory in LD_LIBRARY_PATH
      char first_dir[PATH_MAX];
      const char* colon = strchr(ld_path, ':');
      if (colon) {
        size_t len = colon - ld_path;
        if (len < PATH_MAX) {
          strncpy(first_dir, ld_path, len);
          first_dir[len] = '\0';
          list_directory_contents(first_dir);
        }
      } else {
        list_directory_contents(ld_path);
      }
    }

    fprintf(stderr, "=== END DIAGNOSTIC INFORMATION ===\n");
    return 1;
  }
  fprintf(stdout, "All tests successfull!\n");
  return 0;
}
