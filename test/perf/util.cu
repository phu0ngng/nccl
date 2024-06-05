// This contains an utlities to handle output both to stdout and to
// json files.
//
// An ad-hoc, libc-based approach to writing json has been adopted to
// keep things simple and to avoid injecting a dependency on the
// library for an external JSON utility.
//
// However, this means that the code is a brittle to changes and care
// should be taken when adding/removing things. We also essentially
// give up when passed non-ASCII strings and non-printable characters
// except some of the usual ones.

#include "util.h"
#include <assert.h>
#include <errno.h>

#pragma GCC diagnostic ignored "-Wfree-nonheap-object"
#define PRINT if (is_main_thread) printf

extern int nThreads;
extern int nGpus;
extern size_t minBytes;
extern size_t maxBytes;
extern size_t stepBytes;
extern size_t stepFactor;
extern int datacheck;
extern int warmup_iters;
extern int iters;
extern int agg_iters;
extern int parallel_init;
extern int blocking_coll;
extern int side_comp;
extern int cudaGraphLaunches;


extern FILE *json_report_fp;
extern bool write_json;

typedef enum {
  JSON_NONE, // A pseudo-state meaning that the document is empty
  JSON_KEY,
  JSON_OBJECT_EMPTY,
  JSON_OBJECT_SOME,
  JSON_LIST_EMPTY,
  JSON_LIST_SOME,
} json_state_t;

// We use these statics to mantain a stack of states where we are writing.
// the init_json_output function gets this set up, and it's the finalize_json_output function's job to clean this up.
json_state_t *states = nullptr;
size_t state_cap = 0; // Allocated stack capacity
size_t state_n = 0;   // # of items in the stack.

// This tries to sanitize/quote a string from 'in' into 'out',
// assuming 'out' has length 'lim'.  We mainly quote ",/,\,\t,\n, and
// bail if we encounter non-printable stuff or non-ASCII stuff.
// 'in' should be null-terminated, of course.
//
// We return false if we were not able to copy all of 'in', either for
// length reasons or for unhandled characters.
static bool sanitize_json(char out[], int lim, const char *in) {
  int c = 0;
  while(*in) {
    if(c+1 >= lim) {
      out[c] = 0;
      return false;
    }
    switch(*in) {
    case '"':
    case '\\':
    case '/':
    case '\t':
    case '\n':
      if(c + 2 > lim) {
        out[c] = 0;
        return false;
      }

      out[c++] = '\\';
      if(*in == '\n') {
        out[c++] = 'n';
      }
      else if( *in == '\t') {
        out[c++] = 't';
      }
      else {
        out[c++] = *in;
      }
      break;
    default:
      if (*in >= 0x7F || *in <= 0x1F) {
        out[c] = 0;
        return false;
      }
      out[c++] = *in;
      break;
    }
    ++in;
  }
  out[c] = 0;
  return true;
}

// Push state onto the state stack. Reallocate for extra storage if needed.
// Because JSON_NONE is a pseudo-state, don't allow it to be pushed.
static void json_push_state(json_state_t state) {
  assert(state != JSON_NONE);
  if(state_cap <= (state_n+1)) {
    state_cap = max((size_t)16, state_cap*2);
    states = (json_state_t *)realloc(&states, sizeof(json_state_t)*state_cap);
    assert(states);
  }
  states[state_n++] = state;
}

// Return the current state at the top of the stack
static json_state_t json_curr_state() {
  if(state_n == 0) {
    return JSON_NONE;
  }
  return states[state_n-1];
}

// Replace the stack with state (equivalent to a pop & push if stack is not empty)
static void json_replace_state(json_state_t state) {
  assert(state != JSON_NONE);
  assert(state_n != 0);
  states[state_n-1] = state;
}

// Pop the top state off the stack, or return that the state is empty
static json_state_t json_pop_state() {
  if(state_n == 0) {
    return JSON_NONE;
  }
  return states[--state_n];
}

// Emit a key and separator. Santize the key.
// This is only acceptable if the top state is an object
// Emit a ',' separator of we aren't the first item.
static void json_key(const char *name) {
  switch(json_curr_state()) {
  case JSON_OBJECT_EMPTY:
    json_replace_state(JSON_OBJECT_SOME);
    break;
  case JSON_OBJECT_SOME:
    fprintf(json_report_fp, ",");
    break;
  default:
    assert(0);
    break;
  }
  char tmp[2048];
  sanitize_json(tmp, sizeof(tmp), name);
  fprintf(json_report_fp, "\"%s\":", tmp);
  json_push_state(JSON_KEY);
}

// Helper function for inserting values.
// Only acceptable after keys, top-level, or in lists.
// Emit preceeding ',' if in a list and not first item.
static void json_val_helper() {
  switch(json_curr_state()) {
  case JSON_LIST_EMPTY:
    json_replace_state(JSON_LIST_SOME);
    break;
  case JSON_LIST_SOME:
    fprintf(json_report_fp, ",");
    break;
  case JSON_KEY:
    json_pop_state();
    break;
  case JSON_NONE:
    break;
  default:
    assert(0);
  }
}

// Start an object
static void json_start_object() {
  json_val_helper();
  fprintf(json_report_fp, "{");
  json_push_state(JSON_OBJECT_EMPTY);
}

// Close an object
static void json_finish_object() {
  switch(json_pop_state()) {
  case JSON_OBJECT_EMPTY:
  case JSON_OBJECT_SOME:
    break;
  default:
    assert(0);
  }
  fprintf(json_report_fp, "}");
}

// Start a list
static void json_start_list() {
  json_val_helper();
  fprintf(json_report_fp, "[");
  json_push_state(JSON_LIST_EMPTY);
}

// Close a list
static void json_finish_list() {
  switch(json_pop_state()) {
  case JSON_LIST_EMPTY:
  case JSON_LIST_SOME:
    break;
  default:
    assert(0);
  }
  fprintf(json_report_fp, "]");
}

// Write a null value
static void json_null() {
  json_val_helper();
  fprintf(json_report_fp, "null");
}

// Write a (sanititzed) string
static void json_str(const char *str) {
  if(str == nullptr) {
    json_null();
    return;
  }
  json_val_helper();
  char tmp[2048];
  sanitize_json(tmp, sizeof(tmp), str);
  fprintf(json_report_fp, "\"%s\"", tmp);
}

// Write a bool as "true" or "false" strings.
static void json_bool(bool val) {
  json_str(val ? "true" : "false");
}

// Write an integer value
static void json_int(const int val) {
  json_val_helper();
  fprintf(json_report_fp, "%d", val);
}

// Write a size_t value
static void json_size_t(const size_t val) {
  json_val_helper();
  fprintf(json_report_fp, "%zu", val);
}

// Write a double value
static void json_double(const double val) {
  json_val_helper();
  if(val != val) {
    fprintf(json_report_fp, "\"nan\"");
  }
  else {
    fprintf(json_report_fp, "%lf", val);
  }
}

// Fill buff with a formatted time string corresponding to 'now.
// Write len or fewer bytes.
void format_now(char *buff, int len) {
  time_t now;
  time(&now);
  struct tm *timeinfo = localtime(&now);

  strftime(buff, len, "%Y-%m-%d %H:%M:%S", timeinfo);
}

// Try to set up JSON file output.
// If 'in_path' is NULL, we stop.
// Otherwise, We borrow 'in_path' and try to open it as a new file.
// If it already exists, we probe for new files by appending integers
// until we succeed.
// Then we write argv and envp to the output, santizing them. We also
// write the nccl version.
// We provide some status line to stdout.
// The JSON stream is left with a trailing comma and the top-level
// object open for the next set of top-level items (config and
// results).

// This uses unguarded 'printf' rather than the PRINT() macro because
// is_main_thread is not set up at this point.
void init_json_output(const char *in_path,
                      int argc, char **argv,
                      char **envp) {
  if(in_path == nullptr) {
    return;
  }

  #ifdef MPI_SUPPORT
  int proc;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc);
  if(proc != 0) {
    return;
  }
  #endif

  char *try_path = strdup(in_path);
  int try_count = 0;
  json_report_fp = fopen(try_path, "wx");
  while(json_report_fp == NULL) {
    if(errno != EEXIST) {
      printf("# skipping json output; %s not accessible\n", try_path);
      free(try_path);
      return;
    }
    free(try_path);
    if(asprintf(&try_path, "%s.%d", in_path, try_count++) == -1) {
      printf("# skipping json output; failed to probe destination\n");
      return;
    }
    json_report_fp = fopen(try_path, "wx");
  }

  printf("# Writing Json output to %s\n", try_path);
  free(try_path);

  write_json = true;

  state_cap = 16;
  states = (json_state_t*) malloc(sizeof(json_state_t) * state_cap);
  assert(states);

  json_start_object(); // will be closed finalize_json_output

  json_key("start_time");
  {
    char timebuffer[128];
    format_now(timebuffer, sizeof(timebuffer));
    json_str(timebuffer);
  }

  json_key("args");
  json_start_list();
  for(int i = 0; i < argc; i++) {
    json_str(argv[i]);
  }
  json_finish_list();

  json_key("env");
  json_start_list();
  for(char **e = envp; *e; e++) {
    json_str(*e);
  }
  json_finish_list();
  json_key("nccl_version"); json_int(test_ncclVersion);
}

// This cleans up the json output, finishing the object and closing the file.
// If we were not writing json output, we don't do anything.
void finalize_json_output() {
  if(write_json) {

    json_key("end_time");
    char timebuffer[128];
    format_now(timebuffer, sizeof(timebuffer));
    json_str(timebuffer);

    json_finish_object();

    assert(json_curr_state() == JSON_NONE);
    free(states);
    state_n = 0;
    state_cap = 0;

    fclose(json_report_fp);
    json_report_fp = nullptr;
  }
}

struct rankinfo_t {
  int rank;
  int group;
  int pid;
  char hostname[1024];
  int device;
  char device_hex[128];
  char devinfo[1024];
};

// Helper function to parse the device info lines passed via MPI to the root rank.
// This fills 'rank' with the parsed contents of 'instring'.
static int parse_rankinfo(rankinfo_t *rank, const char *instring) {
  int end;
  sscanf(instring,
         "#  Rank %d Group %d Pid %d on %1024s device %d [%128[^]]] %1024[^\n]\n%n",
         &rank->rank,
         &rank->group,
         &rank->pid,
         rank->hostname,
         &rank->device,
         rank->device_hex,
         rank->devinfo,
         &end);
  return end;
}

static void json_rankinfo(const rankinfo_t *ri) {
  json_start_object();
  json_key("rank");        json_int(ri->rank);
  json_key("group");       json_int(ri->group);
  json_key("pid");         json_int(ri->pid);
  json_key("hostname");    json_str(ri->hostname);
  json_key("device");      json_int(ri->device);
  json_key("device_hex");  json_str(ri->device_hex);
  json_key("device_info"); json_str(ri->devinfo);
  json_finish_object();
}

// Write the start of a benchmark output line containing the bytes &
// op type, both to stdout and to json if we are writing there.
void write_benchmark_line_preamble(size_t nBytes, size_t nElem, const char typeName[], const char opName[], int root) {
  char rootName[100];
  sprintf(rootName, "%6i", root);
  PRINT("%12li  %12li  %8s  %6s  %6s", nBytes, nElem, typeName, opName, rootName);

  if(write_json && is_main_thread) {
    json_start_object();
    json_key("size");  json_int(nBytes);
    json_key("count"); json_int(nElem);
    json_key("type");  json_str(typeName);
    json_key("redop"); json_str(opName);
    json_key("root");  json_str(rootName);
  }
}

// Finish a result record we were writing to stdout/json
void write_benchmark_line_terminator(int actualIters, const char *name) {
  PRINT("  %5d", actualIters);
  PRINT("    %s\n", name);
  if(write_json && is_main_thread) {
    json_key("actual_iterations"); json_int(actualIters);
    json_key("experiment_name");   json_str(name);
    json_finish_object();
  }
}

// Handle a cases where we don't write out of place results
void write_benchmark_line_null_body() {
  PRINT("                                ");  // only do in-place for trace replay
  if(write_json && is_main_thread) {
    json_key("out_of_place"); json_null();
  }
}

void printPerCollPerf(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int actualIters, int per_coll_perf) {
  double varianceTime = 0, varianceAlgBw = 0, varianceBusBw = 0;
  const size_t count = args->nbytes[0][0] / wordSize(type);
  double algBw, busBw;
  char timeStr[100];

  if(write_json && is_main_thread) {
    json_key("per_collective_perf"); json_start_object();
    if (per_coll_perf == 1) {
      json_key("gpus"); json_start_list();
    }
  }

  for (int i = 0; i < args->nGpus; i++) {
    if (write_json && is_main_thread && per_coll_perf == 1) {
      json_start_object();
      json_key("gpu");        json_int(i);
      json_key("iterations"); json_start_list();
    }

    for (int j = 0; j < actualIters; j++) {
      const double timeSec = args->ms[i*(actualIters)+j] / 1.0E3;
      const double timeUsec = timeSec*1.0E6;
      if (timeUsec >= 10000.0) {
        sprintf(timeStr, "%7.0f", timeUsec);
      } else if (timeUsec >= 100.0) {
        sprintf(timeStr, "%7.1f", timeUsec);
      } else {
        sprintf(timeStr, "%7.2f", timeUsec);
      }
      args->collTest->getBw(count, wordSize(type), timeSec, &algBw, &busBw, args->nProcs*args->nThreads*args->nGpus);
      varianceTime += pow((args->meanTime - timeUsec), 2);
      varianceAlgBw += pow((args->meanAlgBw - algBw), 2);
      varianceBusBw += pow((args->meanBusBw - busBw), 2);

      if (write_json && is_main_thread && per_coll_perf == 1) {
        PRINT("\n%35sGpu%2d Coll%3d %4s %7s  %6.2f  %6.2f  %5s\n",
          " ", args->gpus[i], j, " ", timeStr, algBw, busBw, "N/A");

        json_start_object();
        json_key("iteration"); json_int(j);
        json_key("time");      json_double(timeSec);
        json_key("alg_bw");  json_double(algBw);
        json_key("bus_bw");   json_double(busBw);
        json_finish_object();
      }
    }
    if (write_json && is_main_thread && per_coll_perf == 1) {
      json_finish_list(); json_finish_object();
    }
  }

  if (write_json && is_main_thread && per_coll_perf == 1) {
    json_finish_list(); // close all gpus records

  }

  varianceTime /= actualIters;
  varianceAlgBw /= actualIters;
  varianceBusBw /= actualIters;

  const double coeffVarTime = sqrt(varianceTime)/args->meanTime;
  const double coeffVarAlgBw = sqrt(varianceAlgBw)/args->meanAlgBw;
  const double coeffVarBusBw = sqrt(varianceBusBw)/args->meanBusBw;
  PRINT("\n%24sCoefficient of variation %5s %1.4f  %1.4f  %1.4f\n", " ", " ", coeffVarTime, coeffVarAlgBw, coeffVarBusBw);

  if (write_json && is_main_thread && per_coll_perf == 1) {
    json_key("coeff_variance_time");     json_double(coeffVarTime);
    json_key("coeff_variance_alg_bw"); json_double(coeffVarAlgBw);
    json_key("coeff_variance_bus_bw");   json_double(coeffVarBusBw);
    json_finish_object();// close per-coll
  }
}


// Write the performance-related payload to stdout/json.
// We call this function twice at the top level per test: once for out-of-place, and once for in-place.
// The Json output assumes out-of-place happens first.
void write_benchmark_line_body(double timeUsec, double totalTime, double algBw, double busBw, double sideBw, bool reportErrors, int64_t wrongElts, bool report_cputime, bool out_of_place, bool simulate) {
  char timeStr[100];
  if (timeUsec >= 10000.0) {
    sprintf(timeStr, "%7.0f", timeUsec);
  } else if (timeUsec >= 100.0) {
    sprintf(timeStr, "%7.1f", timeUsec);
  } else {
    sprintf(timeStr, "%7.2f", timeUsec);
  }

  char estTimeStr[100];
  if (simulate) {
    if (totalTime >= 10000.0) {
      sprintf(estTimeStr, "%7.0f", totalTime);
    } else if (totalTime >= 100.0) {
      sprintf(estTimeStr, "%7.1f", totalTime);
    } else {
      sprintf(estTimeStr, "%7.2f", totalTime);
    }
  }

  if (reportErrors) {
    if (simulate) {
      if (side_comp == 1) {
        PRINT("  %7s  %6.2f  %6.2f  %6g %6.2f %9s", timeStr, algBw, busBw, (double)wrongElts, sideBw, estTimeStr);
      } else {
        PRINT("  %7s  %6.2f  %6.2f  %6g %9s", timeStr, algBw, busBw, (double)wrongElts, estTimeStr);
      }
    } else {
      if (side_comp == 1) {
        PRINT("  %7s  %6.2f  %6.2f  %6g %6.2f", timeStr, algBw, busBw, (double)wrongElts, sideBw);
      } else {
        PRINT("  %7s  %6.2f  %6.2f  %6g", timeStr, algBw, busBw, (double)wrongElts);
      }
    }
  } else {
    if (simulate) {
      if (side_comp == 1) {
        PRINT("  %7s  %6.2f  %6.2f    N/A %6.2f %9s", timeStr, algBw, busBw, sideBw, estTimeStr);
      } else {
        PRINT("  %7s  %6.2f  %6.2f    N/A %9s", timeStr, algBw, busBw, estTimeStr);
      }
    } else {
      if (side_comp == 1) {
        PRINT("  %7s  %6.2f  %6.2f    N/A %6.2f", timeStr, algBw, busBw, sideBw);
      } else {
        PRINT("  %7s  %6.2f  %6.2f    N/A", timeStr, algBw, busBw);
      }
    }
  }

  if(write_json && is_main_thread) {
    json_key(out_of_place ? "out_of_place" : "in_place");
    json_start_object();
    json_key(report_cputime ? "cpu_time" : "time"); json_double(timeUsec);
    json_key("alg_bw");                            json_double(algBw);
    json_key("bus_bw");                             json_double(busBw);
    json_key("nwrong");                             (reportErrors ? json_double((double)wrongElts) : json_null());
    json_key("side_comp_bw");                       (side_comp == 1 ? json_double(sideBw) : json_null());
    json_key("estimated_time");                     (simulate ? json_double(totalTime) : json_null());
    json_finish_object();
  }
}

// This writes out a report about the run parameters and devices
// involved to stdout and json.  For MPI, this will use a collective
// to gather from each rank to the root.

// Root then consumes this output, printing raw lines for stdout and
// parsing them for JSON for proper formatting.

// Perhaps actually sending records around instead of formatted
// strings would be smarter/easier, but I chose to adapt what was
// already in place.
testResult_t write_device_report(size_t *maxMem, int localRank, int proc, int totalProcs, int color, const char hostname[]) {
  PRINT("# nThread %d nGpus %d minBytes %ld maxBytes %ld step: %ld(%s) warmup iters: %d iters: %d agg iters: %d validation: %d graph: %d\n",
        nThreads, nGpus, minBytes, maxBytes,
        (stepFactor > 1)?stepFactor:stepBytes, (stepFactor > 1)?"factor":"bytes",
        warmup_iters, iters, agg_iters, datacheck, cudaGraphLaunches);
  if (blocking_coll) PRINT("# Blocking Enabled: wait for completion and barrier after each collective \n");
  if (parallel_init) PRINT("# Parallel Init Enabled: threads call into NcclInitRank concurrently \n");
  PRINT("#\n");

  if(write_json && is_main_thread) {
    json_key("config");
    json_start_object();
    json_key("nthreads");      json_int(nThreads);
    json_key("ngpus");         json_int(nGpus);
    json_key("minimum_bytes"); json_size_t(minBytes);
    json_key("maximum_bytes"); json_size_t(maxBytes);
    if(stepFactor > 1) {
      json_key("step_factor");   json_int(stepFactor);
    }
    else {
      json_key("step_bytes");  json_size_t(stepBytes);
    }

    json_key("warmup_iters");          json_int(warmup_iters);
    json_key("iterations");            json_int(iters);
    json_key("aggregated_iterations"); json_int(agg_iters);
    json_key("validation");            json_int(datacheck);
    json_key("graph");                 json_int(cudaGraphLaunches);
    json_key("blocking_collectives");  json_bool(blocking_coll);
    json_key("parallel_init");         json_bool(parallel_init);
  }

  PRINT("# Using devices\n");
#define MAX_LINE 2048
  char line[MAX_LINE];
  int len = 0;
  const char* envstr = getenv("NCCL_TESTS_DEVICE");
  const int gpu0 = envstr ? atoi(envstr) : -1;
  for (int i=0; i<nThreads*nGpus; i++) {
    const int cudaDev = (gpu0 != -1 ? gpu0 : localRank*nThreads*nGpus) + i;
    const int rank = proc*nThreads*nGpus+i;
    cudaDeviceProp prop;
    CUDACHECK(cudaGetDeviceProperties(&prop, cudaDev));
    len += snprintf(line+len, MAX_LINE-len, "#  Rank %2d Group %2d Pid %6d on %10s device %2d [0x%02x] %s\n",
                    rank, color, getpid(), hostname, cudaDev, prop.pciBusID, prop.name);
    *maxMem = std::min(*maxMem, prop.totalGlobalMem);
  }

#if MPI_SUPPORT
  char *lines = (proc == 0) ? (char *)malloc(totalProcs*MAX_LINE) : NULL;
  // Gather all output in rank order to root (0)
  MPI_Gather(line, MAX_LINE, MPI_BYTE, lines, MAX_LINE, MPI_BYTE, 0, MPI_COMM_WORLD);
  if (proc == 0) {
    if(write_json && is_main_thread) {
      json_key("devices");
      json_start_list();
    }
    for (int p = 0; p < totalProcs; p++) {
      PRINT("%s", lines+MAX_LINE*p);
      if(write_json && is_main_thread) {
        rankinfo_t rankinfo;
        parse_rankinfo(&rankinfo, lines + MAX_LINE*p);
        json_rankinfo(&rankinfo);
      }
    }
    if(write_json && is_main_thread) {
      json_finish_list();
    }
    free(lines);
  }
  MPI_Allreduce(MPI_IN_PLACE, maxMem, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
#else
  PRINT("%s", line);
  if(write_json && is_main_thread) {
    rankinfo_t rankinfo;
    parse_rankinfo(&rankinfo, line);
    json_key("devices");
    json_start_list();
    json_rankinfo(&rankinfo);
    json_finish_list();
  }
#endif
  if(write_json && is_main_thread) {
    json_finish_object();
  }

  return testSuccess;
}

// Write a result header to stdout/json.
// Json results object and contained table list are left open
void write_result_header(bool report_cputime, bool simulate) {
  const char* timeStr = report_cputime ? "cputime" : "time";
  PRINT("#\n");
  PRINT("# %10s  %12s  %8s  %6s  %6s                out-of-place                                 in-place          \n", "", "", "", "", "");
  if (simulate) {
    PRINT("# %10s  %12s  %8s  %6s  %6s  %7s  %6s  %6s  %6s  %8s  %7s  %6s  %6s  %6s  %8s  %5s\n", "size", "count", "type", "redop", "root",
          timeStr, "algbw", "busbw", "#wrong", "esttime", timeStr, "algbw", "busbw", "#wrong", "esttime", "#iters");
    PRINT("# %10s  %12s  %8s  %6s  %6s  %7s  %6s  %6s  %6s  %8s  %7s  %6s  %6s  %6s  %8s  %5s\n", "(B)", "(elements)", "", "", "",
          "(us)", "(GB/s)", "(GB/s)", "", "(us)", "(us)", "(GB/s)", "(GB/s)", "", "(us)", "");
  } else {
    PRINT("# %10s  %12s  %8s  %6s  %6s  %7s  %6s  %6s  %6s  %7s  %6s  %6s  %6s  %5s\n", "size", "count", "type", "redop", "root",
          timeStr, "algbw", "busbw", "#wrong", timeStr, "algbw", "busbw", "#wrong", "#iters");
    PRINT("# %10s  %12s  %8s  %6s  %6s  %7s  %6s  %6s  %6s  %7s  %6s  %6s  %6s  %5s\n", "(B)", "(elements)", "", "", "",
          "(us)", "(GB/s)", "(GB/s)", "", "(us)", "(GB/s)", "(GB/s)", "", "");
  }

  if(write_json && is_main_thread) {
    json_key("results"); json_start_list();
  }
}

// Write the footer for results to stdout/json.
// We close the table list and write out the summary items.
// Results object is left open for errors.
void write_result_footer(const int errors[], const double bw[], double check_avg_bw) {

  if(write_json && is_main_thread) {
    json_finish_list();
  }

  PRINT("# Out of bounds values : %d %s\n", errors[0], errors[0] ? "FAILED" : "OK");
  PRINT("# Avg bus bandwidth    : %g %s\n", bw[0], check_avg_bw == -1 ? "" : (bw[0] < check_avg_bw*(0.9) ? "FAILED" : "OK"));
  PRINT("#\n");

  if(write_json && is_main_thread) {
    json_key("out_of_bounds");
    json_start_object();
    json_key("count");      json_int(errors[0]);
    json_key("okay");       json_bool(errors[0] == 0);
    json_finish_object();
    json_key("average_bus_bandwidith");
    json_start_object();
    json_key("bandwidith"); json_double(bw[0]);
    json_key("okay");       check_avg_bw == -1 ? json_str("unchecked") : json_bool(bw[0] >= check_avg_bw*(0.9));
    json_finish_object();
  }
}

// Write out remaining errors to stdout/json.
void write_errors() {
  const char *error = ncclGetLastError(NULL);
  if(error && strlen(error) > 0) {
    PRINT("# error: %s\n", error);
  }
  if(write_json && is_main_thread) {
    json_key("errors");
    json_start_list();
    if(error) {
      json_str(error);
    }
    json_finish_list();
  }
}
