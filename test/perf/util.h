#pragma once

#include "common.h"

void init_json_output(const char *path, int argc, char **argv, char **envp);
void finalize_json_output();

void printPerCollPerf(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int actualIters, int per_coll_perf);
void write_benchmark_line_preamble(size_t nBytes, size_t nElem, const char typeName[], const char opName[], int root);
void write_benchmark_line_terminator(int actualIters, const char *name);
void write_benchmark_line_null_body();
void write_benchmark_line_body(double timeUsec, double totalTime, double algBw, double busBw, double sideBw, bool reportErrors, int64_t wrongElts, bool report_cputime, bool out_of_place, bool simulate);
testResult_t write_device_report(size_t *maxMem, int localRank, int proc, int totalProcs, int color, const char hostname[]);
void write_result_header(bool report_cputime, bool simulate);
void write_result_footer(const int errors[], const double bw[], double check_avg_bw);
void write_errors();
