#pragma once

#include "common.h"

void initJsonOutput(const char *path, int argc, char **argv, char **envp);
void finalizeJsonOutput();

void printPerCollPerf(struct threadArgs* args, ncclDataType_t type, ncclRedOp_t op, int root, int actualIters, int per_coll_perf);
void writeBenchmarkLinePreamble(size_t nBytes, size_t nElem, const char typeName[], const char opName[], int root);
void writeBenchmarkLineTerminator(int actualIters, const char *name);
void writeBenchMarkLineNullBody();
void writeBenchmarkLineBody(double timeUsec, double totalTime, double algBw, double busBw, double sideBw, bool reportErrors, int64_t wrongElts, bool report_cputime, bool out_of_place, bool simulate);
testResult_t writeDeviceReport(size_t *maxMem, int localRank, int proc, int totalProcs, int color, const char hostname[]);
void writeResultHeader(bool report_cputime, bool simulate);
void writeResultFooter(const int errors[], const double bw[], double check_avg_bw);
void writeErrors();
