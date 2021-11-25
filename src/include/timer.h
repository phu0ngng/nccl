/*************************************************************************
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef TIMER_H
#define TIMER_H
#if 1
#include <sys/time.h>
#include <x86intrin.h>
static double gettime() {
  static double freq = -1;
  if (freq == -1) {
    //printf("Calibrating clock, please wait ...");
    fflush(stdout);
    struct timeval tv;
    gettimeofday(&tv, NULL);
    uint64_t timeCycles = __rdtsc();
    double time = - tv.tv_sec*1E6 - tv.tv_usec;
    uint64_t total = 0ULL;
    for (int i=0; i<1000; i++) total += __rdtsc();
    gettimeofday(&tv, NULL);
    timeCycles = __rdtsc() - timeCycles;
    time += tv.tv_sec*1E6 + tv.tv_usec;
    freq = timeCycles/time;
    //printf("Time %g, rdtsc delta %ld, freq %g cycles/usec\n", time, timeCycles, freq);
  }
  return __rdtsc()/freq;
}
static uint64_t counts[8];
static double times[8];
static double startTimes[8];
#define TIME_START(index) do { \
  counts[index]++; \
  startTimes[index] = gettime(); \
} while (0);

#define TIME_STOP(index) do { \
  times[index] += gettime() - startTimes[index]; \
} while (0);

#define TIME_CANCEL(index) do { \
  counts[index]--; \
} while (0);

#define TIME_PRINT(name) do { \
  printf("%s stats", name); \
  for (int i=0; i<8; i++) { \
    if (counts[i]) printf(" [%d] %g/%ld = %g", i, times[i], counts[i], times[i]/counts[i]); \
    counts[i] = 0; \
  } \
  printf("\n"); \
} while (0);
#else
#define TIME_START(index) while(0);
#define TIME_STOP(index) while(0);
#define TIME_CANCEL(index) while(0);
#define TIME_PRINT(name)
#endif
#endif
