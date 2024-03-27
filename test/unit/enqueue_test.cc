#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "nccl.h"

void check(ncclResult_t res, char const* file, int line) {
  if (res != ncclSuccess) {
    fprintf(stderr, "NCCL failed with %d at %s:%d\n", (int)res, file, line);
    exit(1);
  }
}
void check(cudaError_t res, char const* file, int line) {
  if (res != cudaSuccess) {
    fprintf(stderr, "CUDA failed with %s at %s:%d\n", cudaGetErrorName(res), file, line);
    exit(1);
  }
}
#define CHECK(call) check(call, __FILE__, __LINE__)

static uint64_t rngState[2] = {0, 0};
void seedrng(uint64_t seed) {
  rngState[0] = seed;
  rngState[1] = seed;
}
uint64_t rng() {
  rngState[0] ^= 0x212f25a997db0781 ^ rngState[0]>>32;
  rngState[0] *= 0x4f8bee3ae18deaab;
  rngState[1] ^= 0x16db6b85ef759723 ^ rngState[1]>>32;
  rngState[1] *= 0xbf22f4d66d492977;
  rngState[0] += rngState[1]<<14 | rngState[1]>>(64-14);
  rngState[1] += rngState[0]<<37 | rngState[0]>>(64-37);
  return rngState[0];
}

long envget(char const *name, long deft) {
  char const *s = getenv(name);
  long ans = deft;
  sscanf((s?s:""), "%ld", &ans);
  return ans;
}

int main() {
  long bufSize = envget("BUF_SIZE", 1<<20);
  bufSize = (bufSize + 16-1) & -16;

  // initialize ////////////////////////////////////////////////////////////////

  int nRanks = 0;
  CHECK(cudaGetDeviceCount(&nRanks));
  ncclComm_t* comms = new ncclComm_t[nRanks];
  cudaStream_t* streams = new cudaStream_t[nRanks];
  char** bufs = new char*[nRanks];
  uint8_t* hostInput = new uint8_t[bufSize*nRanks];
  uint8_t* hostOutput = new uint8_t[bufSize*nRanks];

  CHECK(ncclCommInitAll(comms, nRanks, NULL));

  for (int r=0; r < nRanks; r++) {
    CHECK(cudaSetDevice(r));
    CHECK(cudaStreamCreateWithFlags(&streams[r], cudaStreamNonBlocking));
    CHECK(cudaMalloc(&bufs[r], bufSize));
  }

  for (int r=0; r < nRanks; r++) {
    uint8_t* vals = hostInput + r*bufSize;
    for (int i=0; i < bufSize; i++) {
      vals[i] = uint8_t(i ^ r);
    }
    CHECK(cudaSetDevice(r));
    CHECK(cudaMemcpyAsync(bufs[r], vals, bufSize, cudaMemcpyHostToDevice, streams[r]));
  }

  // nccl launches /////////////////////////////////////////////////////////////
  constexpr int AllReduce=0, AllGather=1, ReduceScatter=2, SendRecv=3;

  CHECK(ncclGroupStart());
  size_t off = 0;
  seedrng(0);
  while (off + 1024*4*nRanks <= bufSize) {
    uint64_t rbits = rng();
    int coll = rbits%4; rbits>>=2;
    int red = rbits%3; rbits>>=2;
    int nElt = 1 + rbits%1024; rbits>>=10;
    int eltSize = rbits%2 ? 4 : 1; rbits>>=1;

    ncclDataType_t ty = eltSize==1 ? ncclUint8 : ncclUint32;
    ncclRedOp_t op = red==0 ? ncclSum : red==1 ? ncclMin : ncclMax;

    size_t off1 = off;
    for (int r=0; r < nRanks; r++) {
      switch (coll) {
      case AllReduce:
        CHECK(ncclAllReduce(bufs[r]+off, bufs[r]+off, nElt, ty, op, comms[r], streams[r]));
        off1 = off + nElt*eltSize;
        break;
      case AllGather:
        CHECK(ncclAllGather(bufs[r]+off + r*nElt*eltSize, bufs[r]+off, nElt, ty, comms[r], streams[r]));
        off1 = off + nElt*eltSize*nRanks;
        break;
      case ReduceScatter:
        CHECK(ncclReduceScatter(bufs[r]+off, bufs[r]+off + r*nElt*eltSize, nElt, ty, op, comms[r], streams[r]));
        off1 = off + nElt*eltSize*nRanks;
        break;
      case SendRecv:
        CHECK(ncclSend(bufs[r]+off, nElt, ty, (r+1)%nRanks, comms[r], streams[r]));
        CHECK(ncclRecv(bufs[r]+off + nElt*eltSize, nElt, ty, (r-1+nRanks)%nRanks, comms[r], streams[r]));
        off1 = off + 2*nElt*eltSize;
        break;
      }
    }
    off = (off1 + 4-1) & -size_t(4); // align up to 4
  }
  CHECK(ncclGroupEnd());

  // validation ////////////////////////////////////////////////////////////////

  for (int r=0; r < nRanks; r++) {
    uint8_t* dst = hostOutput + r*bufSize;
    CHECK(cudaSetDevice(r));
    CHECK(cudaMemcpyAsync(dst, bufs[r], bufSize, cudaMemcpyDeviceToHost, streams[r]));
  }
  for (int r=0; r < nRanks; r++) {
    CHECK(cudaSetDevice(r));
    CHECK(cudaStreamSynchronize(streams[r]));
  }

  uint64_t errors = 0;
  off = 0;
  seedrng(0);
  while (off + 1024*4*nRanks <= bufSize) {
    uint64_t rbits = rng();
    int coll = rbits%4; rbits>>=2;
    int red = rbits%3; rbits>>=2;
    int nElt = 1 + rbits%1024; rbits>>=10;
    int eltSize = rbits%2 ? 4 : 1; rbits>>=1;

    //ncclDataType_t ty = eltSize==1 ? ncclUint8 : ncclUint32;
    ncclRedOp_t op = red==0 ? ncclSum : red==1 ? ncclMin : ncclMax;

    switch (coll) {
    case AllReduce:
    case AllGather:
    case ReduceScatter:
      for (int ix=0; ix < (coll==AllReduce ? 1 : nRanks)*nElt; ix++) {
        uint32_t expect = 0;
        int r0 = coll==AllGather ? ix/nElt : 0;
        int r1 = coll==AllGather ? r0+1 : nRanks;
        for (int r=r0; r < r1; r++) {
          uint8_t* inps = hostInput + r*bufSize + off;
          uint32_t x = eltSize==1 ? inps[ix] : ((uint32_t*)inps)[ix];
          if (r==r0) expect = x;
          else {
            switch (op) {
            case ncclSum: expect += x; break;
            case ncclMin: expect = x < expect ? x : expect; break;
            case ncclMax: expect = x < expect ? expect : x; break;
            default: break;
            }
          }
        }
        if (eltSize==1) expect = uint8_t(expect);
        r0 = coll==ReduceScatter ? ix/nElt : 0;
        r1 = coll==ReduceScatter ? r0+1 : nRanks;
        for (int r=r0; r < r1; r++) {
          uint8_t* outs = hostOutput + r*bufSize + off;
          uint32_t y = eltSize==1 ? outs[ix] : ((uint32_t*)outs)[ix];
          if (expect != y) printf("exp=%x got=%x\n", expect, y);
          errors += expect != y;
        }
      }
      off += (coll==AllReduce ? 1 : nRanks)*nElt*eltSize;
      break;
    case SendRecv:
      if (eltSize > 1) { nElt *= eltSize; eltSize = 1; }
      for (int r=0; r < nRanks; r++) {
        int sender = (r-1+nRanks)%nRanks;
        uint8_t* inps = hostInput + sender*bufSize + off;
        uint8_t* outs = hostOutput + r*bufSize + off + nElt*eltSize;
        for (int i=0; i < nElt; i++) {
          errors += inps[i] != outs[i];
        }
      }
      off += 2*nElt*eltSize;
      break;
    }
    off = (off + 4-1) & -size_t(4); // align up to 4
  }

  if (errors != 0) {
    fprintf(stderr, "ERROR: Validation failed with %ld errors.\n", (long)errors);
    exit(1);
  }

  // teardown //////////////////////////////////////////////////////////////////
  for (int r=0; r < nRanks; r++) {
    CHECK(ncclCommDestroy(comms[r]));
    CHECK(cudaFree(bufs[r]));
    CHECK(cudaStreamDestroy(streams[r]));
  }
  delete[] bufs;
  delete[] comms;
  delete[] streams;
  delete[] hostInput;
  delete[] hostOutput;

  fprintf(stdout, "SUCCESS\n");
  return 0;
};
