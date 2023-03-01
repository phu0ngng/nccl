#include <algorithm>
#include <atomic>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <future>

#include <unistd.h>
#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include<signal.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#if CUDART_VERSION >= 11000
#include <cuda_bf16.h>
#endif
#include <mpi.h>
#include <nccl.h>

#include "../verifiable/verifiable.h"

using std::size_t;
using std::int8_t;
using std::int16_t;
using std::int32_t;
using std::int64_t;
using std::intptr_t;
using std::uint8_t;
using std::uint16_t;
using std::uint32_t;
using std::uint64_t;
using std::uintptr_t;

// physical host count, my index
int phost_n = 0, phost_me = -1;
// mpi process count, my index
int mpi_rank_n = 0, mpi_rank_me = -1;

bool opt_verbose = false;
bool opt_progress_thread = false;
bool opt_force_fit = false;
bool opt_force_size_one = false;
bool opt_disable_check = false;

// map physical host to list of mpi ranks it owns
std::unique_ptr<std::vector<int>[]> phost_ranks;

// Flag signalling test end by main thread, used by progress thread
std::promise<void> signal_exit;
std::atomic<size_t> global_ops_submitted{0};
std::atomic<size_t> global_ops_completed{0};
std::thread progress_thread;

// How many data ops will this process submit for this trace?
size_t data_op_call_count;

////////////////////////////////////////////////////////////////////////////////

void signalHandler( int signum ) {
   std::cerr << "[" << getpid() << "] Interrupt signal (" << signum << ") received.\n";
   exit(signum);
}

uint64_t hashOf(std::string const &s) {
  uint64_t h = 0x518b236040bc0297u;
  for(char c: s) {
    h += c;
    h ^= h>>32;
    h *= 0x9e3779b97f4a7c13u;
  }
  return h;
}

__host__ __device__ uint64_t hashOf(uint64_t a, uint64_t b=0) {
  a += uint64_t(1)<<32;
  a += b;
  a ^= a>>32;
  a *= 0x9e3779b97f4a7c13u;
  a += b>>16 ^ b<<48;
  a ^= a>>32;
  a *= 0xc4ceb9fe1a85ec53u;
  return a;
}

////////////////////////////////////////////////////////////////////////////////

size_t elementSize(ncclDataType_t elt_ty) {
  switch(elt_ty) {
  case ncclInt8:
  case ncclUint8:
    return 1;
  case ncclFloat16:
  #ifdef __CUDA_BF16_TYPES_EXIST__
  case ncclBfloat16:
  #endif
    return 2;
  case ncclInt32:
  case ncclUint32:
  case ncclFloat32:
    return 4;
  case ncclInt64:
  case ncclUint64:
  case ncclFloat64:
    return 8;
  }
  return 0;
}

////////////////////////////////////////////////////////////////////////////////

template<typename T>
T osEnv(char const *name, T const &default_value) {
  char *s = ::getenv(name);
  if(s == nullptr || *s == '\0')
    return default_value;
  else {
    std::istringstream iss{std::string(s)};
    T value = {};
    iss.operator>>(value);//iss >> value;
    return value;
  }
}
std::string osEnv(char const *name, std::string const &default_value) {
  char *s = ::getenv(name);
  if(s == nullptr || *s == '\0')
    return default_value;
  else
    return std::string(s);
}

////////////////////////////////////////////////////////////////////////////////

/* ByteBuffer: Appendable byte array suited for holding heterogeneous types.
 * std::vector<char> is not suited for this task because it does not guarantee
 * alignment within the buffer. Use the nested `Cursor` class to traverse the
 * values in appended order. It is up to you to ensure the sequence of types
 * pop()'d matches the sequence push()'d.
 */
class ByteBuffer {
  char *buf;
  size_t len, cap;
public:
  ByteBuffer() { buf = nullptr; len = 0; cap = 0;}
  ByteBuffer(ByteBuffer const&) = delete;
  ByteBuffer(ByteBuffer &&that);
  ~ByteBuffer() { ::operator delete(buf); }

  char const* bytes() const { return buf; }
  char      * bytes()       { return buf; }
  size_t size() const { return len; }

  // both reserve and append invalidate cursors.
  void reserve(size_t n);
  template<typename T>
  void append(T const &x);

  class Cursor {
    uintptr_t ptr, end;
  public:
    Cursor(ByteBuffer &that):
      ptr(reinterpret_cast<uintptr_t>(that.bytes())),
      end(ptr + that.size()) {
    }
    bool has_more() const { return ptr != end; }
    template<typename T>
    T& pop();
  };
};

ByteBuffer::ByteBuffer(ByteBuffer &&that) {
  this->buf = that.buf;
  this->len = that.len;
  this->cap = that.cap;
  that.buf = nullptr;
  that.len = that.cap = 0;
}

void ByteBuffer::reserve(size_t n) {
  if(n > cap) {
    size_t cap1 = std::max(n, 2*cap);
    char *buf1 = (char*)::operator new(cap1);
    std::memcpy(buf1, buf, len);
    ::operator delete(buf);
    buf = buf1;
    cap = cap1;
  }
  len = n;
}

template<typename T>
void ByteBuffer::append(T const &x) {
  size_t len1 = (len + alignof(T)-1) & -alignof(T);
  reserve(len1 + sizeof(T));
  std::memcpy(buf + len1, &x, sizeof(T));
}

template<typename T>
T& ByteBuffer::Cursor::pop() {
  ptr = (ptr + alignof(T)-1) & -uintptr_t(alignof(T));
  T &ans = *reinterpret_cast<T*>(ptr);
  ptr += sizeof(T);
  return ans;
}

////////////////////////////////////////////////////////////////////////////////
// SharedTable<T>: a single-assignment thread-safe map from uint64_t to unique
// instances of T which are allocated on first use.

struct SharedTable_Base {
  struct cache_hash {
    size_t operator()(std::pair<void*,uint64_t> const &x) const {
      uintptr_t h = reinterpret_cast<uintptr_t>(x.first);
      h += h<<20;
      h ^= x.second;
      return h;
    }
  };
  std::mutex lock;
  std::unordered_map<uint64_t, void*> table;
  static thread_local std::unordered_map<
      std::pair<void*,uint64_t>, void*, cache_hash
    > cache;

  SharedTable_Base() = default;
  SharedTable_Base(SharedTable_Base const&) = delete;
};
thread_local std::unordered_map<
    std::pair<void*,uint64_t>, void*, SharedTable_Base::cache_hash
  > SharedTable_Base::cache;

template<typename T>
class SharedTable: private SharedTable_Base {
public:
  ~SharedTable();
  T& operator[](uint64_t key);
  // emplace: if key not found, newly allocated T is constructed via T(arg...)
  template<typename ...Arg>
  T& emplace(uint64_t key, Arg ...arg);
};

template<typename T>
SharedTable<T>::~SharedTable() {
  while(!table.empty()) {
    auto it = table.begin();
    delete (T*)it->second;
    table.erase(it);
  }
}

template<typename T>
T& SharedTable<T>::operator[](uint64_t key) {
  return emplace(key);
}

template<typename T>
template<typename ...Arg>
T& SharedTable<T>::emplace(uint64_t key, Arg ...arg) {
  auto it = cache.find({(void*)this, key});
  if(it != cache.end())
    return *(T*)it->second;
  else {
    void *ans;
    { std::unique_lock<std::mutex> locked(lock);
      void **p = &table[key];
      if(*p == nullptr) *p = new T(arg...);
      ans = *p;
    }
    cache[{(void*)this, key}] = ans;
    return *(T*)ans;
  }
}

// Progress thread
void progressThread(std::future<void> exit_future) {
  while (exit_future.wait_for(std::chrono::seconds(1)) == std::future_status::timeout) {
    fprintf(stderr, "[%u] rank 0 data_ops progress: [%lu posted / %lu completed / %lu total]\n",
      getpid(), global_ops_submitted.load(), global_ops_completed.load(), data_op_call_count);
  }

  fprintf(stderr, "[%u] rank 0 signalled: [%lu posted / %lu completed / %lu total]\n",
    getpid(), global_ops_submitted.load(), global_ops_completed.load(), data_op_call_count);
}

////////////////////////////////////////////////////////////////////////////////

struct CudaException: public std::runtime_error {
  static std::string Message(char const *file, int line, char const *expr, cudaError_t error) {
    char m[1024];
    std::snprintf(m, sizeof(m), "CUDA error at %s:%d\n expr: \"%s\"\n error: \"%s\"", file, line, expr, cudaGetErrorString(error));
    return std::string(m);
  }
  CudaException(char const *file, int line, char const *expr, cudaError_t error):
    runtime_error(Message(file, line, expr, error)) {
  }
};

#define CUDA_CHECK(expr) do { cudaError_t err943dfc31 = expr; if(err943dfc31 != cudaSuccess) throw CudaException(__FILE__, __LINE__, #expr, err943dfc31); } while(0)

struct NcclException: public std::runtime_error {
  static std::string Message(char const *file, int line, char const *expr, ncclResult_t result) {
    char m[1024];
    std::snprintf(m, sizeof(m), "NCCL error at %s:%d\n expr: \"%s\"\n error: \"%s\"", file, line, expr, ncclGetErrorString(result));
    return std::string(m);
  }
  NcclException(char const *file, int line, char const *expr, ncclResult_t result):
    runtime_error(Message(file, line, expr, result)) {
  }
};

#define NCCL_CHECK(expr) do { ncclResult_t res943dfc31 = expr; if(res943dfc31 != ncclSuccess) throw NcclException(__FILE__, __LINE__, #expr, res943dfc31); } while(0)

////////////////////////////////////////////////////////////////////////////////

struct SequenceControl {
  std::mutex mux;
  int thread_seq=0; // sequential access mediator
  std::condition_variable cvar;

  std::unique_lock<std::mutex> lock(int seq) {
    std::unique_lock<std::mutex> locked(this->mux);
    while(this->thread_seq != seq)
    {
      this->cvar.wait(locked);
    }

    this->thread_seq += 1;
    this->cvar.notify_all();
    return locked;
  }
};

////////////////////////////////////////////////////////////////////////////////
// VirtualComm: Wrap a NCCL comm, indexed by uint64_t vcomm id

// Scalars on the device for use with premulsum.
struct ScalarsStruct {
  int8_t   i8;
  uint8_t  u8;
  int32_t i32;
  uint32_t u32;
  int64_t i64;
  uint64_t u64;
  half f16;
  #ifdef __CUDA_BF16_TYPES_EXIST__
  __nv_bfloat16 bf16;
  #endif
  float f32;
  double f64;
};

struct VirtualComm {
  ncclComm_t comm;
  uint64_t vunique;                       // hash of unique-id of this comm
  int device;                             // local CUDA device
  int coll_seq=0;                         // counts collective ops launched
  std::atomic<int> nccl_ops_in_flight{0}; // counts current ops in flight per-communicator
  std::unique_ptr<uint64_t[]> p2p_send_seqs, p2p_recv_seqs; // elements sent/recvd per peer
  std::unordered_map<int, ncclRedOp_t> vredops; // dynamically created ncclRedOp_t's indexed by virtual redop id
  ScalarsStruct *device_scalars = nullptr;
  SequenceControl sequence;
};

SharedTable<VirtualComm> vcomm_table;
SharedTable<SequenceControl> rank_group_seq_table;

namespace CudaHelp {
//private:
  struct AddressSpan {
    uintptr_t lo, hi;
  };
  struct ArenaState {
    uintptr_t hi;
    std::vector<AddressSpan> objs;
  };
  struct DeviceState {
    std::map<uintptr_t/*lo*/, ArenaState> arenas;
    std::mutex alloc_mutex;
    char *blob;
    int anon_next=0;
    int64_t alloc_counter=0;
    uint64_t last_vcomm=0;
    cudaStream_t anon_streams[4]={};
    SequenceControl sequence;
    size_t free_memory = 0;
    size_t total_memory = 0;
    std::condition_variable dealloc_cvar;
  };
  struct StreamState {
    int device;
    cudaStream_t stream=nullptr;
  };
  int device_n;
  std::unique_ptr<DeviceState[]> devices;
  std::mutex lock;
  std::unordered_map<cudaStream_t, StreamState> streams;
  std::unordered_map<uint64_t, StreamState*> vstreams;
  thread_local std::unordered_map<cudaStream_t, StreamState*> tl_streams;
  thread_local std::unordered_map<uint64_t, StreamState*> tl_vstreams;

//public:
  void initialize();
  void finalize();

  // Returns nullptr on allocation failure
  void* allocate(int device, size_t size, uintptr_t align_like, uint32_t op_id);
  void deallocate(int device, void *p, uint32_t op_id);
  cudaStream_t pickVirtualStream(uint64_t vid, int device);
  cudaStream_t pickAnonStream(int device);
  template<typename FnRef>
  void streamCallback(int device, cudaStream_t stream, FnRef &&fn);
  void streamWaitOnStream(int device_a, cudaStream_t a, int device_b, cudaStream_t b);
  void synchronize();
  std::unique_lock<std::mutex> sequence_lock(int device, int sequence_number, uint64_t vcomm);
};

std::unique_lock<std::mutex> CudaHelp::sequence_lock(int device, int sequence_number, uint64_t vcomm) {
  std::unique_lock<std::mutex> out = devices[device].sequence.lock(sequence_number);

  if (vcomm != devices[device].last_vcomm) {
    VirtualComm *vc = &vcomm_table[vcomm];
    if (opt_verbose) {
      std::fprintf(stderr, "[%u] sequence_lock device=%u sequence_number=%d vcomm=%ld last_vcomm=%ld\n",
        getpid(), device, sequence_number, vcomm, devices[device].last_vcomm);
    }
    while (vc->nccl_ops_in_flight.load() != 0) {
      sched_yield();
    }
  }

  devices[device].last_vcomm = vcomm;

  return out;
}

void CudaHelp::initialize() {
  CUDA_CHECK(cudaGetDeviceCount(&device_n));
  devices.reset(new DeviceState[device_n]);
}

void CudaHelp::finalize() {
  for(int d=0; d < device_n; d++) {
    std::unique_lock<std::mutex> locked(devices[d].alloc_mutex);
    if(!devices[d].arenas.empty()) {
      cudaSetDevice(d);
      for(auto &xy: devices[d].arenas) {
        cudaFree(reinterpret_cast<void*>(xy.first));
      }
    }
  }
}

void* CudaHelp::allocate(int device, size_t size, uintptr_t align_like, uint32_t op_id) {
  align_like %= 16;
  DeviceState *dev = &devices[device];
  std::unique_lock<std::mutex> locked(dev->alloc_mutex);

  dev->alloc_counter++;
  if (opt_verbose)
  {
    std::fprintf(stderr, "[%u] allocate id=%u alloc counter=%ld\n",
      getpid(), op_id, dev->alloc_counter);
  }

  while (true) {
    size_t total_allocated_arena_capacity = 0;
    size_t max_arena_size = 0;
    size_t wasted_arena_capacity = 0;
    uint32_t arena_counter = 0;

    for(auto &xy: dev->arenas) {
      uintptr_t arena_lo = xy.first;
      uintptr_t arena_hi = xy.second.hi;
      uintptr_t arena_size = (arena_hi - arena_lo);

      if (arena_size > max_arena_size) {
        max_arena_size = arena_size;
      }

      total_allocated_arena_capacity += arena_size;
      if(arena_size < size) {
        wasted_arena_capacity += (arena_hi - xy.second.objs.back().hi);
        arena_counter++;
        continue;
      }

      std::vector<AddressSpan> &objs = xy.second.objs;
      uintptr_t free_lo = arena_lo;
      size_t allocated_arena_size_pre = objs.back().hi - arena_lo;

      uintptr_t free_hi;
      for(auto o = objs.begin(); o != objs.end(); ++o) {
        free_hi = o->lo;
        uintptr_t lo = free_lo + (align_like - free_lo)%16;
        if(lo + size <= free_hi) {
          objs.insert(o, {lo, lo+size});
          size_t allocated_arena_size_post = objs.back().hi - arena_lo;

          if (opt_verbose)
          {
            std::fprintf(stderr, "[%u] Allocate obj #%ld arena #%u id=%u num_arenas=%zu obj_size=0x%lx p=0x%lx align_like=%lx arena_size=0x%lx allocated_arena_size_pre=0x%lx, allocated_arena_size_post=0x%lx\n",
              getpid(), o - objs.begin(), arena_counter, op_id, dev->arenas.size(), size, lo, align_like, arena_hi - arena_lo, allocated_arena_size_pre, allocated_arena_size_post);
          }

          return reinterpret_cast<void*>(lo);
        } else {
          wasted_arena_capacity += (free_hi - free_lo);
        }
        free_lo = o->hi;
      }
      free_hi = arena_hi;
      uintptr_t lo = free_lo + (align_like - free_lo)%16;
      if(lo + size <= free_hi) {
        objs.push_back({lo, lo+size});
        size_t allocated_arena_size_post = objs.back().hi - arena_lo;

        if (opt_verbose)
        {
          std::fprintf(stderr, "[%u] Allocate obj #%ld arena #%u id=%u num_arenas=%zu obj_size=0x%lx p=0x%lx align_like=%lx arena_size=0x%lx allocated_arena_size_pre=0x%lx, allocated_arena_size_post=0x%lx\n",
            getpid(), objs.size() - 1, arena_counter, op_id, dev->arenas.size(), size, lo, align_like, arena_hi - arena_lo, allocated_arena_size_pre, allocated_arena_size_post);
        }

        return reinterpret_cast<void*>(lo);
      } else {
          wasted_arena_capacity += (free_hi - free_lo);
      }

      arena_counter++;
    }

    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaMemGetInfo(&dev->free_memory, &dev->total_memory));

    size_t arena_size = (size + 16 + (256<<20)-1) & -size_t(256<<20);

    if (opt_verbose)
    {
      std::fprintf(stderr, "[%u] Attempting to alloc new arena. id=%u arena_size=0x%lx dev->total_memory=0x%lx dev->free_memory=0x%lx buffer_size=0x%lx num_arenas=%zu, total_allocated_arena_capacity=0x%lx, wasted_arena_capacity=0x%lx\n",
        getpid(), op_id, arena_size, dev->total_memory, dev->free_memory, size, dev->arenas.size(), total_allocated_arena_capacity, wasted_arena_capacity);
    }

    if (arena_size > dev->total_memory) {
      // Assert that this is impossible to replay
      std::fprintf(stderr, "[%u] INVALID MEMORY SIZE. id=%u Attempting to alloc arena_size=0x%lx, dev->total_memory=0x%lx dev->free_memory=0x%lx. Make sure that the trace is being played back on a system with equivalent memory.\n",
        getpid(), op_id, arena_size, dev->total_memory, dev->free_memory);
      std::terminate();
    } else if (arena_size > dev->free_memory) {
      // If we don't have enough free space, and we can't allocate this size because of arena allocation fragmentation
      if (size > max_arena_size) {
        std::fprintf(stderr, "[%u] ERROR - Insufficient GPU Memory. Arena fragmentation makes this impossible to alloc. id=%u Attempting to alloc arena_size=0x%lx, dev->free_memory=0x%lx dev->total_memory=0x%lx buffer_size=0x%lx num_arenas=%zu, total_allocated_arena_capacity=0x%lx, wasted_arena_capacity=0x%lx. Make sure there aren't any replay zombie processes (nvidia-smi). Make sure that the trace is being played back on a system with equivalent memory.\n",
          getpid(), op_id, arena_size, dev->free_memory, dev->total_memory, size, dev->arenas.size(), total_allocated_arena_capacity, wasted_arena_capacity);

        arena_counter = 0;
        for(auto &xy: dev->arenas) {

          uintptr_t arena_lo = xy.first;
          uintptr_t arena_hi = xy.second.hi;
          std::vector<AddressSpan> &objs = xy.second.objs;

          std::fprintf(stderr, "  [%u] arena[%u] arena_size=0x%lx arena_lo=0x%lx arena_hi=0x%lx\n",
            getpid(), arena_counter, arena_hi - arena_lo, arena_lo, arena_hi);

          for(auto o = objs.begin(); o != objs.end(); ++o) {
              std::fprintf(stderr, "    [%u] arena[%u] obj[%zu] size=0x%lx lo=0x%lx hi=0x%lx\n",
                getpid(), arena_counter, o - objs.begin(), o->hi - o->lo, o->lo, o->hi);
          }
          arena_counter++;
        }

        std::terminate();
      } else {
        dev->dealloc_cvar.wait(locked);
      }
    } else {
      // Attempt to cudaMalloc for this new arena
      void* p = nullptr;
      cudaError_t err = cudaMalloc(&p, arena_size);
      if(err == cudaErrorMemoryAllocation) {
        std::fprintf(stderr, "[%u] cudaErrorMemoryAllocation. id=%u Attempting to alloc arena_size=0x%lx, dev->free_memory=0x%lx dev->total_memory=0x%lx buffer_size=0x%lx num_arenas=%zu, total_allocated_arena_capacity=0x%lx, wasted_arena_capacity=0x%lx. Make sure there aren't any replay zombie processes (nvidia-smi). Make sure that the trace is being played back on a system with equivalent memory.\n",
          getpid(), op_id, arena_size, dev->free_memory, dev->total_memory, size, dev->arenas.size(), total_allocated_arena_capacity, wasted_arena_capacity);
        dev->dealloc_cvar.wait(locked);
      } else {
        // Allocation successful, or a problem CUDA error
        CUDA_CHECK(err);

        // Create arena from p and return first obj to user
        uintptr_t arena_lo = reinterpret_cast<uintptr_t>(p);
        ArenaState *arena = &dev->arenas[arena_lo];
        arena->hi = arena_lo + arena_size;
        uintptr_t lo = arena_lo + (align_like - arena_lo)%16;
        arena->objs.push_back({lo, lo+size});
        size_t allocated_arena_size_post = arena->objs.back().hi - arena_lo;

        if (opt_verbose) {
          std::fprintf(stderr, "[%u] Allocated obj #%ld arena #%zu arena_lo=0x%lx id=%u num_arenas=%zu obj_size=0x%lx p=0x%lx align_like=%lx arena_size=0x%lx allocated_arena_size_pre=0x%lx, allocated_arena_size_post=0x%lx\n",
            getpid(), arena->objs.size() - 1, dev->arenas.size() - 1, arena_lo, op_id, dev->arenas.size(), size, arena->hi - arena_lo, lo, align_like, 0L, allocated_arena_size_post);
        }

        // Success
        return reinterpret_cast<void*>(lo);
      }
    }
  }
}

void CudaHelp::deallocate(int device, void *p, uint32_t op_id) {
  DeviceState *dev = &devices[device];
  std::unique_lock<std::mutex> locked(dev->alloc_mutex);

  dev->alloc_counter--;
  if (opt_verbose)
  {
    std::fprintf(stderr, "[%u] deallocate id=%u alloc_counter=%ld\n",
      getpid(), op_id, dev->alloc_counter);
  }

  // Get the memory address in our format
  uintptr_t lo = reinterpret_cast<uintptr_t>(p);

  // Find the first arena which has hi > p
  uint32_t arena_num = 0;
  auto a = dev->arenas.begin();
  while(a->second.hi <= lo) {
    ++a;
    arena_num++;
  }

  // Count down from last to first object in this arena
  std::vector<AddressSpan> &objs = a->second.objs;
  bool removed = false;
  std::vector<AddressSpan>::iterator obj_it = objs.end();
  size_t obj_num = objs.size();
  while (obj_it != objs.begin())
  {
    --obj_it;
    --obj_num;

    auto obj = *obj_it;
    if (obj.lo <= reinterpret_cast<uintptr_t>(p) && obj.hi >= reinterpret_cast<uintptr_t>(p))
    {
      size_t obj_size = obj.hi - obj.lo;
      size_t arena_size = a->second.hi - obj.lo;

      size_t allocated_arena_size_pre = objs.back().hi - objs[0].lo;
      size_t allocated_arena_size_post = allocated_arena_size_pre - obj_size;
      objs.erase(obj_it);

      if (opt_verbose)
      {
        std::fprintf(stderr, "[%u] Deallocate obj #%ld arena #%u id=%u num_arenas=%zu obj_size=0x%lx p=0x%lx arena_size=0x%lx allocated_arena_size_pre=0x%lx, allocated_arena_size_post=0x%lx\n",
          getpid(), obj_num, arena_num, op_id, dev->arenas.size(), obj_size, (intptr_t) p, arena_size, allocated_arena_size_pre, allocated_arena_size_post);
      }

      removed = true;
      break;
    }
  }

  assert(removed);
  dev->dealloc_cvar.notify_all();
}

cudaStream_t CudaHelp::pickVirtualStream(uint64_t vid, int device) {
  if (opt_verbose) {
    fprintf(stderr, "[%u] Getting stream for stream vid=0x%lx device=%d\n", getpid(), vid, device);
  }

  { auto it = tl_vstreams.find(vid);
    if(it != tl_vstreams.end()) {
      if (opt_verbose) {
        fprintf(stderr, "[%u] Found thread_local vstream vid=0x%lx device=%d stream=%p\n", getpid(), vid, device, &it->second->stream);
      }
      return it->second->stream;
    }
  }
  std::unique_lock<std::mutex> locked(lock);
  auto it = vstreams.find(vid);
  StreamState *ss;
  if(it == vstreams.end()) {
    if (opt_verbose) {
      fprintf(stderr, "[%u] Couldnt find vstream for vid=0x%lx device=%d\n", getpid(), vid, device);
    }

    cudaStream_t s;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

    if (opt_verbose) {
      fprintf(stderr, "[%u] Allocated vstream vid=0x%lx device=%d stream=%p\n", getpid(), vid, device, s);
    }

    ss = &streams[s];
    ss->device = device;
    ss->stream = s;
    vstreams[vid] = ss;
    return s;
  }
  else
    ss = it->second;
  tl_vstreams[vid] = ss;
  return ss->stream;
}

cudaStream_t CudaHelp::pickAnonStream(int device) {
  DeviceState *dev = &devices[device];
  std::unique_lock<std::mutex> locked(dev->alloc_mutex);
  int i = dev->anon_next;
  dev->anon_next = (dev->anon_next + 1)%4;
  if(dev->anon_streams[i] == nullptr) {
    cudaStream_t s;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    dev->anon_streams[i] = s;
    { std::unique_lock<std::mutex> main_locked(lock);
      StreamState *ss = &streams[s];
      ss->device = device;
      ss->stream = s;
    }
  }
  return dev->anon_streams[i];
}

template<typename FnRef>
void CudaHelp::streamCallback(int device, cudaStream_t stream, FnRef &&fn) {
  using Fn = typename std::decay<FnRef>::type;
  Fn *pfn = new Fn(static_cast<FnRef&&>(fn));
  cudaHostFn_t host_fn = [](void *arg) { (*(Fn*)arg)(); };
  CUDA_CHECK(cudaSetDevice(device));
  CUDA_CHECK(cudaLaunchHostFunc(stream, host_fn, pfn));
}

void CudaHelp::streamWaitOnStream(int device_a, cudaStream_t stream_a, int device_b, cudaStream_t stream_b) {
  if(stream_a == stream_b) return;
  cudaEvent_t e;
  CUDA_CHECK(cudaSetDevice(device_b));
  CUDA_CHECK(cudaEventCreate(&e));
  CUDA_CHECK(cudaEventRecord(e, stream_b));
  if(device_a != device_b)
    CUDA_CHECK(cudaSetDevice(device_a));
  CUDA_CHECK(cudaStreamWaitEvent(stream_a, e, 0U));
  CUDA_CHECK(cudaEventDestroy(e));
}

void CudaHelp::synchronize() {
  for(auto &xy: streams) {
    cudaStream_t s = xy.first;
    int device = xy.second.device;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaStreamSynchronize(s));
  }
}

////////////////////////////////////////////////////////////////////////////////
// CudaHostPool: pooling allocator on top of cudaHostAlloc with
// cudaHostAllocMapped set.

class CudaHostPool {
  static constexpr size_t hunk_size = 8<<10;
  struct Hunk {
    void *mem;
    size_t popn, edge;
  };
  std::mutex mux;
  std::vector<Hunk> hunks;
  void *hold;
public:
  ~CudaHostPool();
  void* allocate(size_t size, size_t align);
  template<typename T>
  T* allocate() { return (T*)allocate(sizeof(T), alignof(T)); }
  void deallocate(void *p);
};

CudaHostPool the_host_pool;

CudaHostPool::~CudaHostPool() {
  for(auto &h: hunks)
    cudaFreeHost(h.mem);
}

void* CudaHostPool::allocate(size_t size, size_t align) {
  std::unique_lock<std::mutex> locked(mux);
  for(auto &h: hunks) {
    size_t off = (h.edge + align-1) & -align;
    if(off + size <= hunk_size) {
      h.edge = off + size;
      h.popn += 1;
      return (char*)h.mem + off;
    }
  }
  Hunk h;
  CUDA_CHECK(cudaHostAlloc(&h.mem, hunk_size, cudaHostAllocMapped));
  h.popn = 1;
  h.edge = size;
  hunks.push_back(h);
  return h.mem;
}

void CudaHostPool::deallocate(void *p) {
  std::unique_lock<std::mutex> locked(mux);
  uintptr_t u = reinterpret_cast<uintptr_t>(p);
  for(int i=0; i < (int)hunks.size(); i++) {
    uintptr_t lo = reinterpret_cast<uintptr_t>(hunks[i].mem);
    uintptr_t hi = lo + hunk_size;
    if(lo <= u && u < hi) {
      if(0 == --hunks[i].popn)
        hunks[i].edge = 0;
      break;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Mailbox: Multi-producer single-consumer queue of one-shot callbacks

class Mailbox {
public:
  struct Message {
    Message *next;
    virtual void execute_and_destruct() = 0;
  };
  template<typename FnRef>
  static Message* makeMessage(FnRef &&fn);
private:
  template<typename Fn>
  struct message_fn: Message {
    Fn fn;
    message_fn(Fn const &fn): fn(fn) {}
    message_fn(Fn &&fn): fn(static_cast<Fn&&>(fn)) {}
    void execute_and_destruct() { fn(); fn.~Fn(); }
  };
  std::condition_variable cvar;
  std::mutex mux;
  Message *msg_head = nullptr, **msg_ptail = &msg_head;
public:
  Mailbox() = default;
  Mailbox(Mailbox const&) = delete;
  void postMessages(Message *head, Message **ptail);
  template<typename FnRef>
  void post(FnRef &&fn);
  void poll(bool wait);
};

template<typename FnRef>
Mailbox::Message* Mailbox::makeMessage(FnRef &&fn) {
  using Fn = typename std::decay<FnRef>::type;
  void *m = ::operator new(sizeof(message_fn<Fn>));
  return ::new(m) message_fn<Fn>(static_cast<FnRef&&>(fn));
}

void Mailbox::postMessages(Message *head, Message **ptail) {
  std::unique_lock<std::mutex> locked(mux);
  *msg_ptail = head;
  msg_ptail = ptail;
  cvar.notify_one();
}

template<typename FnRef>
void Mailbox::post(FnRef &&fn) {
  Message *m = makeMessage(static_cast<FnRef&&>(fn));
  postMessages(m, &m->next);
}

void Mailbox::poll(bool wait) {
  Message *m;
  { std::unique_lock<std::mutex> locked(mux);
    while(wait && msg_head == nullptr)
      cvar.wait(locked);
    *msg_ptail = nullptr;
    m = msg_head;
    msg_head = nullptr;
    msg_ptail = &msg_head;
  }
  while(m) {
    m->execute_and_destruct();
    Message *m1 = m->next;
    ::operator delete(m);
    m = m1;
  }
}

////////////////////////////////////////////////////////////////////////////////
// VirtualThreads: Collection of worker threads named by a virtual id each
// with a Mailbox. The main thread communicates with workers by posting callbacks
// targeted at virtual thread ids. Backing threads are created on-demand 1:1 with
// virtual thread ids.

class VirtualThreads {
  struct State {
    int vtid;
    std::thread worker;
    Mailbox mbox;
    Mailbox::Message *batch_head=nullptr, **batch_ptail=&batch_head;
  };
  std::unordered_map<int, State*> state_by_vtid;
  std::mutex sync_mux;
  std::condition_variable sync_cvar;
  int sync_count;
  static thread_local bool worker_killed;
public:
  static thread_local int current_vtid;
private:
  static void workerMain(State *st);
public:
  ~VirtualThreads();
  void killAll();
  // Post callback to named thread which is created and launched on first mention.
  // Message is held in a batch until submitBatch().
  template<typename FnRef>
  void postBatch(int vtid, FnRef &&fn);
  void postBatchMessage(int vtid, Mailbox::Message *m);
  // submit all posted messages
  void submitBatch();
  // post same Message to all threads, not batched, returns number of threads
  template<typename Fn>
  int postAll(Fn const &fn);
};

thread_local bool VirtualThreads::worker_killed = false;
thread_local int VirtualThreads::current_vtid = -1;

VirtualThreads::~VirtualThreads() {
  killAll();
}

void VirtualThreads::killAll() {
  for(auto &xy: state_by_vtid) {
    xy.second->mbox.post([]() { worker_killed = true; });
    xy.second->worker.join();
    delete xy.second;
  }
  state_by_vtid.clear();
}

void VirtualThreads::postBatchMessage(int vtid, Mailbox::Message *m) {
  State *&st = state_by_vtid[vtid];
  if(st == nullptr) {
    if (opt_verbose) {
      fprintf(stderr, "[%u] Creating new thread for vtid=%d\n", getpid(), vtid);
    }
    st = new State;
    st->vtid = vtid;
    st->worker = std::thread(workerMain, st);
  }
  *st->batch_ptail = m;
  st->batch_ptail = &m->next;
}

template<typename FnRef>
void VirtualThreads::postBatch(int vtid, FnRef &&fn) {
  postBatchMessage(vtid, Mailbox::makeMessage(static_cast<FnRef&&>(fn)));
}

void VirtualThreads::submitBatch() {
  for(auto &xy: state_by_vtid) {
    if(xy.second->batch_head != nullptr) {
      xy.second->mbox.postMessages(xy.second->batch_head, xy.second->batch_ptail);
      xy.second->batch_head = nullptr;
      xy.second->batch_ptail = &xy.second->batch_head;
    }
  }
}

template<typename Fn>
int VirtualThreads::postAll(Fn const &fn) {
  int n = 0;
  for(auto &xy: state_by_vtid) {
    State *st = xy.second;
    Mailbox::Message *m = Mailbox::makeMessage(fn);
    *st->batch_ptail = m;
    st->batch_ptail = &m->next;
    st->mbox.postMessages(xy.second->batch_head, xy.second->batch_ptail);
    st->batch_head = nullptr;
    st->batch_ptail = &xy.second->batch_head;
    n += 1;
  }
  return n;
}

void VirtualThreads::workerMain(State *st) {
  current_vtid = st->vtid;
  while(!worker_killed) {
    st->mbox.poll(/*wait=*/true);
  }
}

////////////////////////////////////////////////////////////////////////////////
// NCCL groups mess with stream ordering idioms since all of the stream enqueues
// are deferred until ncclGroupEnd(). This presents a dilemma to code that
// doesn't know if its inside a group or not. By wrapping the group API we give
// a nice way of stating that certain work should also be deferred until after
// the group is ended.

thread_local int inside_nccl_group = 0;
thread_local uint32_t op_id = 0;
thread_local uint32_t work_fifo_idx = 0;
thread_local std::vector<std::function<void()>> after_group_fns;

void wrap_ncclGroupStart(size_t line_number) {
  ncclGroupStart();
  inside_nccl_group++;

  if (opt_verbose) {
    fprintf(stderr, "[%u] ncclGroupStart() line_number=%zu inside_nccl_group=%d .\n", getpid(), line_number, inside_nccl_group);
  }
}

void wrap_ncclGroupEnd(size_t line_number) {
  ncclGroupEnd();
  if (opt_verbose) fprintf(stderr, "[%u] wrap_ncclGroupEnd() line_number=%zu inside_nccl_group=%d\n", getpid(), line_number, inside_nccl_group - 1);
  if(0 == --inside_nccl_group) {
    if (opt_verbose) fprintf(stderr, "[%u] ncclGroupEnd() line_number=%zu\n", getpid(), line_number);
    for(auto &fn: after_group_fns) fn();
    after_group_fns.clear();
  } else if (inside_nccl_group < 0) {
    fprintf(stderr, "[%u] ncclGroupEnd() was invoked without an associated ncclGroupStart(). line_number=%zu. Make sure there is a new line between every collective trace (this sometimes happens when catting two logs together.)\n", getpid(), line_number);
    std::terminate();
  }
}

// Run the given callable immediately if not in a group, otherwise defer
// until wrap_ncclGroupEnd().
template<typename Fn>
void after_ncclGroup(Fn &&fn) {
  if(inside_nccl_group)
    after_group_fns.push_back(static_cast<Fn&&>(fn));
  else
    fn();
}

////////////////////////////////////////////////////////////////////////////////
// Our internal represenation for NCCL API calls. We use at most 2 structs to
// encode a call: the header which holds the common fields, and a second struct
// whose type is determined at runtime by inspecting the header's CallCode.

enum class CallCode {
  get_unique_id,
  comm_rank_init,
  comm_split,
  group_start, group_end,
  redop_create_premulsum, redop_destroy,
  allgather, allreduce, broadcast, reduce, reduce_scatter,
  send, recv,
  count
};
struct CallHeader {
  uint64_t vhost;
  int vpid;
  int vtid;
  size_t line_number;
  CallCode code;
};
struct CallGetUniqueId {
  uint64_t vunique;
};
struct CallNcclGroupEnd {
  int group_seq;
};

struct CallNcclGroupStart {
  int group_seq;
};

struct CallCommSplit {
  uint64_t vcomm_src;
  uint64_t vcomm_new;
  int color;
  int key;
  int comm_rank_n;
  int comm_rank_me;
  int vcomm_seq;
};

struct CallCommInitRank {
  uint64_t vcomm;
  uint64_t vunique;
  int device;
  int comm_rank_n;
  int comm_rank_me;
};
struct CallDataOp { // collectives and p2p
  uint64_t sptr, dptr;
  int64_t elt_n;
  ncclDataType_t elt_ty;
  int red_op;
  int root;
  int vcomm_seq;
  int dev_seq;
  uint64_t vcomm, vstream;
};
struct CallRedOpCreatePreMulSum {
  int vredop;
  ncclDataType_t elt_ty;
  int residence;
  int vcomm_seq;
  uint64_t vcomm;
};
struct CallRedOpDestroy {
  int vredop;
  int vcomm_seq;
  uint64_t vcomm;
};

////////////////////////////////////////////////////////////////////////////////

// owned my main thread
Mailbox main_mailbox;
std::list<std::pair<uint64_t, ncclUniqueId>> vuid_submissions;

std::mutex vuid_mutex;
std::condition_variable vuid_cvar;
std::unordered_map<uint64_t, ncclUniqueId> vuid_table; // owned by vuid_mutex

std::atomic<int> failure{0};

////////////////////////////////////////////////////////////////////////////////
// overloads dispatching to how worker threads invoke NCCL API.

void invokeCall(CallHeader const &hdr, CallGetUniqueId const &body) {
  ncclUniqueId uid;
  NCCL_CHECK(ncclGetUniqueId(&uid));
  main_mailbox.post([=]() {
    if (opt_verbose) {
      fprintf(stderr, "[%u] CallGetUniqueId src_line_number=%lu vunique=0x%lx\n",
        getpid(), hdr.line_number, body.vunique);
    }

    vuid_submissions.push_back({body.vunique, uid});
  });
}

void invokeCall(CallHeader const &hdr, CallCommInitRank const &body) {
  ncclUniqueId uid;
  {
    std::unique_lock<std::mutex> locked(vuid_mutex);
    vuid_cvar.wait(locked, [&]()->bool {
      if(vuid_table.count(body.vunique) != 0) {
        uid = vuid_table[body.vunique];

        if (opt_verbose) {
          fprintf(stderr, "[%u] CallCommInitRank found UniqueId vunique=0x%lx src_line_number=%lu comm_rank_n=%u comm_rank_me=%u vcomm=0x%lx\n",
            getpid(), body.vunique, hdr.line_number, body.comm_rank_n, body.comm_rank_me, body.vcomm);
        }

        return true;
      }

      if (opt_verbose) {
        fprintf(stderr, "[%u] CallCommInitRank waiting for UniqueId src_line_number=%lu comm_rank_n=%u comm_rank_me=%u vcomm=0x%lx\n",
          getpid(), hdr.line_number, body.comm_rank_n, body.comm_rank_me, body.vcomm);
      }

      return false;
    });
  }

  uint32_t my_op_id = ++op_id;
  if (opt_verbose)
  {
    fprintf(stderr, "[%u] id=%u CallCommInitRank src_line_number=%lu comm_rank_n=%u comm_rank_me=%u vcomm=0x%lx\n",
      getpid(), my_op_id, hdr.line_number, body.comm_rank_n, body.comm_rank_me, body.vcomm);
  }

  VirtualComm *vc = &vcomm_table[body.vcomm];
  CUDA_CHECK(cudaSetDevice(body.device));
  NCCL_CHECK(ncclCommInitRank(&(vc->comm), body.comm_rank_n, uid, body.comm_rank_me));

  if (opt_verbose)
  {
    fprintf(stderr, "[%u] id=%u CallCommInitRank completed src_line_number=%lu comm_rank_n=%u comm_rank_me=%u vcomm=0x%lx\n",
      getpid(), my_op_id, hdr.line_number, body.comm_rank_n, body.comm_rank_me, body.vcomm);
  }

  // things to do only after ncclGroupEnd()
  // Invoke lock() to bump the sequence number - this will allow all other threads to make forward progress
  after_ncclGroup([=]() {
    std::unique_lock<std::mutex> locked = vc->sequence.lock(0);
    vc->vunique = body.vunique;
    vc->device = body.device;
    vc->nccl_ops_in_flight = 0;
    vc->p2p_send_seqs.reset(new uint64_t[body.comm_rank_n]{/*0...*/});
    vc->p2p_recv_seqs.reset(new uint64_t[body.comm_rank_n]{/*0...*/});
  });
}

void invokeCall(CallHeader const &hdr, CallCommSplit const &body) {
  // Wait for my sequence in the src communicator's flow
  // This can't be invoked until the corresponding commInitRank has completed
  VirtualComm *vc = &vcomm_table[body.vcomm_src];
  std::unique_lock<std::mutex> locked = vc->sequence.lock(body.vcomm_seq);

  uint32_t my_op_id = ++op_id;
  if (opt_verbose)
  {
    fprintf(stderr, "[%u] id=%u CallCommSplit src_line_number=%lu vcomm_src=0x%lx vcomm_new=0x%lx color=%d key=%d comm_rank_n=%u comm_rank_me=%u \n",
      getpid(), my_op_id, hdr.line_number, body.vcomm_src, body.vcomm_new, body.color, body.key, body.comm_rank_n, body.comm_rank_me);
  }

  VirtualComm *vc_new = &vcomm_table[body.vcomm_new];
  VirtualComm *vc_src = &vcomm_table[body.vcomm_src];
  /* Config is NULL for now*/
  CUDA_CHECK(cudaSetDevice(vc_src->device));
  NCCL_CHECK(ncclCommSplit(vc_src->comm, body.color, body.key, &(vc_new->comm), NULL));

  if (opt_verbose)
  {
    fprintf(stderr, "[%u] id=%u CallCommSplit completed src_line_number=%lu vcomm_src=0x%lx vcomm_new=0x%lx color=%d key=%d comm_rank_n=%u comm_rank_me=%u \n",
      getpid(), my_op_id, hdr.line_number, body.vcomm_src, body.vcomm_new, body.color, body.key, body.comm_rank_n, body.comm_rank_me);
  }

  // things to do only after ncclGroupEnd()
  // Invoke lock() to bump the sequence number - this will allow all other threads to make forward progress
  after_ncclGroup([=]() {
    std::unique_lock<std::mutex> locked = vc_new->sequence.lock(0);
    vc_new->vunique = vc_src->vunique;
    vc_new->device = vc_src->device;
    vc_new->nccl_ops_in_flight = 0;
    vc_new->p2p_send_seqs.reset(new uint64_t[body.comm_rank_n]{/*0...*/});
    vc_new->p2p_recv_seqs.reset(new uint64_t[body.comm_rank_n]{/*0...*/});

    // Resize the src vc for a larger communication domain
    // TODO - Synchronization? Copying?
    // vc_src->p2p_recv_seqs.reset(new uint64_t[body.comm_rank_n]{/*0...*/});
    // vc_src->p2p_recv_seqs.reset(new uint64_t[body.comm_rank_n]{/*0...*/});
  });
}


void invokeCall(CallHeader const &hdr, CallNcclGroupStart const &body) {
  if (opt_verbose) fprintf(stderr, "[%u] Waiting to unlock groupStart for mpi_rank_me=%d group_seq=%d\n", getpid(), mpi_rank_me, body.group_seq);
  std::unique_lock<std::mutex> vc_locked = rank_group_seq_table[mpi_rank_me].lock(body.group_seq);
  wrap_ncclGroupStart(hdr.line_number);
}

void invokeCall(CallHeader const &hdr, CallNcclGroupEnd const &body) {
  if (opt_verbose) fprintf(stderr, "[%u] Waiting to unlock groupEnd for mpi_rank_me=%d group_seq=%d\n", getpid(), mpi_rank_me, body.group_seq);
  std::unique_lock<std::mutex> vc_locked = rank_group_seq_table[mpi_rank_me].lock(body.group_seq);
  wrap_ncclGroupEnd(hdr.line_number);
}


void invokeCall(CallHeader const &hdr, CallRedOpCreatePreMulSum const &body) {
  VirtualComm *vc = &vcomm_table[body.vcomm];
  std::unique_lock<std::mutex> locked = vc->sequence.lock(body.vcomm_seq);

  int rank_me;
  NCCL_CHECK(ncclCommUserRank(vc->comm, &rank_me));

  ScalarsStruct host_scalars;
  host_scalars.i8     = ncclVerifiablePremulScalar<int8_t>(rank_me);
  host_scalars.u8     = ncclVerifiablePremulScalar<uint8_t>(rank_me);
  host_scalars.i32    = ncclVerifiablePremulScalar<int32_t>(rank_me);
  host_scalars.u32    = ncclVerifiablePremulScalar<uint32_t>(rank_me);
  host_scalars.i64    = ncclVerifiablePremulScalar<int64_t>(rank_me);
  host_scalars.u64    = ncclVerifiablePremulScalar<uint64_t>(rank_me);
  host_scalars.f16    = ncclVerifiablePremulScalar<half>(rank_me);
  #ifdef __CUDA_BF16_TYPES_EXIST__
    host_scalars.bf16 = ncclVerifiablePremulScalar<__nv_bfloat16>(rank_me);
  #endif
  host_scalars.f32    = ncclVerifiablePremulScalar<float>(rank_me);
  host_scalars.f64    = ncclVerifiablePremulScalar<double>(rank_me);

  if(vc->device_scalars == nullptr) {
    cudaSetDevice(vc->device);
    void *p;
    CUDA_CHECK(cudaMalloc(&p, sizeof(ScalarsStruct)));
    vc->device_scalars = (ScalarsStruct*)p;
    CUDA_CHECK(cudaMemcpy(p, &host_scalars, sizeof(host_scalars), cudaMemcpyHostToDevice));
  }

  int scalar_offset;
  switch(body.elt_ty) {
    case ncclInt8: scalar_offset = offsetof(ScalarsStruct, i8); break;
    case ncclUint8: scalar_offset = offsetof(ScalarsStruct, u8); break;
    case ncclInt32: scalar_offset = offsetof(ScalarsStruct, i32); break;
    case ncclUint32: scalar_offset = offsetof(ScalarsStruct, u32); break;
    case ncclInt64: scalar_offset = offsetof(ScalarsStruct, i64); break;
    case ncclUint64: scalar_offset = offsetof(ScalarsStruct, u64); break;
    case ncclFloat16: scalar_offset = offsetof(ScalarsStruct, f16); break;
    #ifdef __CUDA_BF16_TYPES_EXIST__
    case ncclBfloat16: scalar_offset = offsetof(ScalarsStruct, bf16); break;
    #endif
    case ncclFloat32: scalar_offset = offsetof(ScalarsStruct, f32); break;
    case ncclFloat64: scalar_offset = offsetof(ScalarsStruct, f64); break;
    default: assert(0);
  }

  char *scalar = body.residence == ncclScalarHostImmediate
    ? (char*)&host_scalars
    : (char*)vc->device_scalars;
  scalar += scalar_offset;

  ncclRedOp_t op;
  NCCL_CHECK(ncclRedOpCreatePreMulSum(&op, scalar, (ncclDataType_t)body.elt_ty, (ncclScalarResidence_t)body.residence, vc->comm));
  vc->vredops[body.vredop] = op;

  uint32_t my_op_id = ++op_id;
  if (opt_verbose)
  {
    fprintf(stderr, "[%u] id=%u invokeCall: Posted ncclRedOpCreatePreMulSum. ops_in_flight=%u src_line_number=%lu vcomm=0x%lx vcomm_seq=%d\n",
      getpid(), my_op_id, vc->nccl_ops_in_flight.load(std::memory_order_relaxed), hdr.line_number, body.vcomm, body.vcomm_seq);
  }
}

void invokeCall(CallHeader const &hdr, CallRedOpDestroy const &body) {
  VirtualComm *vc = &vcomm_table[body.vcomm];
  std::unique_lock<std::mutex> locked = vc->sequence.lock(body.vcomm_seq);

  ncclRedOp_t red_op = vc->vredops.at(body.vredop);
  NCCL_CHECK(ncclRedOpDestroy(vc->vredops.at(body.vredop), vc->comm));
  vc->vredops.erase(body.vredop);

  uint32_t my_op_id = ++op_id;
  if (opt_verbose)
  {
    fprintf(stdout, "[%u] id=%u invokeCall: Posted ncclRedOpDestroy. ops_in_flight=%u src_line_number=%lu vcomm=0x%lx vcomm_seq=%d\n",
      getpid(), my_op_id, vc->nccl_ops_in_flight.load(std::memory_order_relaxed), hdr.line_number, body.vcomm, body.vcomm_seq);
  }
}

void invokeCall(CallHeader const &hdr, CallDataOp const &body) {
  VirtualComm *vc = &vcomm_table[body.vcomm];
  std::unique_lock<std::mutex> vc_locked = vc->sequence.lock(body.vcomm_seq);
  std::unique_lock<std::mutex> dev_locked = CudaHelp::sequence_lock(vc->device, body.dev_seq, body.vcomm);

  int device = vc->device;
  int rank_n, rank_me;

  uint32_t my_op_id = ++op_id;
  uint32_t fifo_idx = ++work_fifo_idx % 2048; // NCCL_MAX_OPS

  NCCL_CHECK(ncclCommCount(vc->comm, &rank_n));
  NCCL_CHECK(ncclCommUserRank(vc->comm, &rank_me));
  CUDA_CHECK(cudaSetDevice(device));

  uint64_t seed;
  int verify_rank_n;
  intptr_t verify_elt_n;
  intptr_t verify_elt_ix0;

  void* sptr = nullptr;
  void* dptr = nullptr;

  char const *call_name = "???";

  ncclRedOp_t red_op;
  switch(hdr.code) {
  case CallCode::allreduce:
  case CallCode::reduce:
  case CallCode::reduce_scatter:
    red_op = body.red_op < (int)ncclNumOps
      ? (ncclRedOp_t)body.red_op
      : vc->vredops.at(body.red_op);
    break;
  default:
    red_op = (ncclRedOp_t)body.red_op;
    break;
  }

  size_t elt_sz = elementSize(body.elt_ty);
  cudaStream_t stream_prepare = CudaHelp::pickAnonStream(device);
  cudaStream_t stream_nccl = CudaHelp::pickVirtualStream(body.vstream, device);

  switch(hdr.code) {
  case CallCode::allgather:
    call_name = "AllGather";
    seed = hashOf(vc->vunique, vc->coll_seq++);

    sptr = CudaHelp::allocate(device, body.elt_n*elt_sz, body.sptr, my_op_id);
    dptr = CudaHelp::allocate(device, rank_n*body.elt_n*elt_sz, body.dptr, my_op_id);

    ncclVerifiablePrepareInput(
     sptr, body.elt_n, body.elt_ty, red_op, /*rank_n=*/1, /*rank_me=*/0,
     seed, /*elt_ix0=*/ rank_me*body.elt_n, stream_prepare
    );
    CudaHelp::streamWaitOnStream(device, stream_nccl, device, stream_prepare);
    NCCL_CHECK(ncclAllGather(sptr, dptr, body.elt_n, (ncclDataType_t)body.elt_ty, vc->comm, stream_nccl));
    verify_elt_ix0 = 0;
    verify_elt_n = body.elt_n * rank_n;
    verify_rank_n = 1;
    break;
  case CallCode::allreduce:
    call_name = "AllReduce";
    seed = hashOf(vc->vunique, vc->coll_seq++);

    sptr = CudaHelp::allocate(device, body.elt_n*elt_sz, body.sptr, my_op_id);
    dptr = CudaHelp::allocate(device, body.elt_n*elt_sz, body.dptr, my_op_id);

    ncclVerifiablePrepareInput(sptr, body.elt_n, body.elt_ty, red_op, rank_n, rank_me, seed, /*elt_ix0=*/0, stream_prepare);
    CudaHelp::streamWaitOnStream(device, stream_nccl, device, stream_prepare);
    NCCL_CHECK(ncclAllReduce(sptr, dptr, body.elt_n, (ncclDataType_t)body.elt_ty, red_op, vc->comm, stream_nccl));
    verify_elt_ix0 = 0;
    verify_elt_n = body.elt_n;
    verify_rank_n = rank_n;
    break;
  case CallCode::broadcast:
    call_name = "Broadcast";
    seed = hashOf(vc->vunique, vc->coll_seq++);
    if(rank_me == body.root) {
      sptr = CudaHelp::allocate(device, body.elt_n*elt_sz, body.sptr, my_op_id);
      ncclVerifiablePrepareInput(sptr, body.elt_n, body.elt_ty, red_op, /*rank_n=*/1, /*rank_me=*/0, seed, /*elt_ix0=*/0, stream_prepare);
      CudaHelp::streamWaitOnStream(device, stream_nccl, device, stream_prepare);
    }

    dptr = CudaHelp::allocate(device, body.elt_n*elt_sz, body.dptr, my_op_id);

    NCCL_CHECK(ncclBroadcast(sptr, dptr, body.elt_n, (ncclDataType_t)body.elt_ty, body.root, vc->comm, stream_nccl));
    verify_elt_ix0 = 0;
    verify_elt_n = rank_me == body.root ? 0 : body.elt_n;
    verify_rank_n = 1;
    break;
  case CallCode::reduce:
    call_name = "Reduce";
    seed = hashOf(vc->vunique, vc->coll_seq++);

    sptr = CudaHelp::allocate(device, body.elt_n*elt_sz, body.sptr, my_op_id);
    dptr = CudaHelp::allocate(device, body.elt_n*elt_sz, body.dptr, my_op_id);

    ncclVerifiablePrepareInput(sptr, body.elt_n, body.elt_ty, red_op, rank_n, rank_me, seed, /*elt_ix=*/0, stream_prepare);
    CudaHelp::streamWaitOnStream(device, stream_nccl, device, stream_prepare);
    NCCL_CHECK(ncclReduce(sptr, dptr, body.elt_n, (ncclDataType_t)body.elt_ty, red_op, body.root, vc->comm, stream_nccl));
    verify_elt_ix0 = 0;
    verify_elt_n = rank_me == body.root ? body.elt_n : 0;
    verify_rank_n = rank_n;
    break;
  case CallCode::reduce_scatter:
    call_name = "ReduceScatter";
    seed = hashOf(vc->vunique, vc->coll_seq++);

    sptr = CudaHelp::allocate(device, rank_n*body.elt_n*elt_sz, body.sptr, my_op_id);
    dptr = CudaHelp::allocate(device, body.elt_n*elt_sz, body.dptr, my_op_id);

    ncclVerifiablePrepareInput(sptr, rank_n*body.elt_n, body.elt_ty, red_op, rank_n, rank_me, seed, /*elt_ix0=*/0, stream_prepare);
    CudaHelp::streamWaitOnStream(device, stream_nccl, device, stream_prepare);
    NCCL_CHECK(ncclReduceScatter(
      sptr,
      dptr,
      body.elt_n,
      (ncclDataType_t)body.elt_ty,
      red_op,
      vc->comm,
      stream_nccl)
      );

    verify_elt_ix0 = rank_me*body.elt_n;
    verify_elt_n = body.elt_n;
    verify_rank_n = rank_n;
    break;
  case CallCode::send:
    call_name = "Send";
    seed = vc->vunique ^ hashOf(rank_me, body.root);

    sptr = CudaHelp::allocate(device, body.elt_n*elt_sz, body.sptr, my_op_id);

    ncclVerifiablePrepareInput(
     sptr, body.elt_n, body.elt_ty, red_op, /*rank_n=*/1, /*rank_me=*/0,
     seed, /*elt_ix0=*/vc->p2p_send_seqs[body.root], stream_prepare
    );
    CudaHelp::streamWaitOnStream(device, stream_nccl, device, stream_prepare);
    NCCL_CHECK(ncclSend(
      sptr,
      body.elt_n,
      (ncclDataType_t)body.elt_ty, body.root, vc->comm, stream_nccl));
    verify_elt_n = 0; // no validation on send side
    verify_elt_ix0 = vc->p2p_send_seqs[body.root];
    vc->p2p_send_seqs[body.root] += body.elt_n;
    break;
  case CallCode::recv:
    call_name = "Recv";
    seed = vc->vunique ^ hashOf(body.root, rank_me);
    dptr = CudaHelp::allocate(device, body.elt_n*elt_sz, body.dptr, my_op_id);

    NCCL_CHECK(ncclRecv(
      dptr,
      body.elt_n,
      (ncclDataType_t)body.elt_ty, body.root, vc->comm, stream_nccl));
    verify_elt_n = body.elt_n;
    verify_elt_ix0 = vc->p2p_recv_seqs[body.root];
    vc->p2p_recv_seqs[body.root] += body.elt_n;
    verify_rank_n = 1;
    break;
  default:
    assert(0);
  }

  if (opt_verbose)
  {
    fprintf(stderr, "[%u] id=%u fifo_idx=%u invokeCall: Posted nccl%s. device=%d ops_in_flight=%u src_line_number=%lu vcomm=0x%lx vcomm_seq=%d verify_elt_ix0=%lu\n",
      getpid(), my_op_id, fifo_idx, call_name, device, vc->nccl_ops_in_flight.load(std::memory_order_relaxed), hdr.line_number, body.vcomm, body.vcomm_seq, verify_elt_ix0);
  }

  // Store a pointer to this vcomm nccl_ops_in_flight for bookkeeping
  std::atomic<int> *my_ops = &vc->nccl_ops_in_flight;

  // things to do only after ncclGroupEnd()
  after_ncclGroup([=]() {
    my_ops->fetch_add(1);
    global_ops_submitted.fetch_add(1);

    if (opt_verbose)
    {
      fprintf(stderr, "[%u] id=%u fifo_idx=%u nccl%s after_ncclGroup 1. ops_in_flight=%u\n",
        getpid(), my_op_id, fifo_idx, call_name, my_ops->load(std::memory_order_relaxed));
    }

    // Cleanup
    CudaHelp::streamCallback(device, stream_nccl, [=]() {
      my_ops->fetch_add(-1);
      global_ops_completed.fetch_add(1);

      if (opt_verbose)
      {
        fprintf(stderr, "[%u] nccl%s after_ncclGroup 2 id=%u fifo_idx=%u ops_in_flight=%u sptr=0x%lx dptr=0x%lx\n",
          getpid(), call_name, my_op_id, fifo_idx, my_ops->load(std::memory_order_relaxed), (intptr_t) sptr, (intptr_t) dptr);
      }

      if(sptr != nullptr) {
        CudaHelp::deallocate(device, sptr, my_op_id);

          if (opt_verbose)
          {
            fprintf(stderr, "[%u] nccl%s after_ncclGroup 3 id=%u fifo_idx=%u. deallocated sptr\n",
              getpid(), call_name, my_op_id, fifo_idx);
          }
        }
    });

    if(dptr != nullptr) {
      cudaStream_t stream_verify = CudaHelp::pickAnonStream(device);

      if (opt_verbose)
      {
        fprintf(stderr, "[%u] nccl%s after_ncclGroup 4 id=%u. Waiting on stream_nccl=%p\n", getpid(), call_name, my_op_id, stream_nccl);
      }
      CudaHelp::streamWaitOnStream(device, stream_verify, device, stream_nccl);

      int64_t *bad_elt_n = the_host_pool.allocate<int64_t>();

      ncclVerifiableVerify(
        dptr, nullptr, verify_elt_n, body.elt_ty, red_op,
        verify_rank_n, seed, verify_elt_ix0, bad_elt_n, stream_verify
      );

      CudaHelp::streamCallback(device, stream_verify, [=]() {
        int64_t bads = *bad_elt_n;
        the_host_pool.deallocate(bad_elt_n);

        if (opt_verbose)
        {
          fprintf(stderr, "[%u] after_ncclGroup 5 id=%u - Verifiable stream completed\n", getpid(), my_op_id);
        }

        if(bads != 0) {
          std::fprintf(stderr,
            "[%d/%d] Corrupt elements %.2f%%\n"
            " call: nccl%s(elts=%p, eltn=%lld, type=%d, op=%d, root=%d)\n"
            " seed: 0x%lx verify_elt_ix0=%lu\n",
            rank_me, rank_n, 100*double(bads)/verify_elt_n,
            call_name, (void*)dptr, (long long)verify_elt_n, body.elt_ty, (int)red_op, body.root, (long)seed, verify_elt_ix0
          );
          failure.store(1);
        }

        CudaHelp::deallocate(device, dptr, my_op_id);

        if (opt_verbose)
        {
          fprintf(stderr, "[%u] after_ncclGroup 6 id=%u. Deallocated dptr\n", getpid(), my_op_id);
        }
      });
    }
  });
}

void playTrace(ByteBuffer& trace, std::chrono::duration<double>* duration) {
  // Used to measure runtime after dissemination
  std::chrono::time_point<std::chrono::steady_clock> start;
  int vuid_countdown = 0;

  // Counter of data op collectives
  data_op_call_count = 0;

  { // First we do a pass over the calls to establish a sequential access order
    // to comms for our threads to obey.
    ByteBuffer::Cursor cur(trace);
    struct VCommMeta {
      int rank_n, rank_me;
      int device;
      int seq_bumper=1;
    };
    std::unordered_map<uint64_t, VCommMeta> vcomm_to_meta;

    while(cur.has_more()) {
      CallHeader const &hdr = cur.template pop<CallHeader>();
      switch(hdr.code) {
      case CallCode::get_unique_id:
        cur.template pop<CallGetUniqueId>();
        vuid_countdown += 1;
        break;
      case CallCode::comm_rank_init:
        { auto body = cur.template pop<CallCommInitRank>();
          vcomm_to_meta[body.vcomm].device = body.device;
          vcomm_to_meta[body.vcomm].rank_n = body.comm_rank_n;
          vcomm_to_meta[body.vcomm].rank_me = body.comm_rank_me;
        } break;
      case CallCode::comm_split:
        { auto &body = cur.template pop<CallCommSplit>();
          // Get the device from the src vcomm
          vcomm_to_meta[body.vcomm_new].device = vcomm_to_meta[body.vcomm_src].device;
          vcomm_to_meta[body.vcomm_new].rank_n = body.comm_rank_n;
          vcomm_to_meta[body.vcomm_new].rank_me = body.comm_rank_me;
          // Set the src vcomm rank_n to the new post-split value
          vcomm_to_meta[body.vcomm_src].rank_n = body.comm_rank_n;

          // I don't get to call CommSplit until CommRankInit and all other operations have been invoked
          body.vcomm_seq = vcomm_to_meta[body.vcomm_src].seq_bumper++;
        } break;
      case CallCode::group_start:
        { auto &body = cur.template pop<CallNcclGroupStart>();
        } break;
      case CallCode::group_end:
        { auto &body = cur.template pop<CallNcclGroupEnd>();
        } break;
      case CallCode::redop_create_premulsum:
        { auto &body = cur.template pop<CallRedOpCreatePreMulSum>();
          body.vcomm_seq = vcomm_to_meta[body.vcomm].seq_bumper++;
        } break;
      case CallCode::redop_destroy:
        { auto &body = cur.template pop<CallRedOpDestroy>();
          body.vcomm_seq = vcomm_to_meta[body.vcomm].seq_bumper++;
        } break;
      case CallCode::allgather:
      case CallCode::allreduce:
      case CallCode::broadcast:
      case CallCode::reduce:
      case CallCode::reduce_scatter:
      case CallCode::send:
      case CallCode::recv:
        { CallDataOp &body = cur.template pop<CallDataOp>();
          body.vcomm_seq = vcomm_to_meta[body.vcomm].seq_bumper++;
          data_op_call_count++;
        } break;
      default:
        assert(0);
      }

    }
  }

  VirtualThreads vthreads;
  ByteBuffer::Cursor cur(trace);

  // This is a quick way of seeing how much forward progress is being made
  if (opt_progress_thread && mpi_rank_me == 0) {
    progress_thread = std::thread(&progressThread, std::move(signal_exit.get_future()));
  }

  while(true) {
    int batch = 512;
    while(batch-- && cur.has_more()) {
      CallHeader const &hdr = cur.template pop<CallHeader>();
      switch(hdr.code) {
      case CallCode::get_unique_id:
        { CallGetUniqueId const &body = cur.template pop<CallGetUniqueId>();
          vthreads.postBatch(hdr.vtid, [=]() { invokeCall(hdr, body); });
        } break;
      case CallCode::comm_rank_init:
        { CallCommInitRank const &body = cur.template pop<CallCommInitRank>();
          vthreads.postBatch(hdr.vtid, [=]() { invokeCall(hdr, body); });
        } break;
      case CallCode::comm_split:
        { CallCommSplit const &body = cur.template pop<CallCommSplit>();
          vthreads.postBatch(hdr.vtid, [=]() { invokeCall(hdr, body); });
        } break;
      case CallCode::group_start:
        { CallNcclGroupStart const &body = cur.template pop<CallNcclGroupStart>();
          vthreads.postBatch(hdr.vtid, [=]() { invokeCall(hdr, body); });
        } break;
      case CallCode::group_end:
        { CallNcclGroupEnd const &body = cur.template pop<CallNcclGroupEnd>();
          vthreads.postBatch(hdr.vtid, [=]() { invokeCall(hdr, body); });
        } break;
      case CallCode::redop_create_premulsum:
        { CallRedOpCreatePreMulSum const &body = cur.template pop<CallRedOpCreatePreMulSum>();
          vthreads.postBatch(hdr.vtid, [=]() { invokeCall(hdr, body); });
        } break;
      case CallCode::redop_destroy:
        { CallRedOpDestroy const &body = cur.template pop<CallRedOpDestroy>();
          vthreads.postBatch(hdr.vtid, [=]() { invokeCall(hdr, body); });
        } break;
      case CallCode::allgather:
      case CallCode::allreduce:
      case CallCode::broadcast:
      case CallCode::reduce:
      case CallCode::reduce_scatter:
      case CallCode::send:
      case CallCode::recv:
        { CallDataOp const &body = cur.template pop<CallDataOp>();
          vthreads.postBatch(hdr.vtid, [=]() { invokeCall(hdr, body); });
        } break;
      }
    }
    vthreads.submitBatch();
    main_mailbox.poll(false);

    // Look for ncclUniqueIds submitted from worker threads to submit to our
    // ongoing allreduce
    std::pair<uint64_t, ncclUniqueId> vuid_in, vuid_out;
    if(vuid_submissions.empty()) {
      // We are ready to stop condition when all ids submitted and we've posted entire trace
      vuid_in.first = vuid_countdown != 0 || cur.has_more() ? 1 : 0;
    }
    else {
      vuid_in = vuid_submissions.front();
      if (opt_verbose) {
        fprintf(stderr, "[%u] vuid_in.first=0x%lx vuid_in.second=%s\n",
          getpid(), vuid_in.first, vuid_in.second.internal);
      }
    }

    /* Every rank submits a (vuid,ncclUniqueId) pair having 3 possible values:
     * 1) (vuid>1) A valid ncclUniqueId and its hash id.
     * 2) (vuid=1) No ncclUniqueId, and rank is not ready to stop.
     * 3) (vuid=0) No ncclUniqueId, and rank is ready to stop.
     *
     * The reduction function returns the max of the two vuid's and selects the
     * ncclUniqueId matching the maximal vuid.
     */
    static MPI_Op vuid_choose_op = MPI_OP_NULL;
    if(vuid_choose_op == MPI_OP_NULL) {
      MPI_User_function *vuid_choose_fn =
        [](void *invec, void *inoutvec, int *len, MPI_Datatype *ty) {
          using P = std::pair<uint64_t, ncclUniqueId>;
          P *y = (P*)inoutvec;
          P *x = (P*)invec;
          int n = *len;
          while(n >= sizeof(P)) {
            if(x->first > y->first)
              *y = *x;
            n -= sizeof(P);
            x++, y++;
          }
        };

      MPI_Op_create(vuid_choose_fn, 1, &vuid_choose_op);
    }

    // The purpose of this allreduce is to serve as a makeshift asynchronous broadcast.
    // When a rank creates a UniqueId, it must be disseminated to any other rank which needs it to create a communicator
    //
    MPI_Allreduce(&vuid_in, &vuid_out, sizeof(vuid_in), MPI_BYTE, vuid_choose_op, MPI_COMM_WORLD);
    // Inspect output of allreduce
    if(vuid_out.first > 1) { // A unique id has been disseminated
      if (opt_verbose) {
        fprintf(stderr, "[%u] vuid_out.first=%lu > 0. A unique id has been disseminated.\n",
          getpid(), vuid_out.first);
      }

      if(vuid_in.first == vuid_out.first) {
        if (opt_verbose) {
          fprintf(stderr, "[%u] vuid_in.first (%lu) == vuid_out.first (%lu). Our submission won, drop from outgoing list.\n",
            getpid(), vuid_in.first, vuid_out.first);
        }

        // Our submission won, we drop it from the outgoing list
        vuid_submissions.pop_front();
        vuid_countdown -= 1;
      }

      // Share unique id with all threads
      std::unique_lock<std::mutex> locked(vuid_mutex);
      vuid_table[vuid_out.first] = vuid_out.second;
      vuid_cvar.notify_all();
    }

    if(vuid_out.first == 0) {
      if (opt_verbose) {
        fprintf(stderr, "[%u] vuid_out.first=0x%lx. Everybody submitted ready-to-stop\n",
          getpid(), vuid_out.first);
      }

      break; // Everybody submitted ready-to-stop
    }
  }

  start = std::chrono::steady_clock::now();

  // Synchronize with all threads by telling them to send us a message
  // to decrement countdown.
  int countdown = 0; // Avoid compiler warning #549-D
  countdown = vthreads.postAll([&]() {
    main_mailbox.post([&]() {
      countdown -= 1;
    });
  });
  while (countdown != 0)
    main_mailbox.poll(/*wait=*/true);

  // synchronize all streams
  CudaHelp::synchronize();
  *duration = std::chrono::steady_clock::now() - start;
}

// Helper to translate callcodes to strings
std::string callCodeToString(CallCode code) {
  if (code == CallCode::allgather)
    return "AllGather";
  if (code == CallCode::allreduce)
    return "AllReduce";
  if (code == CallCode::broadcast)
    return "Broadcast";
  if (code == CallCode::reduce)
    return "Reduce";
  if (code == CallCode::reduce_scatter)
    return "ReduceScatter";

  return "Unknown CallCode";
}

//
// Types needed for pre-checking the trace
//
typedef std::vector<std::pair<CallHeader, CallDataOp>> DataOpsVector;
struct GlobalCommMap {
  uint64_t vunique;
  // Map of data ops by rank
  std::unordered_map<uint32_t, DataOpsVector> data_ops_map;
  // Map of custom redops by rank
  std::unordered_map<uint32_t, std::unordered_set<uint64_t>> vredops;
};

void preCheck(std::unordered_map<uint64_t, std::shared_ptr<GlobalCommMap>>& vuniqueToGlobalMap) {
  // 1. For each vunique (vcomm group in the original trace)
  auto globalMapIt = vuniqueToGlobalMap.begin();
  while (globalMapIt != vuniqueToGlobalMap.end()) {
    std::shared_ptr<GlobalCommMap> globalMap = globalMapIt->second;

    // Skip checking if this communicator group only has 1 rank
    if (globalMap->data_ops_map.size() <= 1) {
      ++globalMapIt;
      continue;
    }

    // 2. Create a list of iterators for each rank's data ops
    std::vector<DataOpsVector::iterator> data_ops_it_vector;
    auto it = globalMap->data_ops_map.begin();
    size_t collective_count = it->second.size();
    uint32_t first_rank = it->first;

    // If there are 0 collectives, don't go through with any of this
    if (collective_count == 0) {
      ++globalMapIt;
      continue;
    }

    while (it != globalMap->data_ops_map.end()) {
      if (it->second.size() != collective_count) {
        fprintf(stderr, "TRACE PREPROCESSING ERROR - Missing collectives. rank %u has %zu collectives. rank %u has %zu collectives.\n",
          first_rank, collective_count, it->first, it->second.size());
          std::terminate();
      }

      data_ops_it_vector.push_back(it->second.begin());
      ++it;
    }

    // 3. Inspect the next sequential data op in each iterator and compare it
    // Terminate we've exhausted our baseline rank
    it = globalMap->data_ops_map.begin();
    while (data_ops_it_vector[0] != it->second.end()) {
      // Take the first data op as the baseline comparison
      std::pair<CallHeader, CallDataOp> baselineDataOp = *data_ops_it_vector[0];
      ++data_ops_it_vector[0];
      ++it;

      CallHeader& hdr = baselineDataOp.first;
      CallDataOp& op = baselineDataOp.second;

      for (size_t i = 1; i < data_ops_it_vector.size(); i++)
      {
        // If we haven't run off the edge (we don't expect to)
        if (data_ops_it_vector[i] != it->second.end()) {
          std::pair<CallHeader, CallDataOp> comparisonDataOp = *data_ops_it_vector[i];
          uint32_t temp_rank = it->first;
          ++data_ops_it_vector[i];
          ++it;

          CallHeader& temp_hdr = comparisonDataOp.first;
          CallDataOp& temp_op = comparisonDataOp.second;

          // Check for PreMulSum vredops
          // If they don't match but both are stored as vredops, then consider them matching
          // TODO on stricter validation here
          bool matching_redops = op.red_op == temp_op.red_op;
          if (!matching_redops) {
            auto op1 = globalMap->vredops[first_rank].find(op.red_op);
            auto op2 = globalMap->vredops[temp_rank].find(temp_op.red_op);
            matching_redops = op1 != globalMap->vredops[first_rank].end() &&
                                op2 != globalMap->vredops[temp_rank].end();
            if (!matching_redops) {
              fprintf(stderr, "WARNING - Found non-matching redops without vredop mappings\n");
              fprintf(stderr, "op1=%u rank=%u was_mapped=%u\n",
                op.red_op, first_rank, op1 != globalMap->vredops[first_rank].end());
              fprintf(stderr, "op2=%u rank=%u was_mapped=%u\n",
                temp_op.red_op, temp_rank, op2 != globalMap->vredops[temp_rank].end());
            }
          }

          if (op.elt_n != temp_op.elt_n ||
              op.elt_ty != temp_op.elt_ty ||
              op.root != temp_op.root ||
              !matching_redops ||
              hdr.code != temp_hdr.code) {

            // Human-readable coll_names
            std::string coll_name = callCodeToString(hdr.code);
            std::string temp_coll_name = callCodeToString(temp_hdr.code);

            fprintf(stderr, "TRACE PREPROCESSING ERROR - Mismatch\n");
            fprintf(stderr, "Collective 1:\n");
            fprintf(stderr, "  nccl%s. i=%zu rank=%u src_line_number=%lu elt_n=%lu elt_ty=%u red_op=%u root=%u code=%u\n",
                coll_name.c_str(), 0L, first_rank, hdr.line_number, op.elt_n, op.elt_ty, op.red_op, op.root, (uint32_t) hdr.code);

            fprintf(stderr, "Collective 2:\n");
            fprintf(stderr, "  nccl%s. i=%zu rank=%u src_line_number=%lu elt_n=%lu elt_ty=%u red_op=%u root=%u code=%u\n",
                temp_coll_name.c_str(), i, temp_rank, temp_hdr.line_number, temp_op.elt_n, temp_op.elt_ty, temp_op.red_op, temp_op.root, (uint32_t) temp_hdr.code);
            std::terminate();
          }
        } else {
            fprintf(stderr, "TRACE PREPROCESSING ERROR - Missing collectives. Ran out of data ops entries\n");
            std::string coll_name = callCodeToString(hdr.code);
            fprintf(stderr, "Collective 1:\n");
            fprintf(stderr, "  nccl%s. i=%zu src_line_number=%lu elt_n=%lu elt_ty=%u red_op=%u root=%u code=%u\n",
                coll_name.c_str(), 0L, hdr.line_number, op.elt_n, op.elt_ty, op.red_op, op.root, (uint32_t) hdr.code);
            fprintf(stderr, "Expected %zu more nccl%s collectives.\n", globalMap->data_ops_map.size() - i, coll_name.c_str());
            std::terminate();
        }
      }

      // Reset this iterator to check running over bounds
      it = globalMap->data_ops_map.begin();
    }

    ++globalMapIt;
  }
}

////////////////////////////////////////////////////////////////////////////////
// To correctly preparse the following conditions:
// For each communicator group (vunique), confirm the following:
// Each collective is called in the same order with the same high level args
// Each send is matched by a recv

ByteBuffer loadDebugCallTrace(std::string const &path) {
  if(mpi_rank_me == 0) {
    std::unique_ptr<ByteBuffer[]> rank_bufs(new ByteBuffer[mpi_rank_n]);

    // Track the per-rank program order submission of ncclGroupEnd()
    std::vector<int> rank_group_seqs(mpi_rank_n, 0);

    // This is a map for mapping virtual rank to physical rank in a force-fit run
    std::unordered_map<int, int> force_fit_rank_map;

    struct VHostState {
      int phost = -1;
      int vpids = 0;
      std::string vhost_name;
      std::unordered_map<int, int> vpid_to_rank;

      // Store a sequence number of accesses to a device on a host
      // This is relevant if 2 communicators use the same device
      std::unordered_map<int, int> dev_seq_map;
      std::unordered_map<uint64_t, int> vcomm_to_dev;
    };
    std::unordered_map<uint64_t, VHostState> vhosts;

    std::unordered_map<uint64_t, std::shared_ptr<GlobalCommMap>> vuniqueToGlobalMap;
    std::unordered_map<uint64_t, std::shared_ptr<GlobalCommMap>> vcommToGlobalMap;

    std::unique_ptr<int[]> phost_popn(new int[phost_n]{/*zeros*/});

    std::ifstream file;
    std::istream *input;
    if(path != "-") {
      file.open(path);
      if(!file.good()) {
        std::cerr<<"Invalid path to trace file: "<<path<<std::endl;
        std::terminate();
      }
      input = &file;
    }
    else
      input = &std::cin;

    std::string line;
    size_t line_counter = 0;

    // Parse the rest of the file
    while(std::getline(*input, line)) {
      line_counter++;
      char vhost_name[512];
      int vpid, vtid;
      int got = -1;
      char const *line_ptr = line.c_str();
      int ok = std::sscanf(line_ptr, "%[^:]:%d:%d NCCL CALL %n", vhost_name, &vpid, &vtid, &got);
      if(ok != 3 || got == -1 || line_ptr[0] == '#') {
        if (opt_verbose) {
          fprintf(stderr, "Couldn't parse line: %s\n", line_ptr);
        }
        continue;
      }
      line_ptr += got;

      uint64_t vhost = hashOf(vhost_name);
      if (vhosts.find(vhost) == vhosts.end()) {
       if (vhosts.size() == phost_n) {
          // We've ran out of phosts to fit vhosts
          if (opt_force_fit) {
            if (opt_verbose) fprintf(stderr, "Not enough physical hosts (%d) to accommodate virtual host %s. Skipping trace line to force fit.\n",
                                        phost_n, vhost_name);
            continue;
          } else {
            fprintf(stderr, "Not enough physical hosts (%d) to accommodate virtual host %s.\n", phost_n, vhost_name);
            std::cerr << "line " << line_counter << ": " << line << std::endl;
            std::terminate();
          }
        } else {
          // Create new vhost struct
          vhosts[vhost].vhost_name = vhost_name;
          vhosts[vhost].phost      = vhosts.size()-1;
        }
      }

      VHostState &vhost_st = vhosts[vhost];

      int rank;
      if(vhost_st.vpid_to_rank.count(vpid) == 0) {
        if(phost_popn[vhost_st.phost] == phost_ranks[vhost_st.phost].size()) {
          fprintf(stderr, "Not enough physical processes (%d) on host to accommodate virtual processes.\n", (int)phost_ranks[vhost_st.phost].size());
          std::terminate();
        }
        rank = phost_ranks[vhost_st.phost][phost_popn[vhost_st.phost]++];
        vhost_st.vpid_to_rank[vpid] = rank;
      }
      else
        rank = vhost_st.vpid_to_rank[vpid];

      CallHeader hdr;
      hdr.vhost = vhost;
      hdr.vpid = vpid;
      hdr.vtid = vtid;
      hdr.line_number = line_counter;

      { CallGetUniqueId call;
        if(1 == std::sscanf(line_ptr, "ncclGetUniqueId(%" PRIx64 ")", &call.vunique)) {
          hdr.code = CallCode::get_unique_id;
          rank_bufs[rank].append(hdr);
          rank_bufs[rank].append(call);
          continue;
        }
      }

      { CallCommInitRank call;
        if(5 == std::sscanf(line_ptr, "ncclCommInitRank(%" PRIx64 ",%d,%" PRIx64 ",%d,%d)",
            &call.vcomm, &call.comm_rank_n, &call.vunique, &call.comm_rank_me, &call.device)) {
          hdr.code = CallCode::comm_rank_init;

          if (opt_force_fit) {
            if (opt_verbose) {
              fprintf(stderr, "Force fitting virtual_rank=%d to physical_rank=%d",
                rank, call.comm_rank_me);
              std::cerr << "line " << line_counter << ": " << line << std::endl;
            }
            // Map this relationship
            force_fit_rank_map[call.comm_rank_me] = rank;

            call.comm_rank_me = rank;
            call.comm_rank_n  = mpi_rank_n;
          }

          rank_bufs[rank].append(hdr);
          rank_bufs[rank].append(call);

          // Map this vcomm to the global communicator group
          auto it = vuniqueToGlobalMap.find(call.vunique);
          std::shared_ptr<GlobalCommMap> globalCommMap;
          if (it == vuniqueToGlobalMap.end()) {
            globalCommMap = std::make_shared<GlobalCommMap>();
            globalCommMap->vunique = call.vunique;
            // New vunique implies a new global communicator group
            vuniqueToGlobalMap.emplace(call.vunique, globalCommMap);
          } else {
            globalCommMap = it->second;
          }

          // Track our new vcomm
          vcommToGlobalMap.emplace(call.vcomm, globalCommMap);

          // Map this vcomm to this device.
          vhost_st.vcomm_to_dev[call.vcomm] = call.device;
          continue;
        }
      }

      { CallCommSplit call;
        if(6 == std::sscanf(line_ptr, "ncclCommSplit(%" PRIx64 ",%d,%d,%" PRIx64 ",%d,%d)",
                  &call.vcomm_src, &call.color, &call.key, &call.vcomm_new, &call.comm_rank_me, &call.comm_rank_n)) {
          hdr.code = CallCode::comm_split;

          if (opt_force_fit) {
            if (opt_verbose) {
              fprintf(stderr, "Force fitting virtual_rank=%d to physical_rank=%d",
                rank, call.comm_rank_me);
              std::cerr << "line " << line_counter << ": " << line << std::endl;
            }
            // Map this relationship
            force_fit_rank_map[call.comm_rank_me] = rank;

            call.comm_rank_me = rank;
            call.comm_rank_n  = mpi_rank_n;
          }

          rank_bufs[rank].append(hdr);
          rank_bufs[rank].append(call);

          // Map this vcomm to the global communicator group of its src comm
          auto it = vcommToGlobalMap.find(call.vcomm_src);
          std::shared_ptr<GlobalCommMap> globalCommMap;
          if (it == vcommToGlobalMap.end()) {
            fprintf(stderr, "Splitting comm_src=0x%lx which hasn't yet been initialized. line_number=%zu vhost=%lu vpid=%d vtid=%d\n",
              call.vcomm_src, hdr.line_number, hdr.vhost, hdr.vpid, hdr.vtid);
              std::terminate();
          } else {
            globalCommMap = it->second;
          }

          // Track our new vcomm with the vunique of the src comm
          vcommToGlobalMap.emplace(call.vcomm_new, globalCommMap);

          // Map this vcomm to the device of the src comm
          vhost_st.vcomm_to_dev[call.vcomm_new] = vhost_st.vcomm_to_dev[call.vcomm_src];
          continue;
        }
      }

      { CallNcclGroupStart call;
        if(0 == std::strcmp(line_ptr, "ncclGroupStart()")) {
          hdr.code = CallCode::group_start;
          rank_bufs[rank].append(hdr);
          call.group_seq = rank_group_seqs[rank]++;
          rank_bufs[rank].append(call);
          continue;
        }
      }

      { CallNcclGroupEnd call;
        if(0 == std::strcmp(line_ptr, "ncclGroupEnd()")) {
          hdr.code = CallCode::group_end;
          rank_bufs[rank].append(hdr);
          call.group_seq = rank_group_seqs[rank]++;
          rank_bufs[rank].append(call);
          continue;
        }
      }

      { CallRedOpCreatePreMulSum call;
        uint64_t dummy_ptr;
        if(5 == std::sscanf(line_ptr, "ncclRedOpCreatePreMulSum(%d,%" PRIx64 ",%d,%d,%" PRIx64 ")", &call.vredop, &dummy_ptr, ((int*) &call.elt_ty), &call.residence, &call.vcomm)) {
          hdr.code = CallCode::redop_create_premulsum;
          rank_bufs[rank].append(hdr);
          rank_bufs[rank].append(call);

          // TODO - More info here for stricter validation
          vcommToGlobalMap.at(call.vcomm)->vredops[rank].emplace(call.vredop);
          if (opt_verbose) {
            fprintf(stderr, "Mapping redop %u to rank %u for vcomm %lu\n",
              call.vredop, rank, call.vcomm);
          }

          continue;
        }
      }

      { CallRedOpDestroy call;
        if(2 == std::sscanf(line_ptr, "ncclRedOpDestroy(%d,%" PRIx64 ")", &call.vredop, &call.vcomm)) {
          hdr.code = CallCode::redop_destroy;
          rank_bufs[rank].append(hdr);
          rank_bufs[rank].append(call);
          continue;
        }
      }

      { char coll_name[64];
        CallDataOp call;
        int matched = std::sscanf(line_ptr,
          "nccl%[^(](%" PRIx64 ",%" PRIx64 ",%" PRId64 ",%d,%d,%d,%" PRIx64 ",%" PRIx64 ")",
          coll_name, &call.sptr, &call.dptr, &call.elt_n, (int*) &call.elt_ty, &call.red_op, &call.root, &call.vcomm, &call.vstream
        );

          if (opt_force_fit && call.root != 0) {
            // If we are forcing fit
            if (force_fit_rank_map.find(call.root) != force_fit_rank_map.end()) {
              int proot = force_fit_rank_map.at(call.root);

              if (opt_verbose) {
                fprintf(stderr, "Force fitting virtual_rank call.root=%d to physical_rank=%d",
                  call.root, proot);
                std::cerr << "line " << line_counter << ": " << line << std::endl;
              }
              call.root = proot;
            } else {
              if (opt_verbose) {
                fprintf(stderr, "Skipping CallDataOp: call.root=%d isn't mapped in the available set of vroots",
                  call.root);
                std::cerr << "line " << line_counter << ": " << line << std::endl;
                continue;
              }
            }
          }

        // Debug
        if (opt_force_size_one) {
          call.elt_n = 1;
        }

        if(matched == 9) {
          if(0 == std::strcmp(coll_name, "AllGather"))
            hdr.code = CallCode::allgather;
          else if(0 == std::strcmp(coll_name, "AllReduce"))
            hdr.code = CallCode::allreduce;
          else if(0 == std::strcmp(coll_name, "Broadcast"))
            hdr.code = CallCode::broadcast;
          else if(0 == std::strcmp(coll_name, "Reduce"))
            hdr.code = CallCode::reduce;
          else if(0 == std::strcmp(coll_name, "ReduceScatter"))
            hdr.code = CallCode::reduce_scatter;
          else if(0 == std::strcmp(coll_name, "Send"))
            hdr.code = CallCode::send;
          else if(0 == std::strcmp(coll_name, "Recv"))
            hdr.code = CallCode::recv;
          else
            assert(0);

          // Monotonically increase the sequence number for this device
          int device = vhost_st.vcomm_to_dev[call.vcomm];
          call.dev_seq = vhost_st.dev_seq_map[device]++;

          // Map this call per-global communicator / rank for trace pre-checking
          // Don't check sends and receives for now
          if (hdr.code != CallCode::send && hdr.code != CallCode::recv) {
            auto it = vcommToGlobalMap.find(call.vcomm);
            if (it == vcommToGlobalMap.end()) {
              fprintf(stderr, "ERROR: Couldn't find entry for call.vcomm=0x%" PRIx64 " in vcommToGlobalMap\nMake sure your application has called ncclCommInitRank() on this communicator before using it in a collective\n", call.vcomm);
              assert(0);
            }
            vcommToGlobalMap.at(call.vcomm)->data_ops_map[rank].push_back({hdr, call});
          }

          rank_bufs[rank].append(hdr);
          rank_bufs[rank].append(call);
          continue;
        }
      }
    }

    //
    // PRE-CHECK
    //
    if (!opt_disable_check) {
      preCheck(vuniqueToGlobalMap);
    }

    printf("[%u] Rank %d done pre-processing trace. Dispersing to %d ranks on %zu physical hosts\n",
      getpid(), mpi_rank_me, mpi_rank_n, vhosts.size());

    std::unique_ptr<long long[]> log_sizes(new long long[mpi_rank_n]);
    for(int r=0; r < mpi_rank_n; r++)
      log_sizes[r] = rank_bufs[r].size();
    long long dummy;
    MPI_Scatter(&log_sizes[0], 1, MPI_LONG_LONG, &dummy, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    MPI_Request reqs[32];
    for(int i=0; i < 32; i++) reqs[i] = MPI_REQUEST_NULL;
    for(int r=1; r < mpi_rank_n; r++) {
      if(reqs[r%32] != MPI_REQUEST_NULL)
        MPI_Wait(&reqs[r%32], MPI_STATUS_IGNORE);
      MPI_Isend(rank_bufs[r].bytes(), rank_bufs[r].size(), MPI_BYTE, r, 0, MPI_COMM_WORLD, &reqs[r%32]);
    }
    for(int i=0; i < 32; i++) {
      if(reqs[i] != MPI_REQUEST_NULL)
        MPI_Wait(&reqs[i], MPI_STATUS_IGNORE);
    }

    return std::move(rank_bufs[0]);
  }
  else {
    long long recs_size;
    MPI_Scatter(nullptr, 1, MPI_LONG_LONG, &recs_size, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    ByteBuffer recs;
    recs.reserve(recs_size);
    MPI_Recv(recs.bytes(), recs_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("[%u] Rank %d received %llu bytes of call traces.\n",
      getpid(), mpi_rank_me, recs_size);
    return recs;
  }
}


int main(int arg_n, char **args) {

  // register signal SIGINT and signal handler
  signal(SIGSEGV, signalHandler);

  MPI_Init(&arg_n, &args);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_rank_n);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_me);

  MPI_Barrier(MPI_COMM_WORLD);

  CudaHelp::initialize();

  if(false) {
  shutdown:
    CudaHelp::finalize();
    MPI_Finalize();

    if (opt_progress_thread && mpi_rank_me == 0)
    {
      signal_exit.set_value();
      progress_thread.join();
    }
    return 0;
  }

  bool opt_viable = false;
  bool opt_help = false;
  std::string opt_trace_path;
  for(int i=1; i < arg_n; i++) {
    std::string arg(args[i]);
    if(arg == "-h" || arg == "--help")
      opt_help = true;
    else if (arg == "-v") {
      opt_verbose = true;
    } else if (arg == "-p") {
      opt_progress_thread = true;
    } else if (arg == "-f") {
      opt_force_size_one = true;
    } else if (arg == "-d") {
      opt_disable_check = true;
    } else if (arg == "-x") {
      opt_force_fit = true;
    }  else {
      opt_viable = true;
      opt_trace_path = arg;
    }
  }

  if(!opt_viable || opt_help) {
    if(mpi_rank_me == 0) {
      std::cout<<
        "usage: replay [-h|--help] (<path>|-)\n"
        "  -h, --help   This help message.\n"
        "  -v           Run verbose mode.\n"
        "  -p           Print progress on rank 0 thread.\n"
        "  -d           Disable pre-checking of trace. Allows running of malformed traces\n"
        "  -f           Force all data operations to replay with element count of 1.\n"
        "  -x           Force fit the replay traffic into the physical host and rank count.\n"
        "               This is useful for reproducing performance issues from very large jobs without having to handcraft a minimal trace.\n"
        "               -x will apply the following filters to the replay log:\n"
        "                 Only traces from the first N hosts in the log will be replayed, where N is the physical host count of the replay job.\n"
        "                 The virtual rank specified in ncclCommInitRank() will be mapped to the physical mpi rank of a given process\n"
        "                 The root of collective operations will either be 0 if originally 0, or assigned to the mapped physical rank\n"
        "                 This may produce unreliable results for peer-to-peer (ncclSend and ncclRecv)"
        "  <path>, -    Path to file containing NCCL log. If NCCL log is split over\n"
        "               multiple files then you must concatenate them manually.\n"
        "               \"-\" indicates stdin.\n"
        "\n"
        "It is your responsibility to ensure that the number of MPI hosts and\n"
        "processes launched matches the number of hosts and processes recorded\n"
        "in the trace.\n";
    }
    goto shutdown;
  }

  struct shm_state {
    int phost_me;
  };
  char shm_name[512];
  std::snprintf(shm_name, sizeof(shm_name), "/nccl-test-replay-%lld", osEnv<long long>("OMPI_MCA_ess_base_jobid", 0));

  shm_unlink(shm_name);
  MPI_Barrier(MPI_COMM_WORLD);
  int shm_fd = shm_open(shm_name, O_RDWR|O_CREAT|O_EXCL, S_IRUSR|S_IWUSR);
  int phost_leader = shm_fd != -1;
  if(shm_fd != -1) {
    if(ftruncate(shm_fd, sizeof(shm_state))) {/*ignored*/};
  } else {
    shm_fd = shm_open(shm_name, O_RDWR, S_IRUSR|S_IWUSR);
    assert(shm_fd != -1);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if(phost_leader) shm_unlink(shm_name);
  shm_state *shm = (shm_state*)mmap(nullptr, sizeof(shm_state), PROT_READ|PROT_WRITE, MAP_SHARED, shm_fd, 0);
  assert(shm != nullptr);
  close(shm_fd);

  phost_n = phost_leader;
  MPI_Allreduce(MPI_IN_PLACE, &phost_n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  phost_me = phost_leader;
  MPI_Exscan(MPI_IN_PLACE, &phost_me, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if(mpi_rank_me == 0) phost_me = 0;
  if(phost_leader) shm->phost_me = phost_me;
  MPI_Barrier(MPI_COMM_WORLD);
  if(!phost_leader) phost_me = shm->phost_me;

  std::unique_ptr<int[]> rank_to_phost;
  if(mpi_rank_me == 0) rank_to_phost.reset(new int[mpi_rank_n]);
  MPI_Gather(&phost_me, 1, MPI_INT, rank_to_phost.get(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  if(mpi_rank_me == 0) {
    phost_ranks.reset(new std::vector<int>[phost_n]);
    for(int r=0; r < mpi_rank_n; r++)
      phost_ranks[rank_to_phost[r]].push_back(r);
    for(int h=1; h < phost_n; h++)
      assert(phost_ranks[0].size() == phost_ranks[h].size());
  }

  ByteBuffer trace = loadDebugCallTrace(opt_trace_path);
  if (trace.size() == 0) {
    fprintf(stderr, "trace size is 0. Please provide a valid trace file with at least one NCCL trace.\n");
    failure.store(1);
  }

  std::chrono::duration<double> duration;
  playTrace(trace, &duration);

  // Get end time
  // Gather elapsed time and call counts from all ranks to rank 0
  std::vector<std::chrono::duration<double>> durations(mpi_rank_n);
  std::vector<size_t> call_counts(mpi_rank_n);

  // If we're waiting on the rank 0 progress thread to join, don't proceed
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Gather(&duration, sizeof(duration), MPI_CHAR, &durations[0], sizeof(duration), MPI_CHAR, 0, MPI_COMM_WORLD);
  MPI_Gather(&data_op_call_count, sizeof(data_op_call_count), MPI_CHAR, &call_counts[0], sizeof(data_op_call_count), MPI_CHAR, 0, MPI_COMM_WORLD);

  // Gather op count from all ranks to rank 0
  if(mpi_rank_me == 0) {
    std::cout<<(failure.load() ? "FAILURE" : "SUCCESS") <<std::endl;
    for (size_t i = 0; i < durations.size(); i++)
    {
      std::cout << "[" << i << "] Time elapsed: " << durations[i].count() << " s" << std::endl;
      std::cout << "[" << i << "] NCCL Collective Calls: "   << call_counts[i] << std::endl;
    }
  }

  goto shutdown;
}
