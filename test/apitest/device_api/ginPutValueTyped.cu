#include "nccl_device.h"
#include "ncclDevApiCommon_test.cuh"

constexpr int SRC_RANK = 0;
constexpr int DST_RANK = 1;

// Byte array types for sizes 3, 5, 6, 7 (no single built-in type); comparable for kernel poll.
template<size_t N>
struct Bytes {
  uint8_t d[N];
  __host__ __device__ bool operator==(const Bytes<N>& other) const {
    for (size_t i = 0; i < N; i++)
      if (d[i] != other.d[i]) return false;
    return true;
  }
  __host__ __device__ bool operator!=(const Bytes<N>& other) const { return !(*this == other); }
};
// Allow comparison when LHS is volatile (e.g. volatile T* from putPtr in device poll loop).
template<size_t N>
__host__ __device__ bool operator!=(const volatile Bytes<N>& a, const Bytes<N>& b) {
  for (size_t i = 0; i < N; i++)
    if (a.d[i] != b.d[i]) return true;
  return false;
}

template<typename T>
__global__ void putValueTypedKernel(ncclDevComm comm, ncclWindow_t window, size_t offset, T value) {
#if __CUDA_ARCH__ >= 700
  ncclTeam world = ncclTeamWorld(comm);
  ncclGin gin(comm, 0);

  if (world.nRanks < 2) {
    return;
  }

  if (world.rank == SRC_RANK) {
    gin.putValue(world, DST_RANK, window, offset, value);
  }

  if (world.rank == DST_RANK) {
    volatile T* putPtr = (volatile T*)ncclGetLocalPointer(window, offset);
    while (*putPtr != value) {
      continue;
    }
  }
#endif
}

// Host helper: test value for each type (used to pass into kernel).
template<typename T> T getPutValueTestValue();
template<> uint8_t getPutValueTestValue<uint8_t>() { return 0xAB; }
template<> uint16_t getPutValueTestValue<uint16_t>() { return 0x1234; }
template<> Bytes<3> getPutValueTestValue<Bytes<3>>() { return {{0x12, 0x34, 0x56}}; }
template<> uint32_t getPutValueTestValue<uint32_t>() { return 0x12345678u; }
template<> Bytes<5> getPutValueTestValue<Bytes<5>>() { return {{0x11, 0x22, 0x33, 0x44, 0x55}}; }
template<> Bytes<6> getPutValueTestValue<Bytes<6>>() { return {{0x01, 0x02, 0x03, 0x04, 0x05, 0x06}}; }
template<> Bytes<7> getPutValueTestValue<Bytes<7>>() { return {{0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7}}; }
template<> uint64_t getPutValueTestValue<uint64_t>() { return 0x123456789ABCDEF0ULL; }
template<> float getPutValueTestValue<float>() { return 3.14f; }
template<> double getPutValueTestValue<double>() { return 3.14159265358979323846; }

////////////////////////////////////////////////////////////////////////////////
// Test class (template required by TYPED_TEST_CASE)
////////////////////////////////////////////////////////////////////////////////

typedef ::testing::Types<
  uint8_t,
  uint16_t,
  Bytes<3>,
  uint32_t,
  Bytes<5>,
  Bytes<6>,
  Bytes<7>,
  uint64_t,
  float,
  double
> PutValueTypes;

template <typename T>
class GinPutValueTyped_test : public ncclDevApiCommon_test {
public:
protected:
  std::vector<void*> signalBuffers;
  std::vector<ncclWindow_t> signalWindows;
  std::vector<void*> putBuffers;
  std::vector<ncclWindow_t> putWindows;
  size_t bufferSize;

  void SetUp() override {
    ncclDevApiCommon_test::SetUp();
    bufferSize = sizeof(uint64_t);
    allocateAndRegisterWindows(this->nVis, this->comms, bufferSize, signalBuffers, signalWindows);
    allocateAndRegisterWindows(this->nVis, this->comms, bufferSize, putBuffers, putWindows);
  }

  void TearDown() override {
    deregisterAndFreeWindows(this->nVis, this->comms, signalBuffers, signalWindows);
    deregisterAndFreeWindows(this->nVis, this->comms, putBuffers, putWindows);
    ncclDevApiCommon_test::TearDown();
  }
};

TYPED_TEST_CASE(GinPutValueTyped_test, PutValueTypes);

TYPED_TEST(GinPutValueTyped_test, put_value) {
  using T = TypeParam;
  T value = getPutValueTestValue<T>();

  ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginForceEnable = true;
  TESTCHECK(this->createDevComms(reqs));

  for (int i = 0; i < this->nVis; i++) {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(i));
    putValueTypedKernel<T><<<1, 1, 0, this->streams[i]>>>(this->devComms[i], this->putWindows[i], 0, value);
  }
  this->syncAllDevices();
  cudaError_t err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << "Kernel failed for sizeof(T)=" << sizeof(T) << ": " << cudaGetErrorString(err);
}
