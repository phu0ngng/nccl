#ifndef _NCCL_SYM_PTR__FUNCS_H_
#define _NCCL_SYM_PTR__FUNCS_H_
#include "ptr__types.h"
#include "core__funcs.h"
#include "comm__types.h"

template<typename T>
NCCL_HOST_DEVICE_INLINE constexpr ncclSymPtr<T>::ncclSymPtr(ncclWindow_t window, size_t offset):
  window(window), offset(offset) {
}

template<typename T>
template<typename U>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>::operator ncclSymPtr<U>() const {
  return {window, offset};
}

template<typename T>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& ncclSymPtr<T>::operator+=(int d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) + d);
  return *this;
}
template<typename T>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& ncclSymPtr<T>::operator+=(unsigned int d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) + d);
  return *this;
}

template<typename T>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& ncclSymPtr<T>::operator+=(long d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) + d);
  return *this;
}
template<typename T>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& ncclSymPtr<T>::operator+=(unsigned long d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) + d);
  return *this;
}

template<typename T>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& ncclSymPtr<T>::operator+=(long long d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) + d);
  return *this;
}
template<typename T>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& ncclSymPtr<T>::operator+=(unsigned long long d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) + d);
  return *this;
}

template<typename T>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& ncclSymPtr<T>::operator-=(int d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) - d);
  return *this;
}
template<typename T>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& ncclSymPtr<T>::operator-=(unsigned int d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) - d);
  return *this;
}

template<typename T>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& ncclSymPtr<T>::operator-=(long d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) - d);
  return *this;
}
template<typename T>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& ncclSymPtr<T>::operator-=(unsigned long d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) - d);
  return *this;
}

template<typename T>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& ncclSymPtr<T>::operator-=(long long d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) - d);
  return *this;
}
template<typename T>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T>& ncclSymPtr<T>::operator-=(unsigned long long d) {
  offset = reinterpret_cast<size_t>(reinterpret_cast<T*>(offset) - d);
  return *this;
}

#if __CUDACC__
template<typename T>
NCCL_DEVICE_INLINE T* ncclSymPtr<T>::localPtr() const {
  return (T*)ncclSymGetLocalPointer(window, offset);
}
#endif

#if __CUDACC__
template<typename T>
NCCL_DEVICE_INLINE T* ncclSymPtr<T>::nearPtr(int peer) const {
  return (T*)ncclSymGetNearPointer(window, offset, peer);
}
#endif

#if __CUDACC__
template<typename T>
NCCL_DEVICE_INLINE T* ncclSymPtr<T>::peerPtr(int peer) const {
  return (T*)ncclSymGetPeerPointer(window, offset, peer);
}
#endif

#if __CUDACC__
template<typename T>
NCCL_DEVICE_INLINE T* ncclSymPtr<T>::peerPtr(ncclSymTeam team, int peer) const {
  return (T*)ncclSymGetPeerPointer(window, offset, team, peer);
}
#endif

#if __CUDACC__
template<typename T>
NCCL_DEVICE_INLINE T* ncclSymPtr<T>::multimemPtr(ncclSymMultimemHandle mmHandle) const {
  return (T*)ncclSymGetMultimemPointer(window, offset, mmHandle);
}
#endif

#if __CUDACC__
template<typename T>
NCCL_DEVICE_INLINE T* ncclSymPtr<T>::nearMultimemPtr(ncclSymComm const& comm) const {
  return (T*)ncclSymGetNearMultimemPointer(window, offset, comm);
}
#endif

template<typename T, typename Int>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T> operator+(ncclSymPtr<T> p, Int d) {
  return p += d;
}
template<typename T, typename Int>
NCCL_HOST_DEVICE_INLINE ncclSymPtr<T> operator-(ncclSymPtr<T> p, Int d) {
  return p -= d;
}
template<typename T>
NCCL_HOST_DEVICE_INLINE ptrdiff_t operator-(ncclSymPtr<T> a, ncclSymPtr<T> b) {
  return reinterpret_cast<T*>(a.offset) - reinterpret_cast<T*>(b.offset);
}

template<typename T>
NCCL_HOST_DEVICE_INLINE bool operator==(ncclSymPtr<T> a, ncclSymPtr<T> b) {
  return a.window == b.window && a.offset == b.offset;
}
template<typename T>
NCCL_HOST_DEVICE_INLINE bool operator!=(ncclSymPtr<T> a, ncclSymPtr<T> b) {
  return a.window != b.window || a.offset != b.offset;
}

#endif // _NCCL_SYM_PTR__FUNCS_H_
