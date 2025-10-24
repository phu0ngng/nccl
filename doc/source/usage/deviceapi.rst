******************************
Device-Initiated Communication
******************************

Starting with version 2.28, NCCL provides a device-side communication API, making it possible to use communication
primitives directly from user CUDA kernels.

Device API
----------

Device API consists of the following modules:

 * **LSA (Load/Store Accessible)** -- for communication between devices accessible via memory load/store operations,
   using CUDA P2P.  This includes devices connected over NVLink and some devices connected over PCIe (the latter are
   currently limited to devices with P2P connectivity (as indicated by ``nvidia-smi topo -p2p p``), subject to the
   :ref:`env_NCCL_P2P_LEVEL` distance check).
 * **Multimem** -- for communication between devices using the hardware multicast feature provided by
   NVLink SHARP (available on some datacenter GPUs since the Hopper generation).
 * **GIN (GPU-Initiated Networking)** -- for communication over the network.  This module is under active development
   and will not be covered here at this time.

The device API relies on symmetric memory (see :ref:`window_reg`), which in turn depends on GPU virtual memory
management (see :ref:`env_NCCL_CUMEM_ENABLE`) and optionally -- for multimem support -- on NVLink SHARP (see
:ref:`env_NCCL_NVLS_ENABLE`).

Host-Side Setup
---------------

To perform communication from the device, a device communicator needs to be created using :c:func:`ncclDevCommCreate`.
Data transfer operations on buffers require symmetric memory windows (see :ref:`window_reg`).  A custom
communication kernel can then be launched using the standard CUDA syntax.  The code excerpt below demonstrates
these steps:

.. code:: C

  int main() {
    [...]
    NCCLCHECK(ncclCommInitRank(&comm, nranks, id, rank));

    /* Buffer initialization and window creation */
    char* buffer;
    size_t size = 256*1048576;
    NCCLCHECK(ncclMemAlloc((void**)&buffer, size));
    ncclWindow_t win;
    NCCLCHECK(ncclCommWindowRegister(comm, buffer, size, &win, NCCL_WIN_COLL_SYMMETRIC));

    /* Get device communicator */
    ncclDevComm devComm;
    ncclDevCommRequirements reqs;
    memset(&reqs, 0, sizeof(ncclDevCommRequirements));
    int nCTAs = 16;
    reqs.lsaBarrierCount = nCTAs;
    NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));

    /* Launch user kernel */
    customKernel<<<nCTAs, 256>>>(devComm, win);
    [...]
  }

Depending on the kernel and application requirements, the same window can be used for input and output, or multiple
windows may be needed.  When creating a device communicator, the resources that the kernel will need should be specified
via the requirements list (see :c:type:`ncclDevCommRequirements`).  In the
above example we specify just the number of barriers that the kernel will need, in this case one for each CTA the kernel
is to be launched on (16, each CTA running 256 threads).

Simple Device Kernel
--------------------

.. code:: C

  template <typename T>
  __global__ void inPlaceAllReduceKernel(ncclDevComm devComm, ncclWindow_t win, size_t offset, size_t count) {
    ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamTagLsa(), blockIdx.x };
    bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);

    const int rank = devComm.lsaRank, nRanks = devComm.lsaSize;
    const int globalTid = threadIdx.x + blockDim.x * (rank + blockIdx.x * nRanks);
    const int globalNthreads = blockDim.x * gridDim.x * nRanks;

    for (size_t o = globalTid; o < count; o += globalNthreads) {
      T v = 0;
      for (int peer=0; peer<nRanks; peer++) {
        T* inputPtr = (T*)ncclGetLsaPointer(win, offset, peer);
        v += inputPtr[o];
      }
      for (int peer=0; peer<nRanks; peer++) {
        T* outputPtr = (T*)ncclGetLsaPointer(win, offset, peer);
        outputPtr[o] = v;
      }
    }

    bar.sync(ncclCoopCta(), cuda::memory_order_release);
  }

The above code excerpt shows a simple device kernel -- an in-place variant (the input buffer is reused for the output)
of AllReduce, utilizing LSA support (data is transferred via memory load/store instructions).

The start of the buffer is specified as a (byte-based) *offset* within the previously registered window *win* (see
:ref:`window_reg`); the buffer consists of *count* elements of type *T*.

Before the kernel can start processing data, it needs to ensure that all participants are ready.  It creates a memory
barrier session *bar* (see :c:type:`ncclLsaBarrierSession`) and uses it to synchronize across all the threads of the CTA
(*ncclCoopCta*) and the ranks of the communicator (*devComm*).  *ncclTeamTagLsa* indicates the subset of ranks the
barrier will apply to and *blockIdx.x* is the CTA's local index, used to select the barrier.

The kernel then calculates a globally unique index for each thread as well as the overall thread count, and can finally
start processing data, using an all-to-all communication pattern.  In each iteration, every participating thread loads a
single input element of each communicator rank.  :c:func:`ncclGetLsaPointer` is used to calculate the locally-accessible
address of the start of the buffer within each rank (remote device memory was previously mapped into the local address
space -- see :ref:`window_reg`).  Extracted input data is accumulated and then stored back at each rank.  Before the
kernel terminates, another memory synchronization needs to take place to ensure that all the threads have finished
processing their data.

Note that this simple implementation would likely fall short of achieving the peak bandwidth, as it utilizes neither
vectorization nor loop unrolling.

Multimem Device Kernel
----------------------

.. code-block:: C
  :emphasize-lines: 6,13,15,17-18

  int main() {
    [...]
    memset(&reqs, 0, sizeof(ncclDevCommRequirements));
    int nCTAs = 16;
    reqs.lsaBarrierCount = nCTAs;
    reqs.lsaMultimem = true;
    NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));
    [...]
  }

  template <typename T>
  __global__ void inPlaceAllReduceKernel(ncclDevComm devComm, ncclWindow_t win, size_t offset, size_t count) {
    ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamTagLsa(), blockIdx.x, /*multimem*/true };
    [...]
    T* mmPtr = (T*)ncclGetLsaMultimemPointer(win, offset, devComm);
    for (size_t o = globalTid; o < count; o += globalNthreads) {
      T v = multimem_sum(mmPtr+o);
      multimem_st(mmPtr+o, v);
    }
    [...]
  }

The above code excerpt demonstrates modifications needed to the earlier code segments to enable multimem support (the
lines with critical changes are highlighted).  On the host
side, ``lsaMultimem`` needs to be set in the requirements prior to creating the device communicator
(:c:func:`ncclDevCommCreate` will fail if the necessary hardware support is unavailable).

Within the device kernel, we can switch the memory barrier to a multimem-optimized variant by adding an extra argument
to the constructor.  The processing loop is actually simpler with multimem: :c:func:`ncclGetLsaMultimemPointer` needs to
be invoked just once per kernel.  The returned multicast memory pointer enables access to the device memory of all the
ranks of the communicator without having to iterate over them, and the data can be reduced in hardware.  To keep this
example simple, the implementations of ``multimem_sum`` and ``multimem_st`` are not included. Those need to be
implemented using PTX, e.g., ``multimem.ld_reduce.global.add`` and ``multimem.st.global``.

Thread Groups
-------------

Many functions in the device API take a thread cooperative group as input to indicate which threads within the CTA will take part in the operation. NCCL provides three predefined ones: ``ncclCoopThread()``, ``ncclCoopWarp()`` and ``ncclCoopCta()``.

Users may also pass CUDA cooperative groups, or any class which provides ``thread_rank()``, ``size()`` and ``sync()`` functions.

Teams
-----

To address remote ranks or perform barriers, NCCL refers to subsets of ranks within the global communicator as "teams".
NCCL provides three predefined ones: ``ncclTeamWorld()``, ``ncclTeamLsa()``, and ``ncclTeamRail()``.

Host-Accessible Device Pointer Functions
----------------------------------------

Starting with version 2.29, NCCL provides host-accessible functions that enable host code to obtain pointers to LSA
memory regions.

The four functions are :c:func:`ncclGetLsaMultimemDevicePointer` (multimem base pointer), :c:func:`ncclGetMultimemDevicePointer` (multimem base pointer with custom handle), :c:func:`ncclGetLsaDevicePointer` (LSA peer pointer),
and :c:func:`ncclGetPeerDevicePointer` (world rank peer pointer). Functions automatically discover the associated communicator
from the window object and return ``ncclResult_t`` error codes. 

Usage Example:
.. code:: C

  int main() {
    [...]
    // Allocate symmetric memory buffer
    char* buffer;
    size_t size = 256 * 1024 * 1024;  // 256 MB buffer
    NCCLCHECK(ncclMemAlloc((void**)&buffer, size));

    // Create window with the allocated buffer
    ncclWindow_t win;
    NCCLCHECK(ncclCommWindowRegister(comm, buffer, size, &win, NCCL_WIN_COLL_SYMMETRIC));

    // Get host-accessible pointers
    void* multimemPtr;
    void* lsaPtr;
    void* peerPtr;

    // Get multimem pointer (returns nullptr if multimem not supported)
    NCCLCHECK(ncclGetLsaMultimemDevicePointer(win, 0, &multimemPtr));
    if (multimemPtr == nullptr) {
        // Multimem not available, use fallback
    }

    // Get LSA pointer for peer 1
    NCCLCHECK(ncclGetLsaDevicePointer(win, 0, 1, &lsaPtr));

    // Get peer pointer for world rank 2
    NCCLCHECK(ncclGetPeerDevicePointer(win, 0, 2, &peerPtr));

    // Use pointers in custom kernels or legacy code
    customKernel<<<nCTAs, 256>>>(multimemPtr, lsaPtr, peerPtr);

    // Cleanup
    NCCLCHECK(ncclCommWindowDeregister(comm, &win));
    // Device pointers are invalidated after window deregistration
    NCCLCHECK(ncclMemFree(buffer));
    [...]
  }

Important notes: Pointer lifetime is limited to the shorter of Window and Communicator lifetime. Functions should be called once
and pointers cached for reuse. For detailed function documentation, see :ref:`device_api_host_functions`.
