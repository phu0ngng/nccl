******************************
Device-Initiated Communication
******************************

Starting with version 2.28, NCCL provides a device-side communication API, making it possible to use communication
primitives directly from user CUDA kernels.

Device API
----------

Device API consists of the following modules:

 * **LSA (Load/Store Accessible)** -- for communication between devices accessible via memory load/store operations,
   using CUDA P2P.  This includes devices connected over NVLink and some devices connected over PCIe, so long as they
   have P2P connectivity with each other (as indicated by ``nvidia-smi topo -p2p p``).  Up to NCCL 2.28.3, the
   availability of LSA was also subject to the :ref:`env_NCCL_P2P_LEVEL` distance check, but that is no longer the case
   with newer versions.
 * **Multimem** -- for communication between devices using the hardware multicast feature provided by
   NVLink SHARP (available on some datacenter GPUs since the Hopper generation).
 * **GIN (GPU-Initiated Networking)** -- for communication over the network (since NCCL 2.28.7).

The device API relies on symmetric memory (see :ref:`window_reg`), which in turn depends on GPU virtual memory
management (see :ref:`env_NCCL_CUMEM_ENABLE`) and optionally -- for multimem support -- on NVLink SHARP (see
:ref:`env_NCCL_NVLS_ENABLE`).

Host-Side Setup
---------------

To perform communication from the device kernel, a device communicator needs to be created first, using
:c:func:`ncclDevCommCreate`.
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
    customKernel<<<nCTAs, 512>>>(devComm, win);
    [...]
  }

Depending on the kernel and application requirements, the same window can be used for input and output, or multiple
windows may be needed.  When creating a device communicator, the resources that the kernel will need should be specified
via the requirements list (see :c:type:`ncclDevCommRequirements`).  In the above example we specify just the number of
barriers that our LSA kernel will need, in this case one for each CTA the kernel
is to be launched on (16, each CTA running 512 threads).

Simple LSA Kernel
-----------------

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
      for (int peer = 0; peer < nRanks; peer++) {
        T* inputPtr = (T*)ncclGetLsaPointer(win, offset, peer);
        v += inputPtr[o];
      }
      for (int peer = 0; peer < nRanks; peer++) {
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
(*ncclCoopCta()*; see :ref:`devapi_coops`) and the ranks of the communicator (*devComm*).  *ncclTeamTagLsa* indicates
the subset of ranks the barrier will apply to (see :ref:`devapi_teams`) -- this kernel assumes that all ranks are
LSA-connected. *blockIdx.x* is the CTA's local index, used to select the barrier.

The kernel then calculates a globally unique index for each thread as well as the overall thread count, and can finally
start processing data, using an all-to-all communication pattern.  In each iteration of the outer loop, every
participating thread loads a single input element from each communicator rank (the first inner loop).
:c:func:`ncclGetLsaPointer` is used to calculate the locally-accessible
address of the start of the buffer within each rank (remote device memory was previously mapped into the local address
space -- see :ref:`window_reg`).  Extracted input data is accumulated and the result is stored back at each rank (the
second inner loop).  Before the
kernel terminates, another memory synchronization needs to take place to ensure that all participants have finished
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
example simple, the implementations of ``multimem_sum`` and ``multimem_st`` are not included; they need to be
implemented using PTX, e.g., ``multimem.ld_reduce.global.add`` and ``multimem.st.global``.

.. _devapi_coops:

Thread Groups
-------------

Many functions in the device API take a thread cooperative group as input to indicate which threads within the CTA will
take part in the operation. NCCL provides three predefined ones: ``ncclCoopThread()``, ``ncclCoopWarp()``, and (the most
commonly used) ``ncclCoopCta()``.

Users may also pass CUDA cooperative groups, or any class which provides ``thread_rank()``, ``size()``, and ``sync()``
methods.

.. _devapi_teams:

Teams
-----

To address remote ranks or perform barriers, NCCL refers to subsets of ranks within a communicator as "teams".
NCCL provides three predefined ones:

 * ``ncclTeamWorld()`` -- the "world" team, encompassing all the ranks of a given communicator.
 * ``ncclTeamLsa()`` -- all the peers accessible from the local rank using load/store operations.
 * ``ncclTeamRail()`` -- the set of peers directly accessible from the local rank over the network, assuming that the
   network fabric is rail-optimized (see :ref:`env_NCCL_CROSS_NIC`).

The ``ncclTeam`` structure contains fairly self-explanatory elements ``nRanks``, ``rank``, and ``stride``.  The device
API contains functions to verify team membership, convert rank numbers between teams, etc.  The world and LSA teams are
always contiguous (stride ``1``), whereas the rail team is typically not -- its stride equals the size of the LSA team
(the assumption is thus that each rank *n* within the local LSA team has direct network connectivity with corresponding
ranks *n* of all remote LSA teams).

GIN Device Kernel
-----------------

.. code-block:: C
  :emphasize-lines: 4-6,12-34

  int main() {
    [...]
    memset(&reqs, 0, sizeof(ncclDevCommRequirements));
    int nCTAs = 1;
    reqs.railGinBarrierCount = nCTAs;
    reqs.ginSignalCount = 1;
    NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));
    [...]
  }

  template <typename T>
  __global__ void ginAlltoAllKernel(ncclDevComm devComm, ncclWindow_t win,
                                    size_t inputOffset, size_t outputOffset, size_t count) {
    int ginContext = 0;
    ncclGinSignal_t signalIndex = 0;
    ncclGin gin { devComm, ginContext };
    uint64_t signalValue = gin.readSignal(signalIndex);

    ncclGinBarrierSession<ncclCoopCta> bar { ncclCoopCta(), gin, ncclTeamWorld(devComm),
                                             devComm.railGinBarrier, blockIdx.x };
    bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

    const int rank = devComm.rank, nRanks = devComm.nRanks;
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int nThreads = blockDim.x * gridDim.x;

    const size_t size = count * sizeof(T);
    for (int peer = tid; peer < nRanks; peer += nThreads) {
      gin.put(ncclTeamWorld(devComm), peer, win, outputOffset + rank * size,
              win, inputOffset + peer * size, size, ncclGin_SignalInc{signalIndex});
    }

    gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + nRanks);
    gin.flush(ncclCoopCta());
  }

The above code excerpt demonstrates modifications needed to the earlier host code to enable GIN support, available since
NCCL 2.28.7 (the lines with critical changes are highlighted), and also includes a GIN AlltoAll kernel.  On the host
side, compared to the LSA kernels, we request a launch on just a single CTA (because our kernel doesn't have much to do)
and we set :c:macro:`railGinBarrierCount` and :c:macro:`ginSignalCount` to request GIN-specific barriers and signals
(:c:func:`ncclDevCommCreate` will fail if GIN support is unavailable).  As with LSA barriers, we need as many of them as
CTAs, but signals (used for completion notifications) can be shared between CTAs so, for this simple example, we'll use
just one per rank (for performance-oriented kernels, keeping signals exclusive to each CTA can improve performance).

On the device side, GIN API centers around the :c:type:`ncclGin` object, initialized using the device communicator and a
GIN
context index (``0`` will do for this simple example but, for performance-oriented kernels, using multiple contexts can
provide a performance boost).  To avoid race conditions, the initial value of the signal must be read *prior to* the
synchronizing barrier.  GIN-specific barriers look much like their LSA counterparts, being local to each CTA, but
communicating over the network, not memory.  *ncclTeamWorld* indicates all the ranks of a communicator (this kernel
assumes
that all the ranks can reach one another over the network, which in general need not be the case -- see
:ref:`env_NCCL_CROSS_NIC`).

Unlike with the AllReduce kernels, for AlltoAll the calculated thread index needs to be unique only locally within each
rank.  This is then used to determine the destination peer.  The main GIN data transfer operation is the one-sided
:c:func:`put`, here launched in parallel on all participating threads, one per each destination peer (the loop is needed
merely if the total rank count exceeds the local thread count -- this is why we launched on just a single CTA).
:c:func:`put` takes the usual arguments such as the destination rank and buffer address, the source buffer, and the
transfer size.  It also accepts several optional arguments; the above example takes advantage of the *remoteAction*,
requesting that the destination peer increments the value of its local signal once the payload has been settled.

Once the local signal has been incremented by *nRanks*, we know that every peer has deposited their data in this rank's
output buffer and thus that the buffer is ready; :c:func:`waitSignal` can be used to block until that happens.  Before
terminating, the kernel still needs to :c:func:`flush` all the previously initiated outgoing :c:func:`put` operations --
while that does not guarantee remote completion, it does ensure that the local input buffer is safe to reuse.  We can
skip an explicit barrier at the end, since :c:func:`waitSignal` and :c:func:`flush` together ensure that nobody else is
using this rank's buffers.
