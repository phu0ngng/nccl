**********
Device API
**********

Host-Side Setup
===============

ncclDevComm
-----------

.. c:type:: ncclDevComm

A structure describing a device communicator, as created on the host side using :c:func:`ncclDevCommCreate`.  The
structure is used primarily on the device side; elements that could be of particular interest include:

 .. c:member:: int rank

 The rank within the communicator.

 .. c:member:: int nRanks

 The size of the communicator.

 .. c:member:: int lsaRank
 .. c:member:: int lsaSize

 Rank within the local LSA team and its size (see :ref:`devapi_teams`).

 .. c:member:: uint8_t ginContextCount

 The number of supported GIN contexts (see :cpp:class:`ncclGin`; available since NCCL 2.28.7).

ncclDevCommCreate
-----------------

.. c:function:: ncclResult_t ncclDevCommCreate(ncclComm_t comm, struct ncclDevCommRequirements const* reqs, struct ncclDevComm* outDevComm)

Creates a new device communicator (see :c:type:`ncclDevComm`) corresponding to the supplied host-side communicator
*comm*.  The result is returned in the *outDevComm* buffer (which needs to be supplied by the caller).  The caller needs
to also provide a filled-in list of requirements via the *reqs* argument (see :c:type:`ncclDevCommRequirements`); the
function will allocate any necessary resources to meet them. It is recommended to call :c:func:`ncclCommQueryProperties`
before calling the function; the function will fail if the specified requirements are not supported. Since this is a
collective call, every rank in the communicator needs to participate.  If called within a group, *outDevComm* may not be 
filled in until ``ncclGroupEnd()`` has completed.

Note that this is a *host-side* function.

ncclDevCommDestroy
------------------

.. c:function:: ncclResult_t ncclDevCommDestroy(ncclComm_t comm, struct ncclDevComm const* devComm)

Destroys a device communicator (see :c:type:`ncclDevComm`) previously created using :c:func:`ncclDevCommCreate` and
releases any allocated resources.  The caller must ensure that no device kernel that uses this device communicator could
be running at the time this function is invoked.

Note that this is a *host-side* function.

ncclDevCommRequirements
-----------------------

.. c:type:: ncclDevCommRequirements

A host-side structure specifying the list of requirements when creating device communicators (see
:c:type:`ncclDevComm`). Since NCCL 2.29, this struct must be initialized using ``NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER``.

 .. c:member:: bool lsaMultimem

 Specifies whether multimem support is required for all LSA ranks.

 .. c:member:: int lsaBarrierCount

 Specifies the number of memory barriers to allocate (see :cpp:class:`ncclLsaBarrierSession`).

 .. c:member:: int railGinBarrierCount

 Specifies the number of network barriers to allocate (see :cpp:class:`ncclGinBarrierSession`; available since NCCL
 2.28.7).

 .. c:member:: int barrierCount

 Specifies the minimum number for both the memory and network barriers (see above; available since NCCL 2.28.7).

 .. c:member:: int ginSignalCount

 Specifies the number of network signals to allocate (see :cpp:type:`ncclGinSignal_t`; available since NCCL 2.28.7).

 .. c:member:: int ginCounterCount

 Specifies the number of network counters to allocate (see :cpp:type:`ncclGinCounter_t`; available since NCCL 2.28.7).

 .. c:member:: ncclDevResourceRequirements_t* resourceRequirementsList

 Specifies a list of resource requirements.  This is best set to NULL for now.

 .. c:member:: ncclTeamRequirements_t* teamRequirementsList

 Specifies a list of requirements for particular teams.  This is best set to NULL for now.


ncclCommQueryProperties
-----------------------

.. c:function:: ncclResult_t ncclCommQueryProperties(ncclComm_t comm, ncclCommProperties_t* props)

Exposes communicator properties by filling in *props*. Before calling this function, *props* must be initialized using ``NCCL_COMM_PROPERTIES_INITIALIZER``. Introduced in NCCL 2.29.

Note that this is a *host-side* function.

ncclCommProperties_t
--------------------

.. c:type:: ncclCommProperties_t

A structure describing the properties of the communicator. Introduced in NCCL 2.29. Properties include:

 .. c:member:: int rank

 Rank within the communicator.

 .. c:member:: int nRanks

 Size of the communicator.

 .. c:member:: int cudaDev

 CUDA device index.

 .. c:member:: int nvmlDev

 NVML device index.

 .. c:member:: bool deviceApiSupport

 Whether the device API is supported. If false, a :c:type:`ncclDevComm` cannot be created.

 .. c:member:: bool multimemSupport

 Whether ranks in the same LSA team can communicate using multimem. If false, a :c:type:`ncclDevComm` cannot be created with multimem resources.

 .. c:member:: ncclGinType_t ginType

 The GIN type supported by the communicator. If equal to :c:macro:`NCCL_GIN_TYPE_NONE`, a :c:type:`ncclDevComm` cannot be created with GIN resources.


ncclGinType_t
-------------

.. c:type:: ncclGinType_t

GIN type. Communication between different GIN types is not supported. Possible values include:

 .. c:macro:: NCCL_GIN_TYPE_NONE

   GIN is not supported.

 .. c:macro:: NCCL_GIN_TYPE_PROXY

   Host Proxy GIN type.

 .. c:macro:: NCCL_GIN_TYPE_GDAKI

   GPUDirect Async Kernel-Initiated (GDAKI) GIN type.

LSA
===

All functionality described from this point on is available on the device side only.

ncclLsaBarrierSession
---------------------

.. cpp:class:: template<typename Coop> ncclLsaBarrierSession

 A class representing a memory barrier session.

 .. cpp:function:: ncclLsaBarrierSession(Coop coop, ncclDevComm const& comm, ncclTeamTagLsa tag, uint32_t index, bool multimem=false)

  Initializes a new memory barrier session.  *coop* represents a cooperative group (see :ref:`devapi_teams`).
  *comm* is the device communicator created using :c:func:`ncclDevCommCreate`.
  *ncclTeamTagLsa* is here to indicate which subset of ranks the barrier will apply to.  The identifier of the underlying
  barrier to use is provided by *index* (it should be different for each *coop*; typically set to ``blockIdx.x`` to
  ensure uniqueness between CTAs).  *multimem* requests a hardware-accelerated implementation using memory multicast.

 .. cpp:function:: void arrive(Coop, cuda::memory_order order)

 Signals the arrival of the thread at the barrier session.

 .. cpp:function:: void wait(Coop, cuda::memory_order order)

 Blocks until all threads of all team members arrive at the barrier session.

 .. cpp:function:: void sync(Coop, cuda::memory_order order)

 Synchronizes all threads of all team members that participate in the barrier session (combines ``arrive`` and
 ``wait``).

ncclGetPeerPointer
------------------

.. cpp:function:: void* ncclGetPeerPointer(ncclWindow_t w, size_t offset, int peer)

Returns a load/store accessible pointer to the memory buffer of device *peer* within the window *w*.  *offset* is
byte-based.  *peer* is a rank index within the world team (see :ref:`devapi_teams`).  This function will return NULL if
the *peer* is not within the LSA team.

ncclGetLsaPointer
-----------------

.. cpp:function:: void* ncclGetLsaPointer(ncclWindow_t w, size_t offset, int lsaPeer)

Returns a load/store accessible pointer to the memory buffer of device *lsaPeer* within the window *w*.  *offset* is
byte-based.  This is similar to ``ncclGetPeerPointer``, but here *lsaPeer* is a rank index with the LSA team (see
:ref:`devapi_teams`).

ncclGetLocalPointer
-------------------

.. cpp:function:: void* ncclGetLocalPointer(ncclWindow_t w, size_t offset)

Returns a load-store accessible pointer to the memory buffer of the current device within the window *w*.  *offset* is
byte-based.  This is just a shortcut version of ``ncclGetPeerPointer`` with *devComm.rank* as *peer*, or ``ncclGetLsaPointer`` with *devComm.lsaRank* as *lsaPeer*.

Multimem
========

ncclGetLsaMultimemPointer
-------------------------

.. cpp:function:: void* ncclGetLsaMultimemPointer(ncclWindow_t w, size_t offset, ncclDevComm const& devComm)

Returns a multicast memory pointer associated with the window *w* and device communicator *devComm*.  *offset*
is byte-based.  Availability of multicast memory is hardware-dependent.

.. _device_api_host_functions:

Host-Accessible Device Pointer Functions
========================================

The following functions provide host-side access to device pointer functionality, enabling host code to obtain
pointers to LSA memory regions.

All functions return ``ncclResult_t`` error codes. On success, ``ncclSuccess`` is returned.
On failure, appropriate error codes are returned (e.g., ``ncclInvalidArgument`` for invalid parameters,
``ncclInternalError`` for internal failures), unless otherwise specified.

The returned pointers are valid for the lifetime of the window.
Pointers should not be used after either the window or communicator is destroyed.
Obtained pointers are device pointers.

ncclGetLsaMultimemDevicePointer
--------------------------------

.. c:function:: ncclResult_t ncclGetLsaMultimemDevicePointer(ncclWindow_t window, size_t offset, void** outPtr)

Returns a multimem base pointer for the LSA team associated with the given window. This function provides host-side
access to the multimem memory functionality.

*window* is the NCCL window object (must not be NULL). *offset* is the byte offset within the window.
*outPtr* is the output parameter for the multimem pointer (must not be NULL).

This function requires LSA multimem support (multicast capability on the system). The window must be registered
with a communicator that supports symmetric memory, and the hardware must support NVLink SHARP multicast functionality.

.. note::
   If the system does not support multimem, the function returns ``ncclSuccess`` with ``*outPtr`` set to ``nullptr``.
   This allows applications to gracefully detect and handle the absence of multimem support without breaking
   the communicator. Users should check if the returned pointer is ``nullptr`` to determine availability.

Example:
  .. code:: C

    void* multimemPtr;
    ncclResult_t result = ncclGetLsaMultimemDevicePointer(window, 0, &multimemPtr);
    if (result == ncclSuccess) {
        if (multimemPtr != nullptr) {
            // Use multimemPtr for multimem operations
        } else {
            // Multimem not supported, use fallback approach
        }
    }

ncclGetMultimemDevicePointer
----------------------------

.. c:function:: ncclResult_t ncclGetMultimemDevicePointer(ncclWindow_t window, size_t offset, ncclMultimemHandle multimem, void** outPtr)

Returns a multimem base pointer using a provided multimem handle instead of the window's internal multimem.
This function enables using external or custom multimem handles for pointer calculation.

*window* is the NCCL window object (must not be NULL). *offset* is the byte offset within the window.
*multimem* is the multimem handle containing the multimem base pointer (multimem.mcBasePtr must not be NULL).
*outPtr* is the output parameter for the multimem pointer (must not be NULL).

This function requires LSA multimem support (multicast capability on the system).

.. note::
   If the system does not support multimem, the function returns ``ncclSuccess`` with ``*outPtr`` set to ``nullptr``.
   The function validates that ``multimem.mcBasePtr`` is not nullptr before proceeding.

Example:
  .. code:: C

    // Get multimem handle from device communicator setup
    ncclMultimemHandle customHandle;
    // ... (obtain handle)

    void* multimemPtr;
    ncclResult_t result = ncclGetMultimemDevicePointer(window, 0, customHandle, &multimemPtr);
    if (result == ncclSuccess) {
        if (multimemPtr != nullptr) {
            // Use multimemPtr for multimem operations with custom handle
        } else {
            // Multimem not supported, use fallback approach
        }
    }

ncclGetLsaDevicePointer
-----------------------

.. c:function:: ncclResult_t ncclGetLsaDevicePointer(ncclWindow_t window, size_t offset, int lsaRank, void** outPtr)

Returns a load/store accessible pointer to the memory buffer of a specific LSA peer within the window. This function
provides host-side access to LSA pointer functionality using LSA rank directly.

*window* is the NCCL window object (must not be NULL). *offset* is the byte offset within the window
(must be >= 0 and < window size).
*lsaRank* is the LSA rank of the target peer (must be >= 0 and < LSA team size).
*outPtr* is the output parameter for the LSA pointer (must not be NULL).

On success, ``ncclSuccess`` is returned and the LSA pointer is returned in ``outPtr``.


The window must be registered with a communicator that supports LSA. The LSA rank must be within the valid range
for the LSA team, and the target peer must be load/store accessible (P2P connectivity required).

Example:
  .. code:: C

    void* lsaPtr;
    ncclResult_t result = ncclGetLsaDevicePointer(window, 0, 1, &lsaPtr);
    if (result == ncclSuccess) {
        // Use lsaPtr to access LSA peer 1's memory
    }

ncclGetPeerDevicePointer
-------------------------

.. c:function:: ncclResult_t ncclGetPeerDevicePointer(ncclWindow_t window, size_t offset, int peer, void** outPtr)

Returns a load/store accessible pointer to the memory buffer of a specific world rank peer within the window.
This function converts world rank to LSA rank internally and provides host-side access to peer pointer functionality.

*window* is the NCCL window object (must not be NULL). *offset* is the byte offset within the window.
*peer* is the world rank of the target peer (must be >= 0 and < communicator size).
*outPtr* is the output parameter for the peer pointer (must not be NULL).

On success, ``ncclSuccess`` is returned and the peer pointer is returned in ``outPtr``.


If the peer is not reachable via LSA (not in LSA team), ``outPtr`` is set to NULL and ``ncclSuccess`` is returned.
This matches the behavior of the device-side ``ncclGetPeerPointer`` function.

The window must be registered with a communicator that supports LSA. The peer rank must be within the valid range
for the communicator, and the target peer must be load/store accessible (P2P connectivity required).

Example:
  .. code:: C

    void* peerPtr;
    ncclResult_t result = ncclGetPeerDevicePointer(window, 0, 2, &peerPtr);
    if (result == ncclSuccess) {
        if (peerPtr != NULL) {
            // Use peerPtr to access world rank 2's memory
        } else {
            // Peer 2 is not reachable via LSA
        }
    }

GIN
===

GIN is supported since NCCL 2.28.7.

ncclGin
-------

.. cpp:class:: ncclGin

 A class encompassing major elements of the GIN support.

 .. cpp:function:: ncclGin(ncclDevComm const& comm, int contextIndex)

  Initializes a new ``ncclGin`` object. *comm* is the device communicator created using :c:func:`ncclDevCommCreate`.
  *contextIndex* is the index of the GIN context -- a network communication channel.  Using multiple GIN contexts allows
  the implementation to spread traffic onto multiple connections, avoiding locking and bottlenecks.  Therefore,
  performance-oriented kernels should cycle among the available contexts to improve resource utilization (the number of
  available contexts is available via :c:macro:`ginContextCount`).

 .. cpp:function:: void put(ncclTeam team, int peer, ncclWindow_t dstWnd, size_t dstOffset, ncclWindow_t srcWnd, \
    size_t srcOffset, size_t bytes, \
    RemoteAction remoteAction, LocalAction localAction, Coop coop, DescriptorSmem descriptor, \
                    cuda::thread_scope alreadyReleased, cuda::thread_scope expected_scope)

  Schedules a device-initiated, one-sided data transfer operation from a local buffer to a remote buffer on a peer.

  *peer* is a rank within *team* (see :ref:`devapi_teams`); it may refer to the local rank (a loopback).  The destination
  and source buffers are each specified using the window (*dstWnd*, *srcWnd*) and a byte-based offset (*dstOffset*,
  *srcOffset*).  *bytes* specifies the data transfer count in bytes.

  Arguments beyond the first seven are optional.  *remoteAction* and *localAction* specify actions
  to undertake on the destination peer and on the local rank when the payload has been settled and the input has been
  consumed (respectively).  They default to ``ncclGin_None`` (no action); other options include
  ``ncclGin_Signal{Inc|Add}`` (for *remoteAction*) and ``ncclGin_CounterInc`` (for *localAction*); see
  :ref:`devapi_signals` below for more details.  *coop* indicates the set of threads participating in this operation (see
  :ref:`devapi_coops`); it defaults to ``ncclCoopThread`` (a single device thread), which is the recommended model.

  The visibility of the signal on the destination peer implies the visibility of the put data it is attached to *and all
  the preceding puts to the same peer, provided that they were issued using the same GIN context*.

  The API also defines an alternative, "convenience" variant of this method that uses ``ncclSymPtr`` types to specify the
  buffers and expects size to be conveyed in terms of the number of elements instead of the byte count.  There are also
  two ``putValue`` variants that take a single element at a time (no greater than eight bytes), passed by value.

 .. cpp:function:: void flush(Coop coop, cuda::memory_order ord = cuda::memory_order_acquire)

  Ensures that all the pending transfer operations scheduled by any threads of *coop* are locally consumed, meaning that
  their source buffers are safe to reuse.  Makes no claims regarding the completion status on the remote peer(s).

.. _devapi_signals:

Signals and Counters
--------------------

.. cpp:type:: ncclGinSignal_t

Signals are used to trigger actions on remote peers, most commonly on the completion of a :cpp:func:`ncclGin::put` operation.  They each
have a 64-bit integer value associated with them that can be manipulated atomically.

.. cpp:class:: ncclGin_SignalAdd

 .. cpp:member:: ncclGinSignal_t signal
 .. cpp:member:: uint64_t value

.. cpp:class:: ncclGin_SignalInc

 .. cpp:member:: ncclGinSignal_t signal

These objects can be passed as the *remoteAction* arguments of methods such as :cpp:func:`ncclGin::put` and :cpp:func:`ncclGin::signal` to describe the
actions to perform on the peer on receipt -- in this case, increase the value of a *signal* specified by
index. ``ncclGin_SignalInc{signalIdx}`` is functionally equivalent to ``ncclGin_SignalAdd{signalIdx, 1}``; however, it
may not be mixed with other signal-modifying operations without an intervening signal reset (see below).  Signal values
use "rolling" comparison logic to ensure that an unsigned overflow maintains the property of ``x < x + 1``.

**Signal methods of ncclGin:**

.. cpp:function:: void ncclGin::signal(ncclTeam team, int peer, RemoteAction remoteAction, Coop coop, \
        DescriptorSmem descriptor, cuda::thread_scope alreadyReleased, \
        cuda::thread_scope expected_scope)

.. cpp:function:: uint64_t ncclGin::readSignal(ncclGinSignal_t signal, int bits=64, cuda::memory_order ord = cuda::memory_order_acquire)
.. cpp:function:: void ncclGin::waitSignal(Coop coop, ncclGinSignal_t signal, uint64_t least, int bits=64, cuda::memory_order ord = cuda::memory_order_acquire)
.. cpp:function:: void ncclGin::resetSignal(ncclGinSignal_t signal)

These are signal-specific methods of :cpp:class:`ncclGin`.  :cpp:func:`ncclGin::signal` implements an explicit signal notification without
an accompanying data transfer operation; it takes a subset of arguments of :cpp:func:`ncclGin::put`.  :cpp:func:`ncclGin::readSignal` returns the
bottom *bits* of the value of the *signal*.  :cpp:func:`ncclGin::waitSignal` waits for the bottom *bits* of the *signal* value to meet
or exceed *least*.  Finally, :cpp:func:`ncclGin::resetSignal` resets the *signal* value to ``0`` (this method may not race with
concurrent modifications to the signal).

.. cpp:type:: ncclGinCounter_t

Counters are used to trigger actions on the local rank; as such, they are complementary to signals, which are meant for
remote actions.  Like signals, they use "rolling" comparison logic, but they are limited to storing values of at most 56
bits.

.. cpp:class:: ncclGin_CounterInc

 .. cpp:member:: ncclGinCounter_t counter

This object can be passed as the *localAction* argument of methods such as :cpp:func:`ncclGin::put`.  It is the only action
defined for counters.

**Counter methods of ncclGin:**

.. cpp:function:: uint64_t ncclGin::readCounter(ncclGinCounter_t counter, int bits=56, cuda::memory_order ord = cuda::memory_order_acquire)
.. cpp:function:: void ncclGin::waitCounter(Coop coop, ncclGinCounter_t counter, uint64_t least, int bits=56, cuda::memory_order ord = cuda::memory_order_acquire)
.. cpp:function:: void ncclGin::resetCounter(ncclGinCounter_t counter)

These are counter-specific methods of :cpp:class:`ncclGin` and they are functionally equivalent to their signal
counterparts discussed above.

ncclGinBarrierSession
---------------------

.. cpp:class:: template<typename Coop> ncclGinBarrierSession

 A class representing a network barrier session.

 .. cpp:function:: ncclGinBarrierSession(Coop coop, ncclGin gin, ncclTeamTagRail tag, uint32_t index)

  Initializes a new network barrier session.  *coop* represents a cooperative group (see :ref:`devapi_coops`).  *gin* is
  a previously initialized :cpp:class:`ncclGin` object.  *ncclTeamTagRail* indicates that the barrier will apply to all
  peers on the same rail as the local rank (see :ref:`devapi_teams`).  *index* identifies the underlying barrier to use
  (it should be different for each *coop*; typically set to ``blockIdx.x`` to ensure uniqueness between CTAs).

 .. cpp:function:: ncclGinBarrierSession(Coop coop, ncclGin gin, ncclTeam team, ncclGinBarrierHandle handle, uint32_t index)

  Initializes a new network barrier session.  This is the general-purpose variant to be used, e.g., when communicating
  with ranks from the world team (see :ref:`devapi_teams`), whereas the previous variant was specific to the rail team.
  This variant expects *team* to be passed as an argument, and also takes an extra *handle* argument indicating the
  location of the underlying barriers (typically set to the ``railGinBarrier`` field of the device communicator).

 .. cpp:function:: void sync(Coop coop, cuda::memory_order order, ncclGinFenceLevel fence)

  Synchronizes all threads of all team members that participate in the barrier session. ``ncclGinFenceLevel::Relaxed`` is
  the only defined value for *fence* for now.
