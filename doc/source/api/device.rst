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

 .. c:macro:: rank
 .. c:macro:: nRanks

 Rank and size of the communicator.

 .. c:macro:: lsaRank
 .. c:macro:: lsaSize

 Rank within the local LSA team and its size (see :ref:`devapi_teams`).

 .. c:macro:: ginContextCount

 The number of supported GIN contexts (see :c:macro:`ncclGin`; available since NCCL 2.28.7).

ncclDevCommCreate
-----------------

.. c:function:: ncclResult_t ncclDevCommCreate(ncclComm_t comm, struct ncclDevCommRequirements const* reqs, struct ncclDevComm* outDevComm)

Creates a new device communicator (see :c:type:`ncclDevComm`) corresponding to the supplied host-side communicator
*comm*.  The result is returned in the *outDevComm* buffer (which needs to be supplied by the caller).  The caller needs
to also provide a filled-in list of requirements via the *reqs* argument (see :c:type:`ncclDevCommRequirements`); the
function will allocate any necessary resources to meet them.  The function can fail and return an error code if the
communicator does not support symmetric memory or if the list of requirements cannot be met (e.g., if the multimem
capability is requested on a system lacking the necessary hardware support).

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
:c:type:`ncclDevComm`).

 .. c:macro:: lsaMultimem

 Specifies whether multimem support is required for all LSA ranks.

 .. c:macro:: lsaBarrierCount

 Specifies the number of memory barriers to allocate (see :c:type:`ncclLsaBarrierSession`).

 .. c:macro:: railGinBarrierCount

 Specifies the number of network barriers to allocate (see :c:type:`ncclGinBarrierSession`; available since NCCL
 2.28.7).

 .. c:macro:: barrierCount

 Specifies the minimum number for both the memory and network barriers (see above; available since NCCL 2.28.7).

 .. c:macro:: ginSignalCount

 Specifies the number of network signals to allocate (see :c:type:`ncclGinSignal_t`; available since NCCL 2.28.7).

 .. c:macro:: ginCounterCount

 Specifies the number of network counters to allocate (see :c:type:`ncclGinCounter_t`; available since NCCL 2.28.7).

 .. c:macro:: resourceRequirementsList

 Specifies a list of resource requirements.  This is best set to NULL for now.

 .. c:macro:: teamRequirementsList

 Specifies a list of requirements for particular teams.  This is best set to NULL for now.


LSA
===

All functionality described from this point on is available on the device side only.

ncclLsaBarrierSession
---------------------

.. c:type:: ncclLsaBarrierSession

A class representing a memory barrier session.

 .. c:function:: ncclLsaBarrierSession(Coop coop, ncclDevComm const& comm, ncclTeamTagLsa, uint32_t index, bool multimem=false)

 Initializes a new memory barrier session.  *coop* represents a cooperative group (see :ref:`devapi_teams`).
 *comm* is the device communicator created using :c:func:`ncclDevCommCreate`.
 *ncclTeamTagLsa* is here to indicate which subset of ranks the barrier will apply to.  The identifier of the underlying
 barrier to use is provided by *index* (it should be different for each *coop*; typically set to ``blockIdx.x`` to
 ensure uniqueness between CTAs).  *multimem* requests a hardware-accelerated implementation using memory multicast.

 .. c:function:: void arrive(Coop, cuda::memory_order order)

 Signals the arrival of the thread at the barrier session.

 .. c:function:: void wait(Coop, cuda::memory_order order)

 Blocks until all threads of all team members arrive at the barrier session.

 .. c:function:: void sync(Coop, cuda::memory_order order)

 Synchronizes all threads of all team members that participate in the barrier session (combines :c:func:`arrive` and
 :c:func:`wait`).

ncclGetPeerPointer
------------------

.. c:function:: void* ncclGetPeerPointer(ncclWindow_t w, size_t offset, int peer)

Returns a load/store accessible pointer to the memory buffer of device *peer* within the window *w*.  *offset* is
byte-based.  *peer* is a rank index within the world team (see :ref:`devapi_teams`).  This function will return NULL if
the *peer* is not within the LSA team.

ncclGetLsaPointer
-----------------

.. c:function:: void* ncclGetLsaPointer(ncclWindow_t w, size_t offset, int lsaPeer)

Returns a load/store accessible pointer to the memory buffer of device *lsaPeer* within the window *w*.  *offset* is
byte-based.  This is similar to :c:func:`ncclGetPeerPointer`, but here *lsaPeer* is a rank index with the LSA team (see
:ref:`devapi_teams`).

ncclGetLocalPointer
-------------------

.. c:function:: void* ncclGetLocalPointer(ncclWindow_t w, size_t offset)

Returns a load-store accessible pointer to the memory buffer of the current device within the window *w*.  *offset* is
byte-based.  This is just a shortcut version of :c:func:`ncclGetPeerPointer` with *devComm.rank* as *peer*, or :c:func:`ncclGetLsaPointer` with *devComm.lsaRank* as *lsaPeer*.

Multimem
========

ncclGetLsaMultimemPointer
-------------------------

.. c:function:: void* ncclGetLsaMultimemPointer(ncclWindow_t w, size_t offset, ncclDevComm const& devComm)

Returns a multicast memory pointer associated with the window *w* and device communicator *devComm*.  *offset*
is byte-based.  Availability of multicast memory is hardware-dependent.

GIN
===

GIN is supported since NCCL 2.28.7.

ncclGin
-------

.. c:type:: ncclGin

A class encompassing major elements of the GIN support.

 .. c:function:: ncclGin(ncclDevComm const& comm, int contextIndex)

 Initializes a new ``ncclGin`` object. *comm* is the device communicator created using :c:func:`ncclDevCommCreate`.
 *contextIndex* is the index of the GIN context -- a network communication channel.  Using multiple GIN contexts allows
 the implementation to spread traffic onto multiple connections, avoiding locking and bottlenecks.  Therefore,
 performance-oriented kernels should cycle among the available contexts to improve resource utilization (the number of
 available contexts is available via :c:macro:`ginContextCount`).

 .. c:function:: void put(ncclTeam team, int peer, ncclWindow_t dstWnd, size_t dstOffset, ncclWindow_t srcWnd, size_t srcOffset, size_t bytes, [...])

 Schedules a device-initiated, one-sided data transfer operation from a local buffer to a remote buffer on a peer.

 *peer* is a rank within *team* (see :ref:`devapi_teams`); it may refer to the local rank (a loopback).  The destination
 and source buffers are each specified using the window (*dstWnd*, *srcWnd*) and a byte-based offset (*dstOffset*,
 *srcOffset*).  *size* specifies the data transfer count in bytes.

 Arguments beyond that are optional; we focus here on the first three.  *remoteAction* and *localAction* specify actions
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

 .. c:function:: void flush(Coop coop, cuda::memory_order ord = cuda::memory_order_acquire)

 Ensures that all the pending transfer operations scheduled by any threads of *coop* are locally consumed, meaning that
 their source buffers are safe to reuse.  Makes no claims regarding the completion status on the remote peer(s).

.. _devapi_signals:

Signals and Counters
--------------------

.. c:type:: ncclGinSignal_t

Signals are used to trigger actions on remote peers, most commonly on the completion of a ``put`` operation.  They each
have a 64-bit integer value associated with them that can be manipulated atomically.

 .. c:function:: ncclGin_SignalAdd { ncclGinSignal_t signal; uint64_t value; }
 .. c:function:: ncclGin_SignalInc { ncclGinSignal_t signal; }

 These objects can be passed as the *remoteAction* arguments of methods such as ``put`` and ``signal`` to describe the
 actions to perform on the peer on receipt -- in this case, increase the value of a *signal* specified by
 index. ``ncclGin_SignalInc{signalIdx}`` is functionally equivalent to ``ncclGin_SignalAdd{signalIdx, 1}``; however, it
 may not be mixed with other signal-modifying operations without an intervening signal reset (see below).  Signal values
 use "rolling" comparison logic to ensure that an unsigned overflow maintains the property of ``x < x + 1``.

 .. c:function:: void signal(ncclTeam team, int peer, RemoteAction remoteAction, Coop coop = ncclCoopThread(), [...])
 .. c:function:: uint64_t readSignal(ncclGinSignal_t signal, int bits=64, cuda::memory_order ord = cuda::memory_order_acquire)
 .. c:function:: void waitSignal(Coop coop, ncclGinSignal_t signal, uint64_t least, int bits=64, cuda::memory_order ord = cuda::memory_order_acquire)
 .. c:function:: void resetSignal(ncclGinSignal_t signal)

 These are signal-specific methods of :c:type:`ncclGin`.  ``signal`` implements an explicit signal notification without
 an accompanying data transfer operation; it takes a subset of arguments of :c:func:`put`.  ``readSignal`` returns the
 bottom *bits* of the value of the *signal*.  ``waitSignal`` waits for the bottom *bits* of the *signal* value to meet
 or exceed *least*.  Finally, ``resetSignal`` resets the *signal* value to ``0`` (this method may not race with
 concurrent modifications to the signal).

.. c:type:: ncclGinCounter_t

Counters are used to trigger actions on the local rank; as such, they are complementary to signals, which are meant for
remote actions.  Like signals, they use "rolling" comparison logic, but they are limited to storing values of at most 56
bits.

 .. c:type:: ncclGin_CounterInc { ncclGinCounter_t counter; }

 This object can be passed as the *localAction* argument of methods such as :c:func:`put`.  It is the only action
 defined for counters.

 .. c:function:: uint64_t readCounter(ncclGinCounter_t counter, int bits=56, cuda::memory_order ord = cuda::memory_order_acquire)
 .. c:function:: void waitCounter(Coop coop, ncclGinCounter_t counter, uint64_t least, int bits=56, cuda::memory_order ord = cuda::memory_order_acquire)
 .. c:function:: void resetCounter(ncclGinCounter_t counter)

 These are counter-specific methods of :c:type:`ncclGin` and they are functionally equivalent to their signal
 counterparts discussed above.

ncclGinBarrierSession
---------------------

.. c:type:: ncclGinBarrierSession

A class representing a network barrier session.

 .. c:function:: ncclGinBarrierSession(Coop coop, ncclGin gin, ncclTeamTagRail, uint32_t index)

 Initializes a new network barrier session.  *coop* represents a cooperative group (see :ref:`devapi_coops`).  *gin* is
 a previously initialized :c:type:`ncclGin` object.  *ncclTeamTagRail* indicates that the barrier will apply to all
 peers on the same rail as the local rank (see :ref:`devapi_teams`).  *index* identifies the underlying barrier to use
 (it should be different for each *coop*; typically set to ``blockIdx.x`` to ensure uniqueness between CTAs).

 .. c:function:: ncclGinBarrierSession(Coop coop, ncclGin gin, ncclTeam team, ncclGinBarrierHandle handle, uint32_t index)

 Initializes a new network barrier session.  This is the general-purpose variant to be used, e.g., when communicating
 with ranks from the world team (see :ref:`devapi_teams`), whereas the previous variant was specific to the rail team.
 This variant expects *team* to be passed as an argument, and also takes an extra *handle* argument indicating the
 location of the underlying barriers (typically set to the ``railGinBarrier`` field of the device communicator).

 .. c:function:: void sync(Coop coop, cuda::memory_order order, ncclGinFenceLevel fence)

 Synchronizes all threads of all team members that participate in the barrier session. ``ncclGinFenceLevel::Relaxed`` is
 the only defined value for *fence* for now.
