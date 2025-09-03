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

 Rank and size within the LSA team (the subset of communicator ranks that are load/store accessible).  For now these
 hold the same values as the first group, but things will change with the addition of networking support.

ncclDevCommCreate
-----------------

.. c:function:: ncclResult_t ncclDevCommCreate(ncclComm_t comm, struct ncclDevCommRequirements const* reqs, struct ncclDevComm* outDevComm)

Creates a new device communicator (see :c:type:`ncclDevComm`) corresponding to the supplied host-side communicator
*comm*.  The result is returned in the *outDevComm* buffer (which needs to be supplied by the caller).  The caller needs
to also provide a filled-in list of requirements via the *reqs* argument (see :c:type:`ncclDevCommRequirements`); the
function will allocate any needed resources to meet them.  The function can fail and return an error code if the
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

 .. c:macro:: ncclLsaBarrierSession(Coop coop, ncclDevComm const& comm, ncclTeamTagLsa, uint32_t index, bool multimem=false)

 Initializes a new memory barrier session.  *coop* represents a cooperative group (typically ``ncclCoopCta()`` for all
 threads within the current CTA).  *comm* is the device communicator created using :c:func:`ncclDevCommCreate`.
 *ncclTeamTagLsa* is here to indicate which subset of ranks the barrier will apply to.  The identifier of the underlying
 barrier to use is provided by *index* (it should be different for each *coop*; typically set to ``blockIdx.x`` to
 ensure uniqueness between CTAs).  *multimem* requests a hardware-accelerated implementation using memory multicast.

 .. c:macro:: void arrive(Coop, cuda::memory_order order)

 Signals the arrival of the thread at the barrier session.

 .. c:macro:: void wait(Coop, cuda::memory_order order)

 Blocks until all threads arrive at the barrier session.

 .. c:macro:: void sync(Coop, cuda::memory_order order)

 Synchronizes all threads that participate in the barrier session (combines ``arrive`` and ``wait``).

ncclGetPeerPointer
------------------

.. c:function:: void* ncclGetPeerPointer(ncclWindow_t w, size_t offset, int peer)

Returns a load/store accessible pointer to the memory buffer of device *peer* within the window *w*.  *offset* is
byte-based.  *peer* is a rank index within the world team (the rank within the communicator that was used when creating
the window -- see :c:func:`ncclCommWindowRegister`).  This function will return NULL if the *peer* is not within the LSA
team.

ncclGetLsaPointer
-----------------

.. c:function:: void* ncclGetLsaPointer(ncclWindow_t w, size_t offset, int lsaPeer)

Returns a load/store accessible pointer to the memory buffer of device *lsaPeer* within the window *w*.  *offset* is
byte-based.  This is similar to :c:func:`ncclGetPeerPointer`, but here *lsaPeer* is a rank index with the LSA team (the
subset of communicator ranks that are load/store accessible).  This is only a theoretical distinction for now but it
will become significant when the networking support for symmetric kernels is complete.

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
