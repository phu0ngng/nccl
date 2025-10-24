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
