.. _user_buffer_reg:

************************
User Buffer Registration
************************

User Buffer Registration is a feature that allows NCCL to directly send/receive/operate data through the user buffer without extra internal copy.
It can accelerate collectives and reduce the resource usage (e.g. #channel usage).

NCCL 2.19.x now supports user buffer registration for NVLink Switch; any NCCL collectives that support NVLS algorithm can utilize this feature.
To enable the *CUDA Graph based* buffer registration, users have to comply with several requirements:

 * The buffer is allocated through :c:func:`ncclMemAlloc()`.
 * Offset to the head address of the buffer is same for each rank in collectives.
 * The NCCL operation is launched on a stream captured by a CUDA graph.

Registered buffers will be deregistered when cuda graph is destroyed.

On the other hand, to enable the *Local based* buffer registration, users have to comply with the following requirements:

 * The buffer is allocated through :c:func:`ncclMemAlloc()`.
 * Register buffer with :c:func:`ncclCommRegister` before calling collectives.
 * Call NCCL collectives as usual but similarly keep the offset to the head address of the buffer same for each rank.

Registered buffers will be deregistered when users explicitly call :c:func:`ncclCommDeregister`.
