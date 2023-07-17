************************
User Buffer Registration
************************

User Buffer Registration is a feature that allows NCCL to directly send/receive/operate data through the user buffer without extra internal copy.
It can accelerate collectives and reduce the resource usage (e.g. #channel usage).

NCCL 2.19.x now supports user buffer registration for NVLink Switch; any NCCL collectives that support NVLS algorithm can utilize this feature.
To enable the *CUDA Graph based* buffer registration, users have to comply with several requirements:

 * The buffer is allocated through cuMem* API with CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR flag.
 * The head address and size of the buffer must be aligned to the returned granularity of cuMulticastGetGranularity API
   with CU_MULTICAST_GRANULARITY_RECOMMENDED flag.
 * Offset to the head address of the buffer is same for each rank in collectives.
 * The NCCL operation is launched on a stream captured by a CUDA graph.

Registered buffers will be deregistered when cuda graph is destroyed.
