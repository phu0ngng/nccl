.. _api_flags:

************************
NCCL API Supported Flags
************************

The following show all flags which are supported by NCCL APIs.

.. _win_flags:

Window Registration Flags
-------------------------

.. c:macro:: NCCL_WIN_DEFAULT

 Register buffer into NCCL window with default behavior. The default behavior allows users to
 pass any offset to the buffer head address as the input of NCCL collective operations. However,
 this behavior can cause suboptimal performance in NCCL due to the asymmetric buffer usage.

.. c:macro:: NCCL_WIN_COLL_SYMMETRIC

 Register buffer into NCCL window, and users need to guarantee the offset to the buffer head address
 from all ranks must be equal when calling NCCL collective operations. It allows NCCL to operate
 buffer in a symmetric way and provide the best performance.
