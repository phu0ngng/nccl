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

.. _cta_policy_flags:

NCCL Communicator CTA Policy Flags
----------------------------------

.. c:macro:: NCCL_CTA_POLICY_DEFAULT

  Use the default CTA policy for NCCL communicator. In this policy, NCCL will automatically adjust resource usage and achieve
  maximal performance. This policy is suitable for most applications.

.. c:macro:: NCCL_CTA_POLICY_EFFICIENCY

  Use the CTA efficiency policy for NCCL communicator. In this policy, NCCL will optimize CTA usage and use minimal
  number of CTAs to achieve the decent performance when possible. This policy is suitable for applications which require
  better compute and communication overlap.
