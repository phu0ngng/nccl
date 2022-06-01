*****
Types
*****

The following types are used by the NCCL library.  

ncclComm_t
----------

.. c:type:: ncclComm_t

 NCCL communicator. Points to an opaque structure inside NCCL.

ncclResult_t
------------

.. c:type:: ncclResult_t 

 Return values for all NCCL functions. Possible values are :
 
 .. c:macro:: ncclSuccess

   (``0``)
   Function succeeded.
 .. c:macro:: ncclUnhandledCudaError

   (``1``)
   A call to a CUDA function failed.

 .. c:macro:: ncclSystemError

   (``2``)
   A call to the system failed.

 .. c:macro:: ncclInternalError

   (``3``)
   An internal check failed. This is either a bug in NCCL or due to memory corruption.

 .. c:macro:: ncclInvalidArgument

   (``4``)
   One argument has an invalid value.

 .. c:macro:: ncclInvalidUsage

   (``5``)
   The call to NCCL is incorrect. This is usually reflecting a programming error.

 .. c:macro:: ncclRemoteError

   (``6``)
   A call failed possibly due to a network error or a remote process exiting prematurely.

 Whenever a function returns an error (not ncclSuccess), NCCL should print a more detailed message when the environment variable :ref:`NCCL_DEBUG` is set to "WARN".

ncclDataType_t
--------------

.. c:type:: ncclDataType_t

 NCCL defines the following integral and floating data-types.
 
 .. c:macro:: ncclInt8

  Signed 8-bits integer

 .. c:macro:: ncclChar

  Signed 8-bits integer

 .. c:macro:: ncclUint8

  Unsigned 8-bits integer

 .. c:macro:: ncclInt32

  Signed 32-bits integer

 .. c:macro:: ncclInt

  Signed 32-bits integer

 .. c:macro:: ncclUint32

  Unsigned 32-bits integer

 .. c:macro:: ncclInt64

  Signed 64-bits integer

 .. c:macro:: ncclUint64

  Unsigned 64-bits integer

 .. c:macro:: ncclFloat16

  16-bits floating point number (half precision)

 .. c:macro:: ncclHalf

  16-bits floating point number (half precision)

 .. c:macro:: ncclFloat32

  32-bits floating point number (single precision)

 .. c:macro:: ncclFloat

  32-bits floating point number (single precision)

 .. c:macro:: ncclFloat64

  64-bits floating point number (double precision)

 .. c:macro:: ncclDouble

  64-bits floating point number (double precision)

 .. c:macro:: ncclBfloat16

  16-bits floating point number (truncated precision in bfloat16 format, CUDA 11 or later)


ncclRedOp_t
-----------

.. c:type:: ncclRedOp_t

 Defines the reduction operation.

 .. c:macro:: ncclSum

  Perform a sum (+) operation

 .. c:macro:: ncclProd

  Perform a product (*) operation

 .. c:macro:: ncclMin

  Perform a min operation

 .. c:macro:: ncclMax

 Perform a max operation

 .. c:macro:: ncclAvg

 Perform an average operation, i.e. a sum across all ranks, divided by the number of ranks.


ncclScalarResidence_t
---------------------

.. c:type:: ncclScalarResidence_t

 Indicates where (memory space) scalar arguments reside and when they can be
 dereferenced.

 .. c:macro:: ncclScalarHostImmediate

  The scalar resides in host memory and should be derefenced in the most immediate
  way.

 .. c:macro:: ncclScalarDevice

  The scalar resides on device visible memory and should be dereferenced once
  needed.
