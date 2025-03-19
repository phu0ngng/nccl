*************
Thread Safety
*************

NCCL primitives are generally not thread-safe, however, they are reentrant. Under multi-thread environment, it is not allowed
to issue NCCL operations to a single communicator in parallel with multiple threads; it is not safe to issue NCCL operations
in parallel to independent communicators located on the same device with multiple threads (see :ref:`multi-thread-concurrent-usage`).
If the child communicator shares the resources with the parent communicator (i.e., :ref:`ncclconfig` by `splitShare`), it is not
allowed to issue NCCL operations to the child and parent communicators in parallel.
