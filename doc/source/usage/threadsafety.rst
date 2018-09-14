*************
Thread Safety
*************

NCCL primitives are generally not thread-safe, however, they are reentrant. Multiple threads should use separate communicator objects.


