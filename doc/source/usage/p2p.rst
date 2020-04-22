.. _point-to-point:

****************************
Point-to-point communication
****************************

(Since NCCL 2.7)
Point-to-point communication can be used to express any communication pattern between ranks.
Any point-to-point communication needs two NCCL calls : a call to :c:func:`ncclSend` on one
rank and a corresponding :c:func:`ncclRecv` on the other rank, with the same count and data
type.

Multiple calls to :c:func:`ncclSend` and :c:func:`ncclRecv` and targeting different peers
can be fused together with :c:func:`ncclGroupStart` and :c:func:`ncclGroupEnd` to form more
complex communication patterns such as one-to-all (scatter), all-to-one (gather),
all-to-all or communication with neighbors in a N-dimensional space.

Point-to-point calls within a group will be blocking until that group of calls completes,
but calls within a group can be seen as progressing independently, hence should never block
each other. It is therefore important to merge calls that need to progress concurrently to
avoid deadlocks.

Below are a few examples of classic point-to-point communication patterns used by parallel
applications using MPI. NCCL semantics allow for all variants with different sizes, 
datatypes, and buffers, per rank.

Sendrecv
--------

In MPI terms, a sendrecv operation is when two ranks exchange data, both sending and receiving
at the same time. This can be done by merging both ncclSend and ncclRecv calls into one :

.. code:: C

 ncclGroupStart();
 ncclSend(sendbuff, sendcount, sendtype, peer, comm, stream);
 ncclRecv(recvbuff, recvcount, recvtype, peer, comm, stream);
 ncclGroupEnd();

One-to-all (scatter)
--------------------

A one-to-all operation from a ``root`` rank can be expressed by merging all send and receive
operations in a group :

.. code:: C

 ncclGroupStart();
 if (rank == root) {
   for (int r=0; r<nranks; r++)
     ncclSend(sendbuff[r], size, type, r, comm, stream);
 }
 ncclRecv(recvbuff, size, type, root, comm, stream);
 ncclGroupEnd();

All-to-one (gather)
-------------------

Similarly, an all-to-one operations to a ``root`` rank would be implemented this way :

.. code:: C

 ncclGroupStart();
 if (rank == root) {
   for (int r=0; r<nranks; r++)
     ncclRecv(recvbuff[r], size, type, r, comm, stream);
 }
 ncclSend(sendbuff, size, type, root, comm, stream);
 ncclGroupEnd();

All-to-all
----------

An all-to-all operation would be a merged loop of send/recv operations
to/from all peers :

.. code:: C

 ncclGroupStart();
 for (int r=0; r<nranks; r++) {
   ncclSend(sendbuff[r], sendcount, sendtype, r, comm, stream);
   ncclRecv(recvbuff[r], recvcount, recvtype, r, comm, stream);
 }
 ncclGroupEnd();

Neighbor exchange
-----------------

Finally, exchanging data with neighbors in an N-dimensions space could be done
with :

.. code:: C

 ncclGroupStart();
 for (int d=0; d<ndims; d++) {
   ncclSend(sendbuff[d], sendcount, sendtype, next[d], comm, stream);
   ncclRecv(recvbuff[d], recvcount, recvtype, prev[d], comm, stream);
 }
 ncclGroupEnd();

