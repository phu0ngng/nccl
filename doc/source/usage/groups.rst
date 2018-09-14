.. _group-calls:

***********
Group Calls
***********

Management Of Multiple GPUs From One Thread
-------------------------------------------

When a single thread is managing multiple devices, group semantics must be used.
This is because every NCCL call may have to block, waiting for other threads/ranks to arrive, before effectively posting the NCCL operation on the given stream. Hence, a simple loop on multiple devices like shown below could block on the first call waiting for the other ones:

.. code:: C

 for (int i=0; i<nLocalDevs; i++) {
   ncclAllReduce(..., comm[i], stream[i];
 }

To define that these calls are part of the same collective operation, ncclGroupStart and ncclGroupEnd should be used: 

.. code:: C

  ncclGroupStart();
  for (int i=0; i<nLocalDevs; i++) {
    ncclAllReduce(..., comm[i], stream[i];
  }
  ncclGroupEnd();

This will tell NCCL to treat all calls between ncclGroupStart and ncclGroupEnd as a single call to many devices. 

Caution: When called inside a group, stream operations (like ncclAllReduce) can return without having enqueued the
operation on the stream. Stream operations like cudaStreamSynchronize can therefore be called only after ncclGroupEnd
returns.


Note: Contrary to NCCL 1.x, there is no need to set the CUDA device before every NCCL communication call within a group,
but it is still needed when calling ncclCommInitRank within a group.

Related links:

* :c:func:`ncclGroupStart`
* :c:func:`ncclGroupEnd`

Aggregated Operations (2.2 and later)
-------------------------------------

The group semantics can also be used to have multiple collective operations performed within a single NCCL launch. This
is useful for reducing the launch overhead, in other words, latency, as it only occurs once for multiple operations.

Aggregation of collective operations can be done simply by having multiple calls to NCCL within a ncclGroupStart /
ncclGroupEnd section.

In the following example, we launch one broadcast and two allReduce operations together as a single NCCL launch.

.. code:: C

 ncclGroupStart();
 ncclBroadcast(sendbuff1, recvbuff1, count1, datatype, root, comm, stream);
 ncclAllReduce(sendbuff2, recvbuff2, count2, datatype, comm, stream);
 ncclAllReduce(sendbuff3, recvbuff3, count3, datatype, comm, stream);
 ncclGroupEnd();

It is not permitted to use different streams for a given NCCL communicator. This sequence is erroneous:

.. code:: C

 ncclGroupStart();
 ncclAllReduce(sendbuff1, recvbuff1, count1, comm, stream1);
 ncclAllReduce(sendbuff2, recvbuff2, count2, comm, stream2);
 ncclGroupEnd();

It is, however, permitted to combine aggregation with multi-GPU launch and use different communicators in a group launch
as shown in the Management Of Multiple GPUs From One Thread topic.  When combining multi-GPU launch and aggregation,
ncclGroupStart and ncclGroupEnd can be either used once or at each level. The following example groups the allReduce
operations from different layers and on multiple CUDA devices :

.. code:: C

 ncclGroupStart();
 for (int i=0; i<nlayers; i++) {
   ncclGroupStart();
   for (int g=0; g<ngpus; g++) {
     ncclAllReduce(sendbuffs[g]+offsets[i], recvbuffs[g]+offsets[i], counts[i], datatype[i], comms[g], streams[g]);
   }
   ncclGroupEnd();
 }
 ncclGroupEnd();

Note: The NCCL operation will only be started as a whole during the last call to ncclGroupEnd. The ncclGroupStart and
ncclGroupEnd calls within the for loop are not necessary and do nothing. Also, a given communicator comms[g] is always
used with the same stream streams[g].

Related links:

* :c:func:`ncclGroupStart`
* :c:func:`ncclGroupEnd`
