# NCCL Replay Tool

## Building the replay tool
Build NCCL + tests with MPI:
`make -j test.build MPI=1`

The replay tool depends on MPI for bootstrapping and initialization.

## Generating a trace
Set the following environment variables before running your NCCL application to generate the proper trace file per-rank:
``NCCL_DEBUG_SUBSYS=CALL
NCCL_DEBUG=TRACE
NCCL_DEBUG_FILE=filename.%h.%p``

The ``NCCL_DEBUG_FILE`` variable directs the NCCL debug logging output to a file.
The filename format can be set to *filename.%h.%p* where *%h* is replaced with the
hostname and *%p* is replaced with the process PID. This does not accept the ``~`` character as part of the path, please convert to a relative path first.

### Example
``salloc -N1 -n8 -pP100 mpirun -x NCCL_DEBUG_SUBSYS=CALL -x NCCL_DEBUG=TRACE  -x NCCL_DEBUG_FILE=all_reduce.%h.%p ./build/test/perf/all_reduce_perf -w5 -n5 -b8 -e10M -f2 -d all``

## Understanding a trace file

Here is part of the generated trace file from that example:

``gc01:41073:41073 NCCL CALL ncclGetUniqueId(0x521631e3a56ecc2c)
gc01:41073:41073 NCCL CALL ncclGroupStart()
gc01:41073:41073 [0] NCCL INFO NCCL version 2.16.3a0+cuda11.7
gc01:41073:41128 NCCL CALL ncclCommInitRank(0x55d8c12613d0, 16, 0x521631e3a56ecc2c, 0, 0)
gc01:41073:41073 NCCL CALL ncclGroupEnd()
gc01:41073:41073 NCCL CALL ncclAllReduce(7f5d92a00000,7f5d93400000,10485760,0,0,0,0x55d8c12613d0,0x55d8c09207b0)
gc01:41073:41073 NCCL CALL ncclAllReduce(7f5d92a00000,7f5d93400000,10485760,0,0,0,0x55d8c12613d0,0x55d8c09207b0)``

Each parsable line has the following format:
``<hostName>:<pid>:<tid> NCCL CALL <ncclCall>()``

Each call inside the trace has all the context the replay tool needs to correctly reproduce it:
``nccl<coll_name>(src_ptr, dest_ptr, element_count, element_datatype, reduction_operation, root, comm_pointer, stream_pointer)``

`element_datatype` is of type `ncclDataType_t`, and `reduction_operation` is of type `ncclRedOp_t` except in case of custom reduction operations.

## Replaying a trace
First, you must concatenate the per-rank trace files into a single file.

### Example
``cat all_reduce* > full_all_reduce_trace.txt``

Then run the replay tool pointing to your file using the same node count and ranks per-node:

### Example
``salloc -N1 -n8 -pP100 mpirun ./build/test/replay/replay full_all_reduce_trace.txt``

## Options

### Verbose
``-v``

### Progress Thread
``-p``

This will spin up a thread that prints out the replay progress every second. I find it useful to know that the replay tool isn't hanging and usually run with it on every time.

### Force fit trace
``-x``

This will take a trace recorded on a large node count and attempts to convert it to the supplied dimensions of your MPI run.  This will work well for collectives - it simply scales down nranks of each communicator and converts the root of collectives to a valid rank if applicable.  For point-to-point communications, it will skip replaying any P2P operations whose peer rank is out of the bounds of this replay. Note that you must have the same number of ranks per-node for this to work.

### Force one-sized data ops
``-f``

This converts nelems of each NCCL data operation to either 1 or nranks, depending on which is the minimum valid size. This could be used to measure latency or quickly test a communications pattern.

### Disable trace pre-check
``-d``

The replay tool pre-processes and validates traces before dissemenating and processing work. It checks that ranks specified are in-bounds and that every rank in a communicator group has a matching collective call.  If this isn't the case, the replay tool will simply throw an error and refuse to run.  This option allows the user to force skip this step if they still want to try to run their trace.

## Ordering

The replay tool will order the operations submitted with respect to:
1. Per-communicator ordering
2. Per-stream ordering
3. Per-device ordering

These are usually all the same, but extra sequencing is forced in case the traced application is using external synchronization that NCCL can't see.  Beyond that, NCCL will queue as many operations as possible.

## Data Corruption Checking

The replay tool uses NCCL's internal verifiable device kernel to check the correctness of transferred data after every operation, including checking the result of reductions.  This has a modest performance overhead and cannot be disabled.