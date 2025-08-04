# NCCL alltoall/gather/scatter api
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract
In this feature, we add the alltoall, gather and scatter host APIs and their basic implementation to NCCL.

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

Up to V2.27, NCCL library exposes no host-level collectives for alltoall, gather, or scatter. Users have to implement 
these communication patterns in the user space, using NCCL send/recv APIs. Native host APIs bring several key benefits. First, users do 
not need to implement these communication patterns in their application code. Second, NCCL core can perform further optimizations for these 
communication patterns. One example is building the symmetric kernel for alltoall to directly operate on registered memory. Another example 
is to use copy-engine to reduce the SM usage for these communication patterns. All of these optimizations rely on the native host APIs. 

This proposal adds NCCL native host APIs for alltoall, gather, and scatter operations. The initial implementation will mirror how users currently implement these operations using NCCL send/recv APIs in their application code, but will be integrated directly into the NCCL core using the underlying P2P transport mechanism. This provides a cleaner, more efficient implementation while maintaining the same functionality. Other optimizations, such as symmetric kernel implementations and copy-engine collectives for alltoall/scatter/gather, will be discussed in separate proposals.

### Requirements
1. **Common and standard APIs**
The NCCL host APIs of alltoall, gather and scatter should be common and matched the widely used APIs in other libraries, such as MPI.

2. **Functional on all existing platforms**
The new APIs should be fully functional on all existing platforms/systems, this includes systems with NVLs and networks.

3. **Implementations within NCCL core**
The implementation could leverage the mechanism of NCCL P2P transport and reuse the existing NCCL P2P kernels. As this is the safest approach to make sure the new APIs are fully functional on all existing platforms. However, the implementation should not be a simple wrapper of the existing NCCL send/recv APIs. Instead, the NCCL core should be modified such that it takes the new collective API tasks and then translates these tasks into a set of P2P tasks.

4. **Extensible for future optimizations**
The implementation should be extensible to support future optimizations. For example, the design should be able to integrate the symmetric kernel for these new APIs and the copy-engine collectives.


### NVbugs / Jira Tickets
https://nvbugspro.nvidia.com/bug/5273048
https://nvbugspro.nvidia.com/bug/5304035



</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Design</h2></summary>
<!-- ============================================================================================-->

### Proposed Design

#### 1. API definition

We propose the following APIs:

```c  
// Alltoall
NCCL_API(ncclResult_t, ncclAlltoAll, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream);

// Gather
NCCL_API(ncclResult_t, ncclGather, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm* comm, cudaStream_t stream);

// Scatter
NCCL_API(ncclResult_t, ncclScatter, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int root, ncclComm* comm, cudaStream_t stream);
```	

These APIs are designed to be similar to the existing NCCL APIs.

#### 2. Function wrapper for P2P task construction and configuration
We aim to implement the new APIs using the existing NCCL P2P transport mechanism with "one-shot" algorithm, e.g., one-to-all, all-to-one, etc. These new API calls will enter the enqueue and scheduling pipeline as collective tasks `ncclTaskColl` and then be translated into a series of P2P tasks `ncclTaskP2p`. The current codebase does not have a dedicated function for `ncclTaskP2p` construction and configuration, and the code logic is scattered. We plan to consolidate this logic into a dedicated wrapper function since the new communication APIs will need to configure `ncclTaskP2p` tasks multiple times.

```c
// Example function (extracted from taskAppend function) that:
// wraps the construction of ncclTaskP2p
// appends the task to planner->peers queue
// marks channels that need pre-connect

static void p2pTaskAppend(
    struct ncclComm* comm,
    ncclFunc_t coll,
    void* buff,
    size_t count,
    ncclDataType_t datatype,
    int peer) {

  struct ncclKernelPlanner *planner = &comm->planner;

  // Determine peer and basic parameters.
  const bool isSendNotRecv = (coll == ncclFuncSend);
  const ssize_t nBytes = count * ncclTypeSize(datatype);

  // Allocate and populate the P2P task structure.
  struct ncclTaskP2p* p2p = ncclMemoryPoolAlloc<struct ncclTaskP2p>(
      &comm->memPool_ncclTaskP2p, &comm->memPermanent);
  p2p->func   = coll;
  p2p->buff   = buff;
  p2p->count  = count;
  p2p->datatype = datatype;
  p2p->root   = peer;
  p2p->bytes  = nBytes;
  p2p->eActivationMask = __atomic_load_n(&ncclProfilerEventMask, __ATOMIC_RELAXED);

  // Enqueue onto the appropriate planner queue.
  ncclIntruQueueEnqueue(
    isSendNotRecv ? &planner->peers[peer].sendQueue : &planner->peers[peer].recvQueue,
    p2p);
  planner->nTasksP2p += 1;

  // Mark channels that need pre-connect. This closely mirrors the original
  // inline logic but is now consolidated here for reuse / clarity.

  ...
}

```

#### 3. Collective based on p2pTaskAppend
The collective can be expressed as a sequence of p2pTaskAppend function calls. 
Currently do not plan to create dedicated function wrapper for alltoall, scatter and gather. 

```c
// Example scatter implemented with p2pTaskAppend
if (coll == ncclFuncScatter) {
  size_t offset = 0;
  
  if (rank == root) {
    // Root sends different chunks to each rank
    // self-send handled down stream after calling p2pTaskAppend
    for (int r = 0; r < nRanks; r++) {
      void* buff = (void*)((char*)sendbuff + offset);
      p2pTaskAppend(comm, planner, ncclFuncSend, buff, count, datatype, r);
      offset += count * typeSize;
    }
  } else {
    // Non-root ranks receive their chunk from root
    p2pTaskAppend(comm, planner, ncclFuncRecv, (void*)recvbuff, count, datatype, root);
  }
}

```

#### 3. Transition from ncclTaskColl to ncclTaskP2p
NCCL core has different enqueue and schedule pipelines for P2P and collective operations because they require different preparations. For instance, the P2P pipeline might involve the registration of the send and recv buffers and the collective pipeline might involve operation binning. This means the transition from the collective tasks to P2P tasks should be placed in the right place. Here are several key points:

- The `ncclTaskColl` should be transferred into a series of `ncclTaskP2p` tasks within the `group.cc` or `enqueue.cc`.
- The transition should be placed early enough to avoid too much code changes down the enqueue and schedule pipelines.
- The transition should be extensible for implementation other than P2P transport. For instance, copy-engine implementation and symmetric kernel implementation.
- The transition from `ncclTaskColl` to `ncclTaskP2p` pipeline should be earlier than essential operations for P2P tasks, e.g., P2P setup with `ncclP2PPreconnectFunc`.

There are several options for the transition points:

- **Option 1: Transition at taskAppend**: This option proposes making the transition from collective to P2P tasks in the `taskAppend` function, which is the earliest point where both `ncclTaskColl` and `ncclTaskP2p` are constructed.

  - Advantages:
    - Early transition allows for cleaner pipeline handling
    - Natural place to construct P2P tasks since task creation already happens here
    
```c
// enqueue.cc
static ncclResult_t taskAppend(struct ncclComm* comm, struct ncclInfo* info) {
  struct ncclKernelPlanner *planner = &comm->planner;

  // If it is send/recv call
  if (info->coll == ncclFuncSend || info->coll == ncclFuncRecv) {
    // Must be in thread local group before tasks can be alloc'd in `comm->memScoped`.
    ncclGroupCommJoin(info->comm, ncclGroupTaskTypeCollective);

    // Replace existing code with p2pTaskAppend
    p2pTaskAppend(comm, planner, info->coll, (void*)info->recvbuff, info->count, info->datatype, info->root);
  } 
  // If it is collective call
  else {
    ...

    /* -------Added code section for alltoall, scatter and gather------------*/
    if (info->coll == ncclFuncAlltoall || info->coll == ncclFuncScatter || info->coll == ncclFuncGather) {
      // Collective implmented with p2pTaskAppend
    } 
    /*------------------------------------------------------------------------*/
    
    // Existing code that follows the normal ncclTaskColl path 
    else {
      ...  
    }
  }
  
  ...

  return ncclSuccess;
}
```

  - Challenges:
    - Would require moving CE and symmetric kernel implementation checks to `taskAppend` in the future
    - Cannot reliably detect if an operation is standalone or part of a group during `taskAppend` because `taskAppend` is called during `ncclEnqueueCheck`, before we have complete group operation context
    - Existing CE/symmetric implementations assume specific conditions (single node, one collective task in the group, no P2P tasks in the group) that can only be verified after full operation group is enqueued
    - Require checking when/how grouped ce/sym implementation will be supported or to refactorize the `ncclPrepareTasks` to serialize the ce/sym kernels if grouped sym/ce support is not in scope 

- **Option 2: Transition at ncclPrepareTasks**: 
  We could handle the transition in `ncclPrepareTasks`, where CE and symmetric kernel implementation checks currently happen. This function:
  1. Takes sorted collective tasks from `ncclTaskCollSorterDequeueAll`(&planner->collSorter)
  2. Enqueues them into `planner->collTaskQueue`
  
  To implement this approach, we would need to add the following in the `ncclPrepareTasks`:
  1. Add checks to identify which tasks should be converted to P2P tasks
  2. Convert applicable tasks to `ncclTaskP2p`
  3. Enqueue converted tasks to `planner->peers` instead of `collTaskQueue`
  4. Mark the channels that needs to be preconnected for peer connections

  However, this introduces a sequencing challenge - `ncclPrepareTasks` is called from `groupLaunch` after `ncclP2PPreconnectFunc` has already run. We would need to reorder the pipeline to run `ncclP2PPreconnectFunc` after `ncclPrepareTasks` to properly handle the converted P2P tasks.

```c
// enqueue.cc
ncclResult_t ncclPrepareTasks(struct ncclComm* comm, bool* algoNeedConnect, bool* needConnect, ncclSimInfo_t* simInfo) {
  struct ncclKernelPlanner* planner = &comm->planner;
  planner->persistent = ncclCudaGraphValid(planner->capturingGraph);
  // Tasks from the sorter come out ordered size descending.
  struct ncclTaskColl* task = ncclTaskCollSorterDequeueAll(&planner->collSorter);
  // Tasks are assembled by (fn,op,ty) size ascending.
  struct ncclTaskColl* tasksByFnOpTy[ncclNumFuncs*ncclNumDevRedOps*ncclNumTypes];
  memset(tasksByFnOpTy, 0, sizeof(tasksByFnOpTy));
  int fnOpTyIndices[ncclNumFuncs*ncclNumDevRedOps*ncclNumTypes];
  int fnOpTyCount = 0;

  // Exising code to check if CE/Sym is implemented, if yes, it will return
  ...

  /*---- Added code section if the task is alltoall, scatter and gather-----*/
  if (task->func == ncclFuncAlltoall || task->func == ncclFuncScatter || task->func == ncclFuncGather)
  {
    // collective implmented with p2pTaskAppend
    return ncclSuccess;
  }
  /*-------------------------------------------------------------------------*/

  // Existing code for ncclTaskColl
  ... 
}
```

```c
// group.cc
// Modified the order of pre-connect for P2P tasks
static ncclResult_t groupLaunch(struct ncclAsyncJob *job_, ncclSimInfo_t* simInfo = NULL) {
  ncclResult_t ret = ncclSuccess;
  struct ncclGroupJob *gjob = (struct ncclGroupJob*) job_;
  struct ncclComm **groupCommHeadMain = gjob->groupCommHead;
  struct ncclComm *groupCommPreconnectHeadMain = gjob->groupCommPreconnectHead;
  struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> *asyncJobsMain = &gjob->asyncJobs;
  bool *groupAbortFlag = &gjob->abortFlag;

/* -----------The following code should be moved to a later stage-------
  if (!simInfo && groupCommPreconnectHeadMain != nullptr) {
    struct ncclComm* comm = groupCommPreconnectHeadMain;
    do {
      struct ncclPreconnectJob* job;
      NCCLCHECKGOTO(ncclCalloc(&job, 1), ret, fail);
      job->base.func = ncclP2PPreconnectFunc;
      job->base.undo = nullptr;
      job->base.destructor = free;
      job->base.state = ncclGroupJobRunning;
      job->base.abortFlag = comm->abortFlag;
      job->base.abortFlagDev = comm->abortFlagDev;
      job->comm = comm;
      ncclIntruQueueEnqueue(asyncJobsMain,  (struct ncclAsyncJob*)job);

      struct ncclComm* next = comm->preconnectNext;
      comm->preconnectNext = reinterpret_cast<struct ncclComm*>(0x1);
      comm = next;
    } while (comm != nullptr);
  }

  NCCLCHECKGOTO(asyncJobLaunch(asyncJobsMain, groupAbortFlag), ret, fail);
*/

  // only loop through sym alloc and register tasks
  for (int type = ncclGroupTaskTypeSymRegister; type <= ncclGroupTaskTypeSymRegister; ++type) {
    ...
  }

  /* Connect channels at runtime if cumem is supported */
  if (groupCommHeadMain[ncclGroupTaskTypeCollective] != nullptr) {
    struct ncclComm* cliqueHead = groupCommHeadMain[ncclGroupTaskTypeCollective];
    struct ncclComm* comm = NULL;
    struct ncclIntruQueue<struct ncclAsyncJob, &ncclAsyncJob::next> asyncCollJobs;
    ncclIntruQueueConstruct(&asyncCollJobs);
    do {
      // We need to preconnect connections for collectives clique by clique to avoid
      // race condition for split shared comms which can connect the same connections
      // at the same time.
      comm = cliqueHead;
      do {
        bool needConnect = false;
        bool algoNeedConnect[NCCL_NUM_ALGORITHMS];
        memset(algoNeedConnect, 0, sizeof(bool) * NCCL_NUM_ALGORITHMS);

        CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);
        NCCLCHECKGOTO(ncclPrepareTasks(comm, algoNeedConnect, &needConnect, simInfo), ret, fail);

        if (comm->cuMemSupport && needConnect) {
          struct ncclPreconnectJob* job;
          NCCLCHECKGOTO(ncclCalloc(&job, 1), ret, fail);
          job->base.func = ncclCollPreconnectFunc;
          job->base.undo = nullptr;
          job->base.destructor = free;
          job->base.state = ncclGroupJobRunning;
          job->base.abortFlag = comm->abortFlag;
          job->base.abortFlagDev = comm->abortFlagDev;
          job->comm = comm;
          NCCLCHECKGOTO(ncclCalloc(&job->algoNeedConnect, NCCL_NUM_ALGORITHMS), ret, fail);
          memcpy(job->algoNeedConnect, algoNeedConnect, sizeof(bool) * NCCL_NUM_ALGORITHMS);
          ncclIntruQueueEnqueue(&asyncCollJobs, &job->base);
        }
        comm = comm->groupNext[ncclGroupTaskTypeCollective];
      } while (comm != nullptr && comm->intraComm0 == cliqueHead->intraComm0);
      // connect
      NCCLCHECKGOTO(asyncJobLaunch(&asyncCollJobs, groupAbortFlag), ret, fail);
      while (!ncclIntruQueueEmpty(&asyncCollJobs)) {
        struct ncclAsyncJob* job = ncclIntruQueueDequeue(&asyncCollJobs);
        if (job->destructor) job->destructor((void*)job);
      }
      cliqueHead = comm;
    } while (cliqueHead != nullptr);

    // done with all buffer allocation, start registration and enqueue
    comm = groupCommHeadMain[ncclGroupTaskTypeCollective];
    do {
      CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);
      NCCLCHECKGOTO(ncclTasksRegAndEnqueue(comm), ret, fail);
      comm = comm->groupNext[ncclGroupTaskTypeCollective];
    } while (comm);
  }

/* ------ Pre-connect section moved to here, after ncclPrepareTasks-----*/
  if (!simInfo && groupCommPreconnectHeadMain != nullptr) {
    struct ncclComm* comm = groupCommPreconnectHeadMain;
    do {
      struct ncclPreconnectJob* job;
      NCCLCHECKGOTO(ncclCalloc(&job, 1), ret, fail);
      job->base.func = ncclP2PPreconnectFunc;
      job->base.undo = nullptr;
      job->base.destructor = free;
      job->base.state = ncclGroupJobRunning;
      job->base.abortFlag = comm->abortFlag;
      job->base.abortFlagDev = comm->abortFlagDev;
      job->comm = comm;
      ncclIntruQueueEnqueue(asyncJobsMain,  (struct ncclAsyncJob*)job);

      struct ncclComm* next = comm->preconnectNext;
      comm->preconnectNext = reinterpret_cast<struct ncclComm*>(0x1);
      comm = next;
    } while (comm != nullptr);
  }

  NCCLCHECKGOTO(asyncJobLaunch(asyncJobsMain, groupAbortFlag), ret, fail);
  /*-----------------------------------------------------------------*/

  ...
}
````

We choose option 1 in the end because there is ongoing grouped symmetric kernel/CE implementation effort, which aligns better with the approach in option 1.

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Coding</h2></summary>
<!-- ============================================================================================-->

https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/1049

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Testing and Validation</h2></summary>
<!-- ============================================================================================-->

In the perf test, we already have the alltoall, gather, scatter tests that implemented with NCCL send/recv
APIs. We could either:
  - Replace the existing tests implemented with NCCL send/recv APIs with the host native APIs. 
  - Keep the old send/recv perf test and add extra test cases for the new host native APIs, and compare if there is perf degradation. 


</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Signoff List</h2></summary>
<!-- ============================================================================================-->

Author(s):
  - Zhenhao He

</details>
