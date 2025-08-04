# Multi Request API

## Abstract

The Multi Request API introduces a new capability to NCCL's net plugin that allows grouping multiple send/recv operations together under a single request handle. This feature is particularly beneficial for collective operations like PAT AllGather, where a large number of small data chunks need to be transferred between ranks. By enabling the grouping of related send/recv operations, the API reduces the overhead associated with managing multiple individual requests and opens opportunities for network-level optimizations.

## Motivation and requirements

<details>

### NVbugs / Jira Tickets

* https://nvbugspro.nvidia.com/bug/4642355
* https://jirasw.nvidia.com/browse/NCCL-1969

### User Experience

Users can call send/recv API with an optional value `NCCL_NET_MULTI_REQUEST` in the input/output `request` argument to hint the plugin that the current send/recv request is part of a group of send/recv requests. When the user calls `isend()`/`irecv()` with a `request` parameter that is equal to `NCCL_NET_MULTI_REQUEST`, the plugin shall return a `NULL` pointer instead of a request handle. A single request handle is returned by the plugin only after the final call to `isend()`/`irecv()` with a `request` parameter that differs from `NCCL_NET_MULTI_REQUEST`.

When the user progresses the request handle, the plugin progresses all the requests within that group and would indicate the the completion for the group request only when all requests within the group have completed.

> **Note**
>
> Having a single handle for the whole group is especially important when the group size becomes large, and testing and progressing many requests could become the bottleneck.

### Assumptions, constraints and dependencies

The API allows the user to initiate only a single group of send/recv request at a time, meaning until a request handle is returned for a group, a new group cannot be started.

The current API implicitly enforces a one-to-one correspondence between send and receive requests within a single group, requiring that the number of receive requests on the receiver side matches the number of send requests on the sender side.

### Use Cases

The main use-case is PAT AllGather. In the final stages of PAT AllGather algorithm, especially in large scale, when the communicator size is large (e.g., 256, 512 ranks), each rank sends many chunks of data using send/recv to a single peer (in the last step of PAT AllGather - each rank sends `N/2` chunks to a peer, when `N` is the size of the communicator). The optimization space includes utilizing the fact that the chunks are sent to a single peer and that these chunks reside in a striding memory pattern.

Within the plugin, these aspects can be optimized in different ways, including taking advantage of the striding pattern that appears in PAT AllGather for efficient zero-copy operations even when chunks are small (allowing NCCL to avoid the need for a copy of the small chunks into a continuous intermediate buffer), and offloading the execution and orchestration of the group of send/recv to the network device. In general, offload and orchestration can help mainly collectives that require many send/recv operations without any particular completion ordering between them.

> **Note**
>
> The [App-aware API](https://gitlab-master.nvidia.com/ztiffany/nccl/-/commit/a3ce5e275a80a9fd0839a1bca125ee2adece0c0f) ([aa-net-plugin branch](https://gitlab-master.nvidia.com/ztiffany/nccl/-/commits/aa-net-plugin?ref_type=heads)) allows to pass information about a communicator. Communicator is often used by a single type of collective so in case of PAT AllGather the communicator can hint the plugin about the striding memory pattern. The API can also be extended to express memory patterns.

> **Note**
> The [App-aware API](https://gitlab-master.nvidia.com/ztiffany/nccl/-/commit/a3ce5e275a80a9fd0839a1bca125ee2adece0c0f) ([aa-net-plugin branch](https://gitlab-master.nvidia.com/ztiffany/nccl/-/commits/aa-net-plugin?ref_type=heads)) allows to pass information about a communicator. Communicator is often used by a single type of collective so in case of PAT AllGather the communicator can hint the plugin about the striding memory pattern. The API can also be extended to express memory patterns.

### Platform Requirements

</details>

## Design

<details>

### Design

The plugin, internally can decide when and how to optimize the execution of the send/recv requests within a group depending on internal state visible to the plugin. For example, the plugin can decide when you submit the requests based on the internal states of the SW/HW queues that it control and manages.

### Interface Architecture

The NCCL plugin API is provided through the `irecv()`/`isend()` function's `request` parameter. When this parameter is set to the predefined value `NCCL_NET_MULTI_REQUEST`, it signals the plugin that the current call to `irecv()`/`isend()` is part of a group of calls to `irecv()`/`isend()` that NCCL expects to get a single request handle for the whole group.

Following a number of `isend()`/`irecv()` calls where the `request` parameter is set to `NCCL_NET_MULTI_REQUEST`, the user must make a final call to `isend()`/`irecv()` with a `request` parameter that differs from `NCCL_NET_MULTI_REQUEST`. This final call signals the finalization of the send/recv group to the plugin, prompting it to return a single request handle. This handle enables the user to test and progress the entire group of send/recv operations through a unified request handle. After a group is finalized, no more requests can be appended to it.

For all intermediate `isend()`/`irecv()` operations within a group (defined as operations where the request parameter is equal to `NCCL_NET_MULTI_REQUEST`), the plugin shall return a NULL pointer instead of a request handle. A request handle is returned by the plugin only after the final call to `isend()`/`irecv()` with a `request` parameter that differs from `NCCL_NET_MULTI_REQUEST`. The last call to `isend()` might also not return a request handle if the Clear-to-Send (CTS) message for this send request has not arrived yet and user is expected to call to `isend()`/`irecv()` until it is returned with a handle.


> **Note**
>
> This implementation that returns a request handle only after the CTS has arrived maintains compatibility with the existing send/recv API semantics, wherein the sender-side request handle is only provided after the plugin's `isend()` function detects the arrival of the corresponding CTS message.

> **Note**
>
> When a user finalizes a group by invoking `isend()` or `irecv()` with request `x`, the plugin appends request `x` to the group. In cases where the plugin does not return a handle upon the initial issuance of request `x`, the user must repeatedly invoke the plugin with the same request `x` until a request handle is returned. It is important to note that subsequent invocations of request `x` resulting from retry attempts will not result in additional appending of request `x` to the group. Consequently, the group will maintain only one instance of request `x`. In addition, if the user will call the `isend()`/`irecv()` with request `y` that is not `x`, the plugin will not append request `y` to the group. For example, for the following sequence:
>
> ```c++
> isend(send_buffer1, request = NCCL_NET_MULTI_REQUEST);
> isend(send_buffer2, request = NCCL_NET_MULTI_REQUEST);
> isend(send_buffer3, request); // request is not NCCL_NET_MULTI_REQUEST. Handle is not returned.
> isend(send_buffer4, request); // request is not NCCL_NET_MULTI_REQUEST. Handle is not returned.
> isend(send_buffer5, request); // request is not NCCL_NET_MULTI_REQUEST. Handle was returned.
> ```
>
> The buffers that would be sent are `send_buffer1`, `send_buffer2` and `send_buffer3`. Note that buffers `send_buffer4` and `send_buffer5` are not sent.

The maximal number of send/recv requests within a single group is an queryable using `ncclIbGetProperties()` and reported by `maxMultiRequestSize`.

</details>

## Coding

<details>

The implementation of the `NCCL_NET_MULTI_REQUEST` API optimizes the execution of the send/recv requests within a group by submitting the requests to the network device in a single batch (aka "post-list") and ringing the doorbell on the NIC on the sender side once for the whole group.

On the receiver side, the plugin aggregates all the receive requests within a group to a single CTS message that is delivered to the sender side. Meaning the size of the group determines the size of the CTS message.

> **Note**
>
> Future optimization can be done to take advantage of the striding pattern (the current API proposal does not suggest it but the plugin can auto-detect it the send/recv requests are in striding pattern if proved efficient) and allow more efficient and compact way to perform the data transfer.

### Commit list or MR

* https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/1010

</details>

## Testing and Validation

<details>

### Objectives and Timeline

### Validation

#### Where to run?

#### What to run?

#### Expected output?

<!-- #### Code Coverage Goal Defined -->
<!-- #### KPI Coverage Goals Defined (performance, stress, stability, throughput, latency) -->
<!-- #### Requirement Coverage Goal Defined -->
<!-- #### Quality Thresholds Defined -->
<!-- #### Test Timeline -->
<!-- #### SW Verification and Test Plan -->
<!-- ### Test plan -->
<!-- #### Requirements Tests -->
<!-- #### Interface Tests -->
<!-- #### Fault-injection Tests -->
<!-- #### Resource Usage Tests -->
<!-- #### Design Coverage Testing -->
<!-- #### Boundary Tests -->
<!-- #### Certification Tests -->
<!-- #### Stress Tests -->
<!-- #### Stability Tests -->
<!-- #### Perf and Power KPI Tests -->
<!-- #### Usability & OOBE Tests -->
<!-- #### Manufacturing Diagnostic(Factory) Tests -->

### Performance

#### What is measured?

#### Results

</details>

## Signoff List

<details>

Author(s):
  - Rami Nudelman
  - Sreeram Potluri
  - Kaiming Ouyang

</details>
