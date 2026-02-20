# Railed Gin
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract
<!-- here detail what the feature is about. A few sentence that can be copy-pasted to external actors -->

This feature adds support for "Railed GIN". Each rank connects only to other ranks on the same rail team.

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

### NVbugs / Jira Tickets

https://nvbugspro.nvidia.com/bug/5738377

### User Experience

To use this feature, end users will continue to indicate (lack of) cross-NIC support via `NCCL_CROSS_NIC=0`. Developers using the Device API can request railed GIN when creating a DevComm. If an end-user specifies `NCCL_CROSS_NIC=0`, but the developer requested full GIN connectivity, `DevCommCreate` will fail.

More specificially, we will add a new `ncclGinConnectionType`:

```
typedef enum {
  NCCL_GIN_CONNECTION_NONE,
  NCCL_GIN_CONNECTION_FULL,
  NCCL_GIN_CONNECTION_RAIL, // New
} ncclGinConnectionType_t;
```

Device API users can request Railed-GIN via DevCommRequirements:

```
struct ncclDevCommRequirements {
  ...
  ncclGinConnectionType_t ginConnectionType; // This field already exists
};
```

Users can discover railed GIN support via `ncclCommProperties`. A new field is added to preserve the semantics of the original field:

```
ncclCommProperties {
  ncclGinType_t ginType; // Existing field. If not NONE, devCommCraete with NCCL_GIN_CONNECTION_FULL will succeed
  ncclGinType_t railedGinType; // New field. If not NONE, devCommCreate with NCCL_GIN_CONNECTION_RAIL will succeed
}
```

The definition of `railedGinType` is very similar to ginType, but without the CROSS_NIC check:
```
ncclCommQueryProperties(...) {
  if (comm.ncclGin != NULL) { // pseduocode, we have a few other conditions as well
    if (NCCL_CROSS_NIC != 0) {
      props.ginType = comm.ncclGin.ginType;
      props.railedGinType = com.ncclGin.ginType;
    } else {
      props.ginType = NCCL_GIN_TYPE_NONE;
      props.railedGinType = comm.ncclGin.ginType;
    }
  }

}
```


### Assumptions, constraints and dependencies

We assume full NIC connectivitity within each ncclTeamRail. This assumption may be false if a communicator spans multiple nodes and each node has a different number of ranks. For example, a communicator with 7 ranks that span 2 nodes will be converted to 7 LSA teams, and thus 1 rail team. If cross-NIC communication is not supported, this rail team does not have full connectivitiy and NCCL_GIN_CONNECTION_RAIL will fail. We do not attempt to support, detect, or disallow this case.

We assume consecutive calls to DevCommCreate request the same ginConnectionType. A call to `DevCommCreate` will fail (gracefully) if GIN is requested with GIN_CONNECTION_FULL but was previously requested with GIN_CONNECTION_RAIL. This is a temporary limitation and will be lifted once GIN contexts are per-devcomm (https://nvbugspro.nvidia.com/bug/5813666)

For railed GIN, gin.put or gin.signal on a non-railed peer is undefined behavior and will likely result in silent data corruption.

### Use Cases

Support matrix: GIN can now be used on clusters that do not support cross-rail connections.
Performance: All-to-all connections are resource-intensive. Railed GIN is lightweight and usually sufficient.


### Platform Requirements

<!-- ### Functional Requirements -->
<!-- ### System Requirements -->
<!-- ### Interface Requirements -->
<!-- ### KPI Requirements -->
<!-- ### Security Requirements -->
<!-- ### Legal and Standards Requirements -->
<!-- ### Telemetry Requirements -->
<!-- ### Backward Compatibility Requirements -->
<!-- ### Virtualization Requirements -->

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Design</h2></summary>
<!-- ============================================================================================-->

### Proposed Design

Railed GIN is transparent to the GIN plugin. Currently, the GIN plugin accepts an array of handles to connect to. For full connectivitiy, NCCL passes one handle per rank in the communicator. With railed GIN, NCCL needs to pass one handle per rank in the rail team:

```
ncclResult_t ncclGinConnectOnce(struct ncclComm* comm, ncclGinConnectionType_t connectType) {
  ...
  char* allHandles; // boostrapAllGather already populates one per rank in the communicator
  void** handles;
  int nGinRanks;
  int myGinRank;
  if (connectType == NCCL_GIN_CONNECTION_FULL) { // pseudocode roughly reflects the current implementaion
    nGinRanks = comm->nRanks;
    myGinRank = comm->rank;
    for (int i = 0; i < nGinRanks; i++) {
      handles[i] = allHandles+ i * NCCL_NET_HANDLE_MAX_SIZE;
    }
  } else if (connectType == NCCL_GIN_CONNECTION_RAIL) { // psuedocode for new implementation
    nGinRanks = ncclTeamRail(comm).nRanks;
    myGinRank = ncclTeamRail(comm).rank;
    for (int i = 0; i < nGinRanks; i++) {
      int worldRank = ncclTeamRankToWorld(comm, ncclTeamRail(comm), i);
      handles[i] = allHandles + worldRank * NCCL_NET_HANDLE_MAX_SIZE;
    }
  }
  NCCLCHECKGOTO(ginState->ncclGin->connect(comm->ginContext, handles, nGinRanks, myGinRank,
                listenComm, ginState->ginComms + n), ret, fail);
}
```

GIN functions now need to translate a world rank into a GIN rank before calling the backend:

```
// NEW: internal convenience function
int ncclWorldRankToGin(ncclGinCtx ctx, int worldRank) {
  return worldRank / ctx->nTeams; // nTeams is 1 for CONNECTION_FULL and # rails for CONNECTION_RAIL
}

template<unsigned beMask>
template<
  typename T,
  typename RemoteAction,
  typename LocalAction,
  typename Coop,
  typename DescriptorSmem
>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::put(
    ncclTeam team, int peer,
    ncclSymPtr<T> dstElts, ncclSymPtr<T> srcElts, size_t nElts,
    RemoteAction remoteAction, LocalAction localAction,
    Coop coop,
    DescriptorSmem descriptor,
    cuda::thread_scope givenRelease,
    cuda::thread_scope requiredRelease
  ) const {
  int worldPeer = ncclTeamRankToWorld(net->comm, team, peer); // Already done
  int ginPeer = ncclWorldRankToGin(this, worldPeer); // NEW LINE
  this->put(
    team, ginPeer, dstElts.window, dstElts.offset, srcElts.window, srcElts.offset, nElts*sizeof(T),
    remoteAction, localAction, coop, descriptor, givenRelease, requiredRelease
  );
}
```

#### Impact on gin functions

Puts to a peer that is not in the rail team may result in silent data corruption.

#### Impact on hostRma

HostRma will continue to be disabled if NCCL_CROSS_NIC=0

#### Impact on internal symmetric kernels

This PLC assumes internal kernels require GIN full connectivitiy and does not attempt to transition them to railed GIN.

No changes are necessary, and there is no functional or performance change.

<!-- ### Interface Architecture -->

<!-- ### System KPIs & Metrics -->
<!-- ### Data Architecture -->
<!-- ### Security Design -->
<!-- ### Debugging & Troubleshooting -->
<!-- ### Logging and Instrumentation -->
<!-- ### Operational Considerations -->

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Coding</h2></summary>
============================================================================================

### Commit list or MR

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Testing and Validation</h2></summary>
<!-- ============================================================================================-->

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

<!-- ============================================================================================-->
<details>
<summary><h2>Signoff List</h2></summary>
<!-- ============================================================================================-->

Author(s):
- Katie

</details>
