# DGX Spark Multi-Node Ring Subnet-Aware Routing
<!-- use this template to share the details of your feature. Non-commented sections are strongly encouraged -->

## Abstract
<!-- here detail what the feature is about. A few sentence that can be copy-pasted to external actors -->
On a 3-node DGX Spark direct-connect ring, each node has 2 NIC ports on different subnets, each cabled to a different neighbor. NCCL core often selects the wrong NIC for a given peer connection, creating QPs on a port that has no L2 path to the destination. This feature adds subnet-aware device selection in the IB transport: the listener embeds its RoCE GIDs in the connection handle (exchanged via bootstrap TCP before connect), and the connector uses those GIDs to find a local NIC on a matching subnet before any QPs are created. The behavior is gated by an opt-in environment variable to avoid any regression on existing deployments.

<!-- ============================================================================================-->
<details>
<summary><h2>Motivation and requirements</h2></summary>
<!-- ============================================================================================-->

The officially supported DGX Spark direct-connect ring size is **3 nodes**. In this configuration:

1. Each Spark has 4 PCIe physical functions (PFs) across 2 PCIe root complexes (RCs), and 2 physical QSFP ports. Each RC exposes 2 PFs (one for each QSFP port), and each QSFP port is fed by 2 PFs (one from each RC).
2. Each of the 2 QSFP cables on a node terminates on a different neighbor, on a different subnet.
3. A given QSFP port (and the PFs feeding it) can only reach the one peer it is physically cabled to.

NCCL's default IB transport device selection does not factor in physical reachability. With NIC Fusion, multiple PFs are merged into a single logical device under the assumption that any PF in the fused device can reach any peer (true on a switched fabric). On a direct-connect ring, this assumption breaks: NCCL may pick a fused PF that has no cable to the target peer, creating QPs on a port that has no L2 path to the destination. The user-visible symptom is significantly reduced inter-node bandwidth or outright connection failure even though all cables are wired correctly.

### NVbugs / Jira Tickets

- https://nvbugs/5916811

### User Experience

The feature is opt-in via two environment variables:

- `NCCL_IB_SUBNET_AWARE_ROUTING` (default `0`): set to `1` to enable subnet-aware NIC selection in the IB transport. When disabled, NCCL behavior is unchanged.
- `NCCL_IB_SUBNET_PREFIX_LEN` (default `24`): IPv4 subnet prefix length used to determine whether two RoCE GIDs are on the same subnet. Range `[1, 32]`. The default `/24` matches the typical netplan configuration used on DGX Spark direct-connect deployments.

On the 3-Spark ring, this feature must be paired with `NCCL_IB_MERGE_NICS=0` (disable NIC Fusion). See *Limitation: device-level granularity* below for why.

NCCL's IB transport sets up each peer-to-peer connection asymmetrically. One side calls `ncclIbListen()` (passive side, opens a TCP listening socket and publishes a handle); the other side calls `ncclIbConnect()` (active side, reads the handle and initiates the connection); the passive side then calls `ncclIbAccept()` to complete the handshake. So **listener and accept run on the same node** (the receive side) at two different points in time, and **connector runs on the other node** (the send side). Because the device override must happen on **both** ends and **before any QPs are created**, the feature inserts logic at all three call sites:

1. **Listener** (early, on the receive side, before the connector has contacted it): embeds its RoCE GIDs into the connection handle so the active side can use them.
2. **Connector** (active side, after reading the handle but before opening QPs): selects a local merged IB device whose PF GID shares a subnet with one of the listener's GIDs.
3. **Accept** (receive side, after the connector has sent its device metadata): inspects the remote sender's GIDs and re-selects the local device if the topology-code default cannot reach the peer's subnet.

If no subnet match is found (e.g., misconfigured cabling or single-subnet deployment), the original device selected by the topology code is retained, preserving existing behavior.

### Assumptions, constraints and dependencies

1. RoCE only. The feature is a no-op for IB link layer (subnet-prefix matching does not apply to LIDs).
2. Each cable must be on a dedicated subnet for the matching to be effective. If multiple cables share the same subnet (e.g., two NICs on a node with IPs in `192.168.0.0/24`), the kernel's weak host model and ARP behavior already cause traffic to collapse onto a single NIC; subnet-aware routing cannot help in that case.
3. Up to 2 listener GIDs are embedded in the handle (`listenGids[2]`). The change fits within the existing 128-byte `ncclIbHandle` size limit. All-zero GID slots (from older NCCL builds or IB deployments) are safely ignored via `validGid()`.
4. The feature does not change topology or graph search; it only overrides the per-connection device choice late in the IB connect/accept path.

### Use Cases

1. **3-node DGX Spark direct-connect ring** — the officially supported and tested configuration. Each node directly cables to 2 neighbors on different subnets.
2. 2-Spark back-to-back direct-attach (1 or 2 cables) — regression check; the feature must preserve existing NIC Fusion behavior.
3. Switched RoCE / IB clusters — regression check; the feature is a no-op when not enabled.

Larger ring sizes (e.g., 4-node) may work with this code, but are not part of the officially supported DGX Spark topology and are not validated.

### Platform Requirements

- 3 DGX Spark systems wired in a ring via QSFP cables (no switch).
- Per-cable subnets configured via netplan (or equivalent) with one /24 per physical link.

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Design</h2></summary>
<!-- ============================================================================================-->

### Proposed Design

#### Background

DGX Spark NIC architecture (per node). Each QSFP port is fed by 2 PFs (one from each RC):

```
   PCIe Side                           CX-7           Network Side

   RC0 ── PF f0 (rocep1s0f0)   ──┐
                                 ├──── QSFP Port 0 ── cable to neighbor A
   RC1 ── PF f0 (roceP2p1s0f0) ──┘

   RC0 ── PF f1 (rocep1s0f1)   ──┐
                                 ├──── QSFP Port 1 ── cable to neighbor B
   RC1 ── PF f1 (roceP2p1s0f1) ──┘
```

NIC Fusion at default merge level (`PATH_PORT`) merges PFs that share a PCI bus ID with different function numbers. Each fused merged device contains 2 PFs:
- `rocep1s0f0 + rocep1s0f1` (RC0, ports 0 and 1)
- `roceP2p1s0f0 + roceP2p1s0f1` (RC1, ports 0 and 1)

In a 3-node ring, port 0 cable goes to one neighbor and port 1 cable goes to another neighbor. The two PFs in a fused device therefore reach **different** peers. Without subnet awareness, NCCL has no way to know which PF in the fused device is wired to which peer, and may bind a QP to the wrong PF.

#### 3-node ring topology

Each cable carries traffic from **2 PFs** (one from each RC, both feeding the same physical port). To utilize both RCs in parallel and reach the per-cable PCIe-bound max throughput, the two PFs on the same cable must be on **different subnets**. When they share a subnet, the kernel's routing table has two equally-specific routes for that subnet and tends to favor one of them, leaving the other PF underutilized and reducing observed throughput.

Therefore each cable consumes **2 distinct /24 subnets** (one per PF / per IP), and a 3-node ring with 3 cables uses **6 subnets** in total.

Per-node interface naming on DGX Spark:

| Interface       | RC  | Port   |
| --------------- | --- | ------ |
| `enp1s0f0np0`   | RC0 | Port 0 |
| `enP2p1s0f0np0` | RC1 | Port 0 |
| `enp1s0f1np1`   | RC0 | Port 1 |
| `enP2p1s0f1np1` | RC1 | Port 1 |

Ring topology (each cable carries 2 subnets, one per PF feeding the port):

```
                          ┌──────────────────────────────────┐
                          │             Spark 1              │
                          │   Port 0           Port 1        │
                          │  enp1s0f0np0      enp1s0f1np1    │
                          │  enP2p1s0f0np0    enP2p1s0f1np1  │
                          └────┬───┬──────────────┬───┬──────┘
                               │   │              │   │
                  192.168.0/24 │   │ 192.168.1/24 │   │ 192.168.3/24
                  192.168.2/24 │   │              │   │
                               │   │              │   │
                ┌──────────────┘   │              │   └──────────────┐
                │   ┌──────────────┘              └──────────────┐   │
                │   │                                            │   │
       ┌────────┴───┴───────────┐                       ┌────────┴───┴───────────┐
       │   Port 1               │                       │            Port 0      │
       │ enp1s0f1np1            │                       │      enp1s0f0np0       │
       │ enP2p1s0f1np1          │                       │      enP2p1s0f0np0     │
       │            Spark 2     │                       │     Spark 3            │
       │ enp1s0f0np0            │                       │      enp1s0f1np1       │
       │ enP2p1s0f0np0          │                       │      enP2p1s0f1np1     │
       │   Port 0               │                       │            Port 1      │
       └────────┬───┬───────────┘                       └────────┬───┬───────────┘
                │   │                                            │   │
                │   └──────────── 192.168.5/24 ──────────────────┘   │
                └──────────────── 192.168.4/24 ──────────────────────┘

  Cable subnets (each cable carries 2 subnets, one per PF):
    Spark 1 Port 0  ── 192.168.0/24, 192.168.1/24 ──  Spark 2 Port 1
    Spark 1 Port 1  ── 192.168.2/24, 192.168.3/24 ──  Spark 3 Port 0
    Spark 2 Port 0  ── 192.168.4/24, 192.168.5/24 ──  Spark 3 Port 1
```

A subnet uniquely identifies one PF (and therefore which port and which peer it is wired to). The connector matches subnets to pick the correct local PF / merged device for each peer.

#### Approach

At IB connection setup time, exchange RoCE GIDs in-band via the existing handle and connection metadata, and use them to select a local merged device whose PFs share a subnet with the remote peer. All overrides happen **before any QPs are created**, so no fall-back / repair logic is needed in the data path.

Flow:

1. **Listener** (`ncclIbListen`): after socket setup, queries GIDs for all RoCE PFs in the assigned merged device and embeds them in the `ncclIbHandle` returned to the bootstrap layer.
2. **Connector** (`ncclIbConnect`): before any device operations, calls `ncclIbFindDevBySubnet(handle->listenGids, ...)` to override `dev` when the default device is on a different subnet than the peer.
3. **Accept** (`ncclIbAccept`, after receiving connection metadata from the sender): extracts RoCE GIDs from `remMeta.devs[]` and calls `ncclIbFindDevBySubnet()` to override `lComm->dev` and `rComm->base.vProps` if the default device cannot reach the peer's subnet.

#### Device selection algorithm

`ncclIbFindDevBySubnet(remoteGids, nRemoteGids, defaultDev)`:

```
if no remoteGids are valid:
    return defaultDev   # IB or single-subnet deployment

# Preserve NIC Fusion when all PFs in the default device match the peer
if defaultDev is valid:
    checked, matched = 0
    for each RoCE PF in defaultDev:
        query localGid
        checked++
        if localGid subnet matches any remoteGid subnet:
            matched++
    if checked > 0 and matched == checked:
        return defaultDev   # full reachability, keep fused device

# Otherwise scan all merged devices for one whose PF matches a remote subnet
for each merged device d != defaultDev:
    for each RoCE PF in d:
        query localGid
        if localGid subnet matches any remoteGid subnet:
            return d

return defaultDev   # no better match; let connect fail loudly if cable missing
```

The "default device first, all PFs must match" check protects the 2-cable back-to-back case: when both ports of a fused device reach the same peer, fusion must be preserved to use both cables. Only when the default device cannot fully reach the peer (e.g., one port is wired to a different neighbor) does the search fall back to a single-PF device.

#### Helper functions

- **`ncclIbGidSameSubnet(localGid, remoteGid)`** — Compares two GIDs:
  - Same address family required.
  - IPv4-mapped GIDs (`::ffff:a.b.c.d`): mask both IPv4 addresses with `NCCL_IB_SUBNET_PREFIX_LEN` (default `/24`, matches the typical Spark netplan setup) and compare.
  - Native IPv6 GIDs: compare the 64-bit `subnet_prefix`.
- **`ncclIbSubnetMatchesAny(localGid, remoteGids, n)`** — Returns true if `localGid` shares a subnet with any valid GID in the `remoteGids` array.
- **`ncclIbFindDevBySubnet(remoteGids, n, defaultDev)`** — Core routing logic, described above.

#### Limitation: device-level granularity

The override returns a **merged device index**, not a specific PF inside a merged device. NCCL's existing connection setup takes only a single `dev` integer; there is no way to say "use merged device X but only PF f0 inside it."

This matters in the 3-node ring with NIC Fusion enabled. NCCL fuses each RC's two PFs into one merged device:

- `merged_dev[0]` = `rocep1s0f0 + rocep1s0f1` (RC0)
- `merged_dev[1]` = `roceP2p1s0f0 + roceP2p1s0f1` (RC1)

Inside `merged_dev[0]`, `rocep1s0f0` is wired (via Port 0) to one peer, and `rocep1s0f1` is wired (via Port 1) to a different peer. So no single merged device fully reaches any one peer in the ring.

When setting up a connection to peer A, the code correctly identifies which PF is on peer A's subnet — but it can only return the merged device that contains that PF. NCCL then spreads QPs across **both** PFs in the merged device, including the one wired to peer B, which has no L2 path to peer A. Those QPs fail or stall.

For the supported 3-Spark ring, the workaround is to disable NIC Fusion via `NCCL_IB_MERGE_NICS=0`. With fusion off, each PF is its own merged device, so returning a device index is equivalent to picking a specific PF.

A follow-on improvement would extend the override to PF-level granularity, e.g., by registering a 1-PF virtual device on the fly via `makeVDevice` when the topology-default merged device has mixed reachability. That would let NIC Fusion remain enabled (preserving its benefits on 2-Spark / switched setups) while still routing each connection through the correct single PF on the ring.

### Interface Architecture

No public NCCL API changes. The change extends the internal IB connection handle and adds two NCCL parameters:

```c
struct ncclIbHandle {
  // ... existing fields ...
  // GIDs of the listener's device PFs, used by the connector to find a local
  // NIC on the same subnet (for multi-subnet RoCE direct-connect topologies).
  // Zero-valued slots are ignored (validGid() returns false).
  union ibv_gid listenGids[2];
};

NCCL_PARAM(IbSubnetAwareRouting, "IB_SUBNET_AWARE_ROUTING", 0);
NCCL_PARAM(IbSubnetPrefixLen,    "IB_SUBNET_PREFIX_LEN",    24);
```

The `ncclIbHandle` change is wire-format compatible only between matching NCCL versions; both peers must run a build that includes this change for the embedded GIDs to be interpreted correctly. With `NCCL_IB_SUBNET_AWARE_ROUTING=0` (default), the new GID slots remain zeroed and the matching logic short-circuits, preserving compatibility with the prior behavior on switched / single-subnet deployments.

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Coding</h2></summary>
<!-- ============================================================================================-->

### Commit list or MR

- Branch: `zyang/dgxspark-ring-support`
- Files modified: `src/transport/net_ib/connect.cc`
<!-- TODO: add MR link once filed -->

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Testing and Validation</h2></summary>
<!-- ============================================================================================-->

### Objectives and Timeline

1. Confirm that `NCCL_IB_SUBNET_AWARE_ROUTING=0` (default) produces no regression vs. the unmodified NCCL on existing deployments (switched fabrics, 2-Spark back-to-back, IB clusters).
2. Confirm that `NCCL_IB_SUBNET_AWARE_ROUTING=1` enables correct NIC selection and full bandwidth on the 3-Spark direct-connect ring.

### Validation

#### Where to run?

- **3-Spark direct-connect ring** (primary target) — each cable on its own /24 subnet.
- 2-Spark back-to-back (1 cable and 2 cables) — regression check.
- Existing switched RoCE / IB cluster — regression check.

Larger ring sizes are not officially supported and not part of validation.

#### What to run?

- `nccl-tests/build/all_gather_perf` on the 3-Spark ring with the feature enabled.
- Same test with the env var unset (default `0`) on switched / 2-node setups to confirm no regression.
- `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=NET` to inspect device override messages.

Required env vars on the 3-Spark ring:

```
-x NCCL_IB_HCA=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1
-x NCCL_IB_SUBNET_AWARE_ROUTING=1
-x NCCL_IB_MERGE_NICS=0      # required: see "Limitation: device-level granularity"
-x NCCL_NET_PLUGIN=none      # use the in-tree NCCL IB transport, not an external plugin
-x NCCL_SOCKET_IFNAME=enP7s7 # bootstrap interface (management NIC, not the QSFP NICs)
```

Example invocation used during validation:

```sh
CUDA_HOME="/usr/local/cuda" && \
MPI_HOME="/usr/lib/aarch64-linux-gnu/openmpi" && \
NCCL_HOME="$HOME/nccl/build" && \
LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64/:$MPI_HOME/lib:$LD_LIBRARY_PATH" && \
mpirun -np 3 -H 10.137.203.173:1,10.137.203.164:1,10.137.203.126:1 \
  --mca plm_rsh_agent "ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no" \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  -x UCX_NET_DEVICES=enP7s7 \
  -x NCCL_SOCKET_IFNAME=enP7s7 \
  -x OMPI_MCA_btl_tcp_if_include=enP7s7 \
  -x NCCL_IB_HCA=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1 \
  -x NCCL_IB_SUBNET_AWARE_ROUTING=1 \
  -x NCCL_DEBUG=INFO \
  -x NCCL_NET_PLUGIN=none \
  -x NCCL_IB_MERGE_NICS=0 \
  $HOME/nccl-tests/build/all_gather_perf -b 16G -e 16G -f 2
```

#### Expected output?

1. 3-Spark ring with feature on: connections succeed on all peers, full per-link bandwidth realized.
2. 2-Spark back-to-back: identical behavior with feature on/off (the default device already reaches the only peer, so `ncclIbFindDevBySubnet()` keeps it; NIC Fusion is preserved).
3. Switched / IB clusters: feature off (default) — no behavior change.

### Performance

#### What is measured?

NCCL bus bandwidth (`busbw`) reported by `all_gather_perf` on the 3-Spark ring.

#### Results

Verified on a 3 DGX Spark ring setup using `nccl-tests/build/all_gather_perf` — connections succeed on all ring peers and bandwidth is restored vs. the broken default behavior.

```
       size    count       type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong
        (B)    (elt)                                       (us)  (GB/s)  (GB/s)               (us)  (GB/s)  (GB/s)
17179869168  1431655764     float    none      -1   496916   34.57   23.05       0   496765   34.58   23.06       0

# Avg bus bandwidth    : 23.0522
# Collective test concluded: all_gather_perf
```

Bus bandwidth ~23 GB/s, close to the per-cable PCIe-bound max of ~25 GB/s.

</details>

<!-- ============================================================================================-->
<details>
<summary><h2>Signoff List</h2></summary>
<!-- ============================================================================================-->

Author(s):
  - Zifu Yang (zyang@nvidia.com)

</details>
