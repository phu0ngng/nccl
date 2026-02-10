# Dynamic GID Detection in IB RoCE networks
<details>
<summary><h2>Requirements</h2></summary>

### Introduction

None.

#### Overview

None.

#### Assumptions

NCCL is used with IB/RoCE networks.

#### Constraints

None.

#### Dependencies

None.

#### References

[NVBug NCCL-4170289](https://nvbugspro.nvidia.com/bug/4170289)

### Use cases

Users need NCCL to dynamically configure RoCE interfaces for fast
inter-node communication between GPUs, even when the user has not
specified any GID index (through the NCCL_IB_GID_INDEX variable) to be
used by NCCL. This might happen, for example, when it is not possible to
specify a valid GID index for all the nodes using a single variable (if
candidate GIDs have different indices across nodes).

### Requirements

In the absence of a user defined GID index (NCCL_IB_GID_INDEX unset),
NCCL should try to determine, among the available HCA ports, a valid GID
index to use for creating network end-points. In order to do this NCCL
should allow the user to filter GIDs based on their address family
(AF_INET vs AF_INET6), RoCE version (RoCE v1 vs RoCE v2), and address
range (last resort).

### System Requirements

Clusters with RoCE HCAs.

### Signoff list

Author : Giuseppe Congiu
</details>

<details>
<summary><h2>Design</h2></summary>

### Problem Statement

GIDs are 128-bits integer values used in RoCE HCAs to identify (along
with QP numbers) endpoints during communication. Every HCA port has
several GIDs organized in a table. The table size can be queried using
**ibv_query_port** and is returned in the **gid_tlb_len** field of the
**struct ibv_port_attr** object. Not all the GID entries in the table
might be configured (those that are not have a special value, more on
this later).

GIDs follow the Ipv6 format. The 64 MSBs encode the subnet prefix, while
the 64 LSB encore the local id of the address. When Ipv4 (AF_INET)
addresses are used, these are mapped to Ipv6 addresses as follows.
Non-multicast addresses take the form
0000:0000:0000:0000:0000:ffff:\<Ipv4\>, multicast addresses take the
form ff0e:0000:0000:0000:0000:ffff:\<Ipv4\>. GIDs that have not been
configured have either all their bits set to 0 or the 16 MSBs set to
0xff80 and should be ignored. Link Local GIDs should also be ignored as
they are only used for subnet communication and are preconfigured by the
system (RFC 4291, IP Version 6 Addressing Architecture). Link local GIDs
take the form fe80:0000:0000:0000:\<local identifier\>.

In RoCE NICs that support it, HCA GIDs can be configured to work either
in version 1, 2 or both. In the last case every configured GID will be
duplicated in the GID table. One entry will refer to RoCE v1 and the
other to RoCE v2. The RoCE version can be queried (only) for configured
GIDs by reading the content of
/sys/class/infiniband/\<device\>/ports/\<port\>/gid_attrs/types/\<entry\>
(e.g., /sys/class/infiniband/mlx5_0/ports/1/gid_attrs/types/0), and can
assume the string values of "IB/RoCE v1" or "RoCE v2", for RoCE version
1 and 2, respectively. The difference between RoCE v1 and v2 is that v2
is UDP/IP based whereas v1 is not and works directly on top of the
Ethernet link layer.

Normally, the first usable GID index in a RoCE HCA configured to support
both v1 and v2 is 3. Thus, the NCCL_IB_GID_INDEX variable in NCCL should
be set to 3. However, network state transitions (link up/down) can cause
the GID indices to change, making NCCL_IB_GID_INDEX no longer usable (at
least on the affected network interfaces). To solve this problem a
dynamic GID detection strategy is adopted.

### Design

Dynamic GID detection is performed whenever the NCCL_IB_GID_INDEX
variable is left unset. This guarantees backward compatibility with
environments setting the value of the variable. The dynamic detection
strategy allows NCCL users to filter GIDs based on the following
criteria:

1.  IP address family (IF_INET or AF_INET6)
2.  IP address range (Classless Inter-Domain Routing, or CIDR) and
3.  RoCE version (IB/RoCE v1 vs RoCE v2)

Each of the criteria above is controlled through a new NCCL variable:

1.  NCCL_IB_ADDR_FAMILY (default is AF_INET)
2.  NCCL_IB_ADDR_RANGE (default is ::/0)
3.  NCCL_IB_ROCE_VERSION_NUM (default is 2)

The GID table below is used to highlight the GID selection strategy
implemented in src/transport/net_ib.cc:

| Dev    | Port | Index | GID                                     | Ipv4       | Type |
|--------|------|-------|-----------------------------------------|------------|------|
| mlx5_0 | 1    | 0     | fe80:0000:0000:0000:ba59:9fff:fe1a:e3ea |            | v1   |
| mlx5_0 | 1    | 1     | fe80:0000:0000:0000:ba59:9fff:fe1a:e3ea |            | v2   |
| mlx5_0 | 1    | 2     | 0000:0000:0000:0000:0000:ffff:0a0a:0a01 | 10.10.10.1 | v1   |
| mlx5_0 | 1    | 3     | 0000:0000:0000:0000:0000:ffff:0a0a:0a01 | 10.10.10.1 | v2   |
| mlx5_1 | 1    | 0     | fe80:0000:0000:0000:ba59:9fff:fe1a:e3eb |            | v1   |
| mlx5_1 | 1    | 1     | fe80:0000:0000:0000:ba59:9fff:fe1a:e3eb |            | v2   |

If the user does not change the default values for NCCL_IB_ADDR_FAMILY
and NCCL_IB_ROCE_VERSION_NUM, NCCL will pick the GID at index 3 from
mlx5_0. The logic through which this selection happens is following
described:

    int gidIndex = 0;
    for (int gidIndexCandidate = 1; gidIndexCandidate < gid_tbl_len; ++gidIndexCandidate) {
        union ibv_gid gid, gidCandidate;
        ibv_query_gid(context, portNum, gidIndex, &gid);
        ibv_query_gid(context, portNum, gidIndexCandidate, &gidCandidate);

        int prefixlen;
        sa_family_t gidFam = getGidAddrFamily(&gid);
        sa_family_t gidCandidateFam = getGidAddrFamily(&gidCandidate);
        sa_family_t usrFam = envIbAddrFamily();
        void *prefix = envIbAddrRange(usrFam, &prefixlen);
        bool gidCandidateMatchSubnet = matchGidAddrPrefix(usrFam, prefix, prefixlen, &gidCandidate);

        if (gidCandidateFam != gidFam && gidCandidateFam == usrFam && gidCandidateMatchSubnet) {
            gidIndex = gidIndexCandidate;
        } else {
            if (gidCandidateFam != usrFam || !validGid(&gidCandidate) || !gidCandidateMatchSubnet) {
                return ncclSuccess;
            }
            int usrRoceVer = roceVer;
            int gidRoceVerNum, gidRoceVerNumCandidate;
            const char* deviceName = ibv_get_device_name(context->device);
            ncclIbRoceGetVersionNum(deviceName, portNum, gidIndex, &gidRoceVerNum);
            ncclIbRoceGetVersionNum(deviceName, portNum, gidIndexCandidate, &gidRoceVerNumCandidate);
            if ((gidRoceVerNum != gidRoceVerNumCandidate || !validGid(&gid)) && gidRoceVerNumCandidate == usrRoceVer) {
                gidIndex = gidIndexCandidate;
            }
        }
    }

Overall, the sequence of values assumed by gidIndex is 0, 1, 2, 3. The
first if branch allows for transitioning from index 1 to index 2 in the
GID table of mlx5_0. The else branch allows for transitioning from index
0 to 1 and 2 to 3 in the GID table of mlx5_0, to select the right RoCE
version.

### Signoff list

Author : Giuseppe Congiu
</details>

<details>
<summary><h2>Coding</h2></summary>

[MR
359](https://gitlab-master.nvidia.com/nccl/nccl/-/merge_requests/359)
</details>

<details>
<summary><h2>Testing</h2></summary>

### Introduction

#### Test strategy

None

#### Test environment

None

### Signoff list

Author : Giuseppe Congiu
</details>

