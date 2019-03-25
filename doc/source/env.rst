#####################
Environment Variables
#####################

NCCL has an extensive set of environment variables to tune for specific usage.

They can also be set statically in /etc/nccl.conf (for an administrator to set system-wide values) or in ~/.nccl.conf (for users). For example, those files could contain :

.. code:: C

 NCCL_SOCKET_IFNAME=eth0
 NCCL_DEBUG=WARN
 NCCL_RINGS=0 1 2 3|3 2 1 0

NCCL_P2P_DISABLE
----------------

The ``NCCL_P2P_DISABLE`` variable disables the P2P transport, which uses CUDA direct access between GPUs, using NVLink or PCI.

Values accepted
^^^^^^^^^^^^^^^
Define and set to 1 to disable direct GPU-to-GPU communication.

NCCL_P2P_LEVEL
--------------
(since 2.3.4)

Finely control when to use the P2P transport between GPUs. The level describes the maximum distance between GPUs where we use P2P.

Values accepted
^^^^^^^^^^^^^^^
0 : Never use P2P. (always disabled)

1 : Use P2P when GPUs are on the same PCI switch.

2 : Use P2P when GPUs are connected through PCI switches (potentially multiple hops).

3 : Use P2P when GPUs are on the same PCI root complex, potentially going through the CPU.

4 : Use P2P even across PCI root complexes including traversing the interconnect within a NUMA node.

5 : Use P2P even across the SMP interconnect between NUMA nodes (e.g., QPI/UPI). (always enabled)

NCCL_SHM_DISABLE
----------------
The ``NCCL_SHM_DISABLE`` variable disables the Shared Memory (SHM) transports. SHM is used between devices when peer-to-peer cannot happen, therefore, host memory is used.  NCCL uses network (IB or IP sockets) to communicate between the CPU sockets when SHM is disabled.

Values accepted
^^^^^^^^^^^^^^^
Define and set to 1 to disable communication through shared memory.

NCCL_SOCKET_IFNAME
------------------

The ``NCCL_SOCKET_IFNAME`` variable specifies which IP interface to use for communication.

Values accepted
^^^^^^^^^^^^^^^
Define to a list of prefixes to filter interfaces to be used by NCCL. For example, eth,ib would only select interfaces
starting with eth or ib. Using the ^ symbol, NCCL will exclude interfaces starting with any prefix in that list. For
example, ^eth,ib would select interfaces not starting with eth or ib.

Note:  By default, the loopback interface (lo) and docker interfaces (docker*) would not be selected unless there are no other interfaces avaible. If you prefer to use lo or docker* over other interfaces, you would need to explicitly select them using ``NCCL_SOCKET_IFNAME``.

.. _NCCL_DEBUG:

NCCL_DEBUG
----------

The ``NCCL_DEBUG`` variable controls the debug information that is displayed from NCCL. This variable is commonly used for debugging.

Values accepted
^^^^^^^^^^^^^^^
VERSION - Prints the NCCL version at the start of the program.

WARN - Prints an explicit error message whenever any NCCL call errors out.

INFO - Prints debug information

NCCL_BUFFSIZE
-------------
The ``NCCL_BUFFSIZE`` variable controls the amount of buffer to share data between 2 GPUs.

Use this variable if you encounter memory constraint issues when using NCCL or you think that a different buffer size would improve performance.

Values accepted
^^^^^^^^^^^^^^^
Default is 4194304 (4 MB).

Values are integers, in bytes. The recommendation is to use powers of 2. For example,  1024 will give a 1K buffer.


NCCL_NTHREADS
-------------
The ``NCCL_NTHREADS`` variable sets the number of CUDA threads per CUDA block. NCCL will launch  one block per communication ring.

Use this variable if you think your GPU clocks are low and you want to increase the number of threads.

You can also use this variable to reduce the number of threads to decrease the GPU workload.

Values accepted
^^^^^^^^^^^^^^^
Default is 256.

The values allowed are 64, 128 and 256.

NCCL_RINGS
----------
The ``NCCL_RINGS`` variable overrides the rings that NCCL forms by default. Rings are sequences of ranks. They can be any permutations of ranks.

NCCL filters out any rings that do not contain the number of ranks in the NCCL communicator. In general, the ring
formation is dependent on the hardware topology connecting the GPUs in your system.

Values accepted
^^^^^^^^^^^^^^^
A list of ranks from 0 to n-1, where n is the number of GPUs in your communicator.

The ranks can be separated by any non-digit character, for example, " ", "-", except "|".

Multiple rings can be specified separated by the pipe character "|".

For example, if you have 4 GPUs in a communicator, you can form communication rings as such: "0 1 2 3  |  3 2 1 0".
This will form two rings, one in each direction.

NCCL_MAX_NRINGS
---------------
(since 2.0.5)

The ``NCCL_MAX_NRINGS`` variable limits the number of rings NCCL can use. Reducing the number of rings also reduces the
number of CUDA blocks used for communication, hence the impact on GPU computing resources.

Values accepted
^^^^^^^^^^^^^^^
Any value above or equal to 1.

NCCL_MIN_NRINGS
---------------
(since 2.2.0)

Controls the minimum number of rings you want NCCL to use. Increasing the number of rings also increases the number of
CUDA blocks NCCL uses, which may be useful to improve performance; however, it uses more CUDA compute resources.

This is especially useful when using aggregated collectives on platforms where NCCL would usually only create one ring.

Values accepted
^^^^^^^^^^^^^^^
Default is platform dependent. Set to a integer value, up to 12 (up to 2.2) or 16 (2.3 and later).

NCCL_CHECKS_DISABLE
-------------------
(since 2.0.5, deprecated in 2.2.12)

Disable argument checks. Checks are useful during development but can increase the latency. They can be disabled to
improve performance in production.

Values accepted
^^^^^^^^^^^^^^^
Default is 0, set to 1 to disable checks.

NCCL_CHECK_POINTERS
-------------------
(since 2.2.12)

Enable checking of the CUDA memory pointers on each collective call. Checks are useful during development but can
increase the latency.

Values accepted
^^^^^^^^^^^^^^^
Default is 0, set to 1 to enable checking.

Setting to 1 restores the original behavior of NCCL prior to 2.2.12.

NCCL_LAUNCH_MODE
----------------
(since 2.1.0)

Controls how NCCL launches CUDA kernels.

Values accepted
^^^^^^^^^^^^^^^
The default value is to use cooperative groups (CUDA 9) for processes managing more than one GPU.

Setting it to PARALLEL uses the previous launch system which can be faster but is prone to deadlocks when one process
manages multiple GPUs.

NCCL_IB_DISABLE
---------------

The ``NCCL_IB_DISABLE`` variable disables the IB/RoCE transport that is to be used by NCCL. Instead, NCCL will fallback to
using IP sockets.

Values accepted
^^^^^^^^^^^^^^^
Define and set to 1 to disable the use of InfiniBand Verbs for communication (and force another method, e.g. IP sockets).

NCCL_IB_HCA
-----------
The ``NCCL_IB_HCA`` variable specifies which RDMA interface to use for communication.

Values accepted
^^^^^^^^^^^^^^^
Define to a list of prefixes to filter interfaces to be used by NCCL.

Using the ^ symbol, NCCL will exclude interfaces starting with any prefix in that list of prefix.
Specific ports can also be specified using ":".

Examples:
mlx5 : Use all ports of all cards starting with mlx5.

mlx5_0:1,mlx5_1:1 : Use ports 1 of cards mlx5_0 and mlx5_1.

^mlx5_1:2 : Do not use port 2 of card mlx5_1.

NCCL_IB_TIMEOUT
---------------
The ``NCCL_IB_TIMEOUT`` variable controls the InfiniBand Verbs Timeout.

The timeout is computed as 4.096 µs * 2 ^ timeout, and the right value is dependent on the size of the network.
Increasing that value can help on very large networks, for example, if NCCL is failing on a call to ibv_poll_cq with
error 12.

For more information, see section 12.7.34 of the InfiniBand specification Volume 1
(https://www.infinibandta.org/ibta-specifications-download) (Local Ack Timeout).

Values accepted
^^^^^^^^^^^^^^^
The default value used by NCCL is 14.

Values can be 1-22.

NCCL_IB_RETRY_CNT
-----------------
(since 2.1.15)

Controls the InfiniBand retry count.

For more information, see section 12.7.38 of the InfiniBand specification Volume 1
(https://www.infinibandta.org/ibta-specifications-download).

Values accepted
^^^^^^^^^^^^^^^
The default value is 7.

NCCL_IB_GID_INDEX
-----------------
(since 2.1.4)

Defines the Global ID index used in RoCE mode. See the show_gids command to set this value.

For more information, see the InfiniBand specification Volume 1
(https://www.infinibandta.org/ibta-specifications-download) or vendor documentation.

Values accepted
^^^^^^^^^^^^^^^
The default value is 0.

NCCL_IB_SL
----------
(since 2.1.4)

Defines the InfiniBand Service Level.

For more information, see the InfiniBand specification Volume 1
(https://www.infinibandta.org/ibta-specifications-download) or vendor documentation.

Values accepted
^^^^^^^^^^^^^^^
The default value is 1.

NCCL_IB_TC
----------
(since 2.1.15)

Defines the InfiniBand traffic class field.

For more information, see the InfiniBand specification Volume 1
(https://www.infinibandta.org/ibta-specifications-download) or vendor documentation.

Values accepted
^^^^^^^^^^^^^^^
The default value is 0.

NCCL_IB_CUDA_SUPPORT
--------------------
(removed in 2.4.0, see NCCL_NET_GDR_LEVEL)

The ``NCCL_IB_CUDA_SUPPORT`` variable is used to force or disable the usage of GPU Direct RDMA.
By default, NCCL enables GPU Direct RDMA, if the topology permits it. This variable can disable this behavior or force
the usage of GPU Direct RDMA in all cases.

Values accepted
^^^^^^^^^^^^^^^
Define and set to 0 to disable GPU Direct RDMA.

Define and set to 1 to force the usage of GPU Direct RDMA.

NCCL_NET_GDR_LEVEL (formerly NCCL_IB_GDR_LEVEL)
-----------------------------------------------
(since 2.3.4. In 2.4.0, NCCL_IB_GDR_LEVEL is renamed NCCL_NET_GDR_LEVEL)

Finely control when to use GPU Direct RDMA between a NIC and a GPU. The level describes the maximum distance between
the NIC and the GPU.

Values accepted
^^^^^^^^^^^^^^^
0 : Never use GPU Direct RDMA. (always disabled)

1 : Use GPU Direct RDMA when GPU and NIC are on the same PCI switch.

2 : Use GPU Direct RDMA when GPU and NIC are connected through PCI switches (potentially multiple hops).

3 : Use GPU Direct RDMA when GPU and NIC are on the same PCI root complex, potentially going through the CPU.

4 : Use GPU Direct RDMA even across PCI root complexes including traversing the interconnect within a NUMA node.

5 : Use GPU Direct RDMA even across the SMP interconnect between NUMA nodes (e.g., QPI/UPI). (always enabled)

NCCL_NET_GDR_READ
-----------------
The ``NCCL_NET_GDR_READ`` variable enables GPU Direct RDMA when sending data. By default, NCCL uses GPU Direct RDMA to
receive data directly in GPU memory. However, when sending data, the data is first stored in CPU memory, then goes to
the InfiniBand card.

Note: Reading directly GPU memory when sending data is known to be slightly slower than reading from CPU memory.

Values accepted
^^^^^^^^^^^^^^^
Default value is 0.

Define and set to 1 to use GPU Direct RDMA to send data to the NIC directly (bypassing CPU).

NCCL_SINGLE_RING_THRESHOLD
--------------------------
(since 2.1.0, deprecated in 2.3)

Set the limit under which NCCL will only use one ring. This will limit bandwidth but improve latency.

Values accepted
^^^^^^^^^^^^^^^
The default value is 262144 (256kB) on GPUs with compute capability 7 and above. Otherwise, the default value is 131072 (128kB).

Values are integers, in bytes.

NCCL_LL_THRESHOLD
-----------------
(since 2.1.0)

Set the size limit under which NCCL uses low-latency algorithms.

Values accepted
^^^^^^^^^^^^^^^
Default is 16384 (up to 2.2) or is dependent on the number of ranks (2.3 and later).

Values are integers, in bytes.

NCCL_TREE_THRESHOLD
-------------------
(since 2.4.0)

Set the size limit under which NCCL uses tree algorithms instead of rings.

Values accepted
^^^^^^^^^^^^^^^
Default is dependent on the number of ranks.

Values are integers, in bytes.

NCCL_IGNORE_CPU_AFFINITY
-------------------
(since 2.4.6)

Flag to cause NCCL to ignore the job's supplied CPU affinity and use the GPU affinity only.

Values accepted
^^^^^^^^^^^^^^^
Default is 0, set to 1 to cause NCCL to ignore the job's supplied CPU affinity.


NCCL_DEBUG_FILE
---------------
(since 2.2.12)

Direct the NCCL debug logging output to a file. The filename format can be set to *filename.%h.%p* where *%h* is replaced with the
hostname and *%p* is replaced with the process PID.

Values accepted
^^^^^^^^^^^^^^^
The default output file is stdout unless this env variable is set.

Setting ``NCCL_DEBUG_FILE`` will cause NCCL to create and overwrite any previous files of that name.

Note: If the filename is not unique across all the job processes, then the output may be lost or corrupted.

NCCL_DEBUG_SUBSYS
-----------------
(since 2.3.4)

Filter the ``NCCL_DEBUG=INFO`` output based on subsystem. A comma separated list of the subsystems to include in the NCCL
debug log traces.

Prefixing the subsystem name with ‘^’ will disable the logging for that subsystem.

Values accepted
^^^^^^^^^^^^^^^
The default value is INIT.

Supported subsystem names are INIT (stands for initialization),COLL (stands for collectives), P2P (stands for
peer-to-peer), SHM (stands for shared memory), NET (stands for network) and ALL (includes every subsystem).
