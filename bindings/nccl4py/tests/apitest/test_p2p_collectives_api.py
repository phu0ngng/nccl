"""API tests for NCCL collective operations.

Tests all collective operations (reduce, broadcast, gather,
reduce_scatter, alltoall, scatter) with real NCCL across multiple ranks.
Validates data correctness by initializing with known values and checking results.
"""
import pytest
import numpy as np

import nccl.core as nccl
from conftest import requires_nccl_version

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _allocate_buffer(data: list, dtype: str, allocator: str):
    """Allocate and initialize buffer using specified allocator.

    Args:
        data: List of values to initialize buffer with
        dtype: Data type string (e.g., "float32", "int32")
        allocator: One of "cupy", "torch", "interop.cupy", "interop.torch"

    Returns:
        Allocated and initialized array/tensor
    """
    if allocator == "cupy":
        if not HAS_CUPY:
            pytest.skip("CuPy not installed")
        arr = cp.array(data, dtype=dtype)
        cp.cuda.Stream.null.synchronize()
        return arr
    elif allocator == "torch":
        if not HAS_TORCH:
            pytest.skip("PyTorch not installed")
        tensor = torch.tensor(data, dtype=getattr(torch, dtype), device='cuda')
        torch.cuda.synchronize()
        return tensor
    elif allocator == "interop.cupy":
        if not HAS_CUPY:
            pytest.skip("CuPy not installed")
        arr = nccl.cupy.empty(len(data), dtype=dtype)
        arr[:] = cp.array(data, dtype=dtype)
        cp.cuda.Stream.null.synchronize()
        return arr
    elif allocator == "interop.torch":
        if not HAS_TORCH:
            pytest.skip("PyTorch not installed")
        tensor = nccl.torch.empty(len(data), dtype=getattr(torch, dtype))
        tensor[:] = torch.tensor(data, dtype=getattr(torch, dtype), device='cuda')
        torch.cuda.synchronize()
        return tensor
    else:
        pytest.skip(f"Unknown allocator: {allocator}")


def _allocate_empty_buffer(shape: int | tuple, dtype: str, allocator: str):
    """Allocate empty buffer without initialization.

    Args:
        shape: Buffer shape (int or tuple of ints)
        dtype: Data type string
        allocator: One of "cupy", "torch", "interop.cupy", "interop.torch"

    Returns:
        Allocated but uninitialized array/tensor
    """
    if allocator == "cupy":
        if not HAS_CUPY:
            pytest.skip("CuPy not installed")
        return cp.empty(shape, dtype=dtype)
    elif allocator == "torch":
        if not HAS_TORCH:
            pytest.skip("PyTorch not installed")
        size = shape if isinstance(shape, int) else shape[0] if len(shape) == 1 else shape
        return torch.empty(size, dtype=getattr(torch, dtype), device='cuda')
    elif allocator == "interop.cupy":
        if not HAS_CUPY:
            pytest.skip("CuPy not installed")
        return nccl.cupy.empty(shape, dtype=dtype)
    elif allocator == "interop.torch":
        if not HAS_TORCH:
            pytest.skip("PyTorch not installed")
        return nccl.torch.empty(shape, dtype=getattr(torch, dtype))
    else:
        pytest.skip(f"Unknown allocator: {allocator}")


def _sync(allocator: str):
    """Synchronize buffer with proper allocator."""
    if allocator in ["cupy", "interop.cupy"]:
        cp.cuda.Stream.null.synchronize()
    else:
        torch.cuda.synchronize()

def _to_numpy(buf):
    """Convert buffer to numpy array with proper synchronization.

    Args:
        buf: CuPy array or PyTorch tensor

    Returns:
        numpy.ndarray
    """
    if hasattr(buf, 'get'):  # CuPy array
        return buf.get()
    else:  # PyTorch tensor
        return buf.cpu().numpy()


# --- Point-to-Point Tests ---

@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("allocator", ["cupy", "torch", "interop.cupy", "interop.torch"])
def test_send_recv(nccl_comm, rank_info, allocator):
    """Test send/recv with different allocators."""
    if rank_info.nccl_size % 2 != 0 and rank_info.nccl_rank == rank_info.nccl_size - 1:
        pytest.skip("Odd number of ranks, skip last rank")

    count = 10
    self_rank = rank_info.nccl_rank
    peer_rank = self_rank + 1 if self_rank % 2 == 0 else self_rank - 1

    # Prepare send data and expected receive data
    send_list = [(self_rank + 1) * 100 + i for i in range(count)]
    expected_list = [(peer_rank + 1) * 100 + i for i in range(count)]

    # Allocate buffers
    send_data = _allocate_buffer(send_list, "float32", allocator)
    recv_data = _allocate_empty_buffer(count, "float32", allocator)
    expected = np.array(expected_list, dtype=np.float32)

    # Send/recv (even ranks send first)
    if self_rank % 2 == 0:
        nccl_comm.send(send_data, peer_rank)
        nccl_comm.recv(recv_data, peer_rank)
    else:
        nccl_comm.recv(recv_data, peer_rank)
        nccl_comm.send(send_data, peer_rank)

    _sync(allocator)
    result = _to_numpy(recv_data)
    assert np.allclose(result, expected), f"{allocator}: expected {expected}, got {result}"

    # Test with group
    send_list2 = [(self_rank + 1) * 1000 + i for i in range(count)]
    expected_list2 = [(peer_rank + 1) * 1000 + i for i in range(count)]

    send_data = _allocate_buffer(send_list2, "float32", allocator)
    expected = np.array(expected_list2, dtype=np.float32)

    with nccl.group():
        nccl_comm.send(send_data, peer_rank)
        nccl_comm.recv(recv_data, peer_rank)

    _sync(allocator)
    result = _to_numpy(recv_data)
    assert np.allclose(result, expected), f"{allocator} (group): expected {expected}, got {result}"


# --- Collective Tests ---

@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("allocator", ["cupy", "torch", "interop.cupy", "interop.torch"])
def test_all_reduce(nccl_comm, rank_info, allocator):
    """Test reduce (AllReduce mode) with different buffer allocators."""
    count = 10

    # Prepare send data and expected result as lists
    send_list = [rank_info.nccl_rank * i for i in range(count)]
    expected_list = [(rank_info.nccl_size - 1) * rank_info.nccl_size / 2 * i for i in range(count)]

    # Allocate buffers
    send_data = _allocate_buffer(send_list, "float32", allocator)
    recv_data = _allocate_empty_buffer(count, "float32", allocator)
    expected = np.array(expected_list, dtype=np.float32)

    # Perform AllReduce
    nccl_comm.allreduce(send_data, recv_data, nccl.SUM)
    _sync(allocator)
    result = _to_numpy(recv_data)
    assert np.allclose(result, expected), f"{allocator}: expected {expected}, got {result}"

    # Perform AllReduce with explicit stream
    nccl_comm.allreduce(send_data, recv_data, nccl.SUM, stream=0)
    _sync(allocator)
    result = _to_numpy(recv_data)
    assert np.allclose(result, expected), f"{allocator} (stream=0): expected {expected}, got {result}"

    # Perform in-place AllReduce
    nccl_comm.allreduce(send_data, send_data, nccl.SUM)
    _sync(allocator)
    result = _to_numpy(send_data)
    assert np.allclose(result, expected), f"{allocator} (in-place): expected {expected}, got {result}"


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("allocator", ["cupy", "torch", "interop.cupy", "interop.torch"])
def test_broadcast(nccl_comm, rank_info, allocator):
    """Test broadcast with different allocators."""
    root = 0
    count = 10

    # Prepare send data and expected result as lists
    if rank_info.nccl_rank == root:
        send_list = [(root + 1) * 100 + i for i in range(count)]
    else:
        send_list = [0] * count
    expected_list = [(root + 1) * 100 + i for i in range(count)]

    # Allocate buffers
    send_data = _allocate_buffer(send_list, "float32", allocator)
    recv_data = _allocate_empty_buffer(count, "float32", allocator)
    expected = np.array(expected_list, dtype=np.float32)

    # Perform broadcast
    nccl_comm.broadcast(send_data, recv_data, root=root)
    _sync(allocator)
    result = _to_numpy(recv_data)
    assert np.allclose(result, expected), f"{allocator}: expected {expected}, got {result}"

    # Perform broadcast with explicit stream
    nccl_comm.broadcast(send_data, recv_data, root=root, stream=0)
    _sync(allocator)
    result = _to_numpy(recv_data)
    assert np.allclose(result, expected), f"{allocator} (stream=0): expected {expected}, got {result}"

    # Perform in-place broadcast
    nccl_comm.broadcast(send_data, send_data, root=root)
    _sync(allocator)
    result = _to_numpy(send_data)
    assert np.allclose(result, expected), f"{allocator} (in-place): expected {expected}, got {result}"

    # Perform broadcast with invalid sendbuf for non-root ranks
    if rank_info.nccl_rank == root:
        nccl_comm.broadcast(send_data, recv_data, root=root)
    else:
        nccl_comm.broadcast(None, recv_data, root=root)
    _sync(allocator)
    result = _to_numpy(recv_data)
    assert np.allclose(result, expected), f"{allocator} (None sendbuf): expected {expected}, got {result}"


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("allocator", ["cupy", "torch", "interop.cupy", "interop.torch"])
def test_reduce(nccl_comm, rank_info, allocator):
    """Test reduce with different allocators."""
    root = 0
    count = 10

    # Prepare send data and expected result as lists
    send_list = [rank_info.nccl_rank * i for i in range(count)]
    expected_list = [(rank_info.nccl_size - 1) * rank_info.nccl_size / 2 * i for i in range(count)]

    # Allocate buffers
    send_data = _allocate_buffer(send_list, "float32", allocator)
    recv_data = _allocate_empty_buffer(count, "float32", allocator)
    expected = np.array(expected_list, dtype=np.float32)

    # Perform reduce
    nccl_comm.reduce(send_data, recv_data, nccl.SUM, root=root)
    _sync(allocator)
    if rank_info.nccl_rank == root:
        result = _to_numpy(recv_data)
        assert np.allclose(result, expected), f"{allocator}: expected {expected}, got {result}"

    # Perform reduce with explicit stream
    nccl_comm.reduce(send_data, recv_data, nccl.SUM, root=root, stream=0)
    _sync(allocator)
    if rank_info.nccl_rank == root:
        result = _to_numpy(recv_data)
        assert np.allclose(result, expected), f"{allocator} (stream=0): expected {expected}, got {result}"

    # Perform reduce with invalid recvbuf for non-root ranks
    if rank_info.nccl_rank == root:
        nccl_comm.reduce(send_data, recv_data, nccl.SUM, root=root)
    else:
        nccl_comm.reduce(send_data, None, nccl.SUM, root=root)
    _sync(allocator)
    if rank_info.nccl_rank == root:
        result = _to_numpy(recv_data)
        assert np.allclose(result, expected), f"{allocator} (None recvbuf): expected {expected}, got {result}"

    # Perform in-place reduce
    nccl_comm.reduce(send_data, send_data, nccl.SUM, root=root)
    _sync(allocator)
    if rank_info.nccl_rank == root:
        result = _to_numpy(send_data)
        assert np.allclose(result, expected), f"{allocator} (in-place): expected {expected}, got {result}"


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("allocator", ["cupy", "torch", "interop.cupy", "interop.torch"])
def test_all_gather(nccl_comm, rank_info, allocator):
    """Test gather (AllGather mode) with different allocators."""
    count = 10

    # Prepare send data and expected result as lists
    send_list = [rank_info.nccl_rank * 100 + i for i in range(count)]
    expected_list = []
    for r in range(rank_info.nccl_size):
        for i in range(count):
            expected_list.append(r * 100 + i)

    # Allocate buffers
    send_data = _allocate_buffer(send_list, "float32", allocator)
    recv_data = _allocate_empty_buffer(count * rank_info.nccl_size, "float32", allocator)
    expected = np.array(expected_list, dtype=np.float32)

    # Perform AllGather
    nccl_comm.allgather(send_data, recv_data)
    _sync(allocator)
    result = _to_numpy(recv_data)
    assert np.allclose(result, expected), f"{allocator}: expected {expected}, got {result}"

    # Perform AllGather with explicit stream
    nccl_comm.allgather(send_data, recv_data, stream=0)
    _sync(allocator)
    result = _to_numpy(recv_data)
    assert np.allclose(result, expected), f"{allocator} (stream=0): expected {expected}, got {result}"

    # Perform in-place AllGather: sendbuf == recvbuf + rank * sendcount
    in_place_send_data = recv_data[rank_info.nccl_rank * count : (rank_info.nccl_rank + 1) * count]
    in_place_send_data[:] = send_data
    nccl_comm.allgather(in_place_send_data, recv_data)
    _sync(allocator)
    result = _to_numpy(recv_data)
    assert np.allclose(result, expected), f"{allocator} (in-place): expected {expected}, got {result}"


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("allocator", ["cupy", "torch", "interop.cupy", "interop.torch"])
def test_reduce_scatter(nccl_comm, rank_info, allocator):
    """Test reduce_scatter with different allocators."""
    count = 10

    # Prepare send data and expected result as lists
    send_list = [i * rank_info.nccl_rank for i in range(count * rank_info.nccl_size)]
    expected_list = [(rank_info.nccl_size - 1) * rank_info.nccl_size / 2 * (count * rank_info.nccl_rank + i) for i in range(count)]

    # Allocate buffers
    send_data = _allocate_buffer(send_list, "float32", allocator)
    recv_data = _allocate_empty_buffer(count, "float32", allocator)
    expected = np.array(expected_list, dtype=np.float32)

    # Perform reduce_scatter
    nccl_comm.reduce_scatter(send_data, recv_data, nccl.SUM)
    _sync(allocator)
    result = _to_numpy(recv_data)
    assert np.allclose(result, expected), f"{allocator}: expected {expected}, got {result}"

    # Perform reduce_scatter with explicit stream
    nccl_comm.reduce_scatter(send_data, recv_data, nccl.SUM, stream=0)
    _sync(allocator)
    result = _to_numpy(recv_data)
    assert np.allclose(result, expected), f"{allocator} (stream=0): expected {expected}, got {result}"

    # Perform in-place reduce_scatter: recvbuf == sendbuf + rank * recvcount
    in_place_recv = send_data[rank_info.nccl_rank * count : (rank_info.nccl_rank + 1) * count]
    nccl_comm.reduce_scatter(send_data, in_place_recv, nccl.SUM)
    _sync(allocator)
    result = _to_numpy(in_place_recv)
    assert np.allclose(result, expected), f"{allocator} (in-place): expected {expected}, got {result}"


@requires_nccl_version("2.28.3")
@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("allocator", ["cupy", "torch", "interop.cupy", "interop.torch"])
def test_all_to_all(nccl_comm, rank_info, allocator):
    """Test alltoall with different allocators."""
    count = 10

    # Prepare send data: each rank sends different data to each peer
    # Rank i sends to rank j: (i+1)*100 + j*count + offset
    send_list = []
    for r in range(rank_info.nccl_size):
        for i in range(count):
            send_list.append((rank_info.nccl_rank + 1) * 100 + r * count + i)

    # Expected: rank receives from each peer r: (r+1)*100 + my_rank*count + offset
    expected_list = []
    for r in range(rank_info.nccl_size):
        for i in range(count):
            expected_list.append((r + 1) * 100 + rank_info.nccl_rank * count + i)

    # Allocate buffers
    send_data = _allocate_buffer(send_list, "float32", allocator)
    recv_data = _allocate_empty_buffer(count * rank_info.nccl_size, "float32", allocator)
    expected = np.array(expected_list, dtype=np.float32)

    # Perform alltoall
    nccl_comm.alltoall(send_data, recv_data)
    _sync(allocator)
    result = _to_numpy(recv_data)
    assert np.allclose(result, expected), f"{allocator}: expected {expected}, got {result}"

    # Perform alltoall with explicit stream
    nccl_comm.alltoall(send_data, recv_data, stream=0)
    _sync(allocator)
    result = _to_numpy(recv_data)
    assert np.allclose(result, expected), f"{allocator} (stream=0): expected {expected}, got {result}"


@requires_nccl_version("2.28.3")
@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("allocator", ["cupy", "torch", "interop.cupy", "interop.torch"])
def test_gather(nccl_comm, rank_info, allocator):
    """Test gather with different allocators."""
    root = 0
    count = 10

    # Prepare send data and expected result as lists
    send_list = [rank_info.nccl_rank * 100 + i for i in range(count)]
    expected_list = []
    for r in range(rank_info.nccl_size):
        for i in range(count):
            expected_list.append(r * 100 + i)

    # Allocate buffers
    send_data = _allocate_buffer(send_list, "float32", allocator)
    recv_data = _allocate_empty_buffer(count * rank_info.nccl_size, "float32", allocator)
    expected = np.array(expected_list, dtype=np.float32)

    # Perform gather
    nccl_comm.gather(send_data, recv_data, root=root)
    _sync(allocator)
    if rank_info.nccl_rank == root:
        result = _to_numpy(recv_data)
        assert np.allclose(result, expected), f"{allocator}: expected {expected}, got {result}"

    # Perform gather with explicit stream
    nccl_comm.gather(send_data, recv_data, root=root, stream=0)
    _sync(allocator)
    if rank_info.nccl_rank == root:
        result = _to_numpy(recv_data)
        assert np.allclose(result, expected), f"{allocator} (stream=0): expected {expected}, got {result}"

    # Perform gather with invalid recvbuf for non-root ranks
    if rank_info.nccl_rank == root:
        nccl_comm.gather(send_data, recv_data, root=root)
    else:
        nccl_comm.gather(send_data, None, root=root)
    _sync(allocator)
    if rank_info.nccl_rank == root:
        result = _to_numpy(recv_data)
        assert np.allclose(result, expected), f"{allocator} (None recvbuf): expected {expected}, got {result}"

    # Perform in-place gather: sendbuf == recvbuf + rank * sendcount
    # Need fresh recv_data for in-place test
    in_place_recv_data = _allocate_empty_buffer(count * rank_info.nccl_size, "float32", allocator)
    in_place_send_data = in_place_recv_data[rank_info.nccl_rank * count : (rank_info.nccl_rank + 1) * count]
    in_place_send_data[:] = send_data
    nccl_comm.gather(in_place_send_data, in_place_recv_data, root=root)
    _sync(allocator)
    if rank_info.nccl_rank == root:
        result = _to_numpy(in_place_recv_data)
        assert np.allclose(result, expected), f"{allocator} (in-place): expected {expected}, got {result}"


@requires_nccl_version("2.28.3")
@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("allocator", ["cupy", "torch", "interop.cupy", "interop.torch"])
def test_scatter(nccl_comm, rank_info, allocator):
    """Test scatter with different allocators."""
    root = 0
    count = 10

    # Prepare send data and expected result as lists
    if rank_info.nccl_rank == root:
        send_list = []
        for r in range(rank_info.nccl_size):
            for i in range(count):
                send_list.append(r * 100 + i)
    else:
        send_list = [0] * (count * rank_info.nccl_size)
    expected_list = [rank_info.nccl_rank * 100 + i for i in range(count)]

    # Allocate buffers
    send_data = _allocate_buffer(send_list, "float32", allocator)
    recv_data = _allocate_empty_buffer(count, "float32", allocator)
    expected = np.array(expected_list, dtype=np.float32)

    # Perform scatter
    nccl_comm.scatter(send_data, recv_data, root=root)
    _sync(allocator)
    result = _to_numpy(recv_data)
    assert np.allclose(result, expected), f"{allocator}: expected {expected}, got {result}"

    # Perform scatter with explicit stream
    nccl_comm.scatter(send_data, recv_data, root=root, stream=0)
    _sync(allocator)
    result = _to_numpy(recv_data)
    assert np.allclose(result, expected), f"{allocator} (stream=0): expected {expected}, got {result}"

    # Perform scatter with invalid sendbuf for non-root ranks
    if rank_info.nccl_rank == root:
        nccl_comm.scatter(send_data, recv_data, root=root)
    else:
        nccl_comm.scatter(None, recv_data, root=root)
    _sync(allocator)
    result = _to_numpy(recv_data)
    assert np.allclose(result, expected), f"{allocator} (None sendbuf): expected {expected}, got {result}"

    # Perform in-place scatter: recvbuf == sendbuf + rank * recvcount
    in_place_recv_data = send_data[rank_info.nccl_rank * count : (rank_info.nccl_rank + 1) * count]
    nccl_comm.scatter(send_data, in_place_recv_data, root=root)
    _sync(allocator)
    result = _to_numpy(in_place_recv_data)
    assert np.allclose(result, expected), f"{allocator} (in-place): expected {expected}, got {result}"


@requires_nccl_version("2.29.3")
@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("allocator", ["cupy"])
def test_signal_basic(nccl_comm, rank_info, allocator):
    """Tests basic signaling between paired ranks.

    Each even rank pairs with the next odd rank (0<->1, 2<->3, etc.).
    Both ranks signal each other and wait for the peer's signal.
    """
    self_rank = rank_info.nccl_rank

    if rank_info.nccl_size % 2 != 0 and self_rank == rank_info.nccl_size - 1:
        pytest.skip("Odd number of ranks, skip last rank")

    peer_rank = self_rank + 1 if self_rank % 2 == 0 else self_rank - 1

    # Send a signal to the peer
    nccl_comm.signal(peer_rank, stream=0)

    # Create a wait descriptor and wait for the signal from peer
    desc = nccl.WaitSignalDesc(peer_rank)
    nccl_comm.wait_signal(desc, stream=0)

    _sync(allocator)


@requires_nccl_version("2.29.3")
@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("allocator", ["cupy"])
def test_signal_multiple(nccl_comm, rank_info, allocator):
    """Tests multiple signals between paired ranks.

    Each rank sends multiple signals to its peer and waits for the
    same number of signals from the peer using op_count.
    """
    self_rank = rank_info.nccl_rank

    if rank_info.nccl_size % 2 != 0 and self_rank == rank_info.nccl_size - 1:
        pytest.skip("Odd number of ranks, skip last rank")

    peer_rank = self_rank + 1 if self_rank % 2 == 0 else self_rank - 1
    num_signals = 5

    # Send multiple signals to the peer
    for _ in range(num_signals):
        nccl_comm.signal(peer_rank, stream=0)

    # Wait for all signals at once using op_count
    desc = nccl.WaitSignalDesc(peer_rank, num_signals)
    nccl_comm.wait_signal(desc, stream=0)

    _sync(allocator)


@requires_nccl_version("2.29.3")
@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("allocator", ["cupy"])
def test_signal_with_group(nccl_comm, rank_info, allocator):
    """Tests signal/wait_signal within a group context.

    Wraps signal and wait_signal operations in a group to batch them.
    """
    self_rank = rank_info.nccl_rank

    if rank_info.nccl_size % 2 != 0 and self_rank == rank_info.nccl_size - 1:
        pytest.skip("Odd number of ranks, skip last rank")

    peer_rank = self_rank + 1 if self_rank % 2 == 0 else self_rank - 1

    # Use group to batch signal and wait_signal
    with nccl.group():
        nccl_comm.signal(peer_rank, stream=0)
        desc = nccl.WaitSignalDesc(peer_rank)
        nccl_comm.wait_signal(desc, stream=0)

    _sync(allocator)


@requires_nccl_version("2.29.3")
@pytest.mark.mpi(min_size=3)
@pytest.mark.parametrize("allocator", ["cupy"])
def test_signal_ring(nccl_comm, rank_info, allocator):
    """Tests ring-style signaling where each rank signals the next."""
    self_rank = rank_info.nccl_rank
    nranks = rank_info.nccl_size

    next_rank = (self_rank + 1) % nranks
    prev_rank = (self_rank - 1 + nranks) % nranks

    # Signal the next rank in the ring
    nccl_comm.signal(next_rank, stream=0)

    # Wait for signal from the previous rank
    desc = nccl.WaitSignalDesc(prev_rank)
    nccl_comm.wait_signal(desc, stream=0)

    _sync(allocator)


@requires_nccl_version("2.29.3")
@pytest.mark.mpi(min_size=3)
@pytest.mark.parametrize("allocator", ["cupy"])
def test_signal_multiple_descriptors(nccl_comm, rank_info, allocator):
    """Tests waiting for signals from multiple peers using multiple descriptors.

    Each rank signals all other ranks, then waits for signals from all others
    using a list of wait descriptors (one per peer).
    """
    self_rank = rank_info.nccl_rank
    nranks = rank_info.nccl_size

    # Signal all other ranks
    for peer in range(nranks):
        if peer != self_rank:
            nccl_comm.signal(peer, stream=0)

    # Wait for signals from all other ranks using multiple descriptors
    descs = [
        nccl.WaitSignalDesc(peer)
        for peer in range(nranks)
        if peer != self_rank
    ]
    nccl_comm.wait_signal(descs, stream=0)

    _sync(allocator)


# --- PutSignal tests ---

@requires_nccl_version("2.29.3")
@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("allocator", ["interop.cupy"])
def test_put_signal_basic(nccl_comm, rank_info, allocator):
    """Test put_signal between paired ranks with direct destination window."""
    self_rank = rank_info.nccl_rank
    nranks = rank_info.nccl_size

    if nranks % 2 != 0 and self_rank == nranks - 1:
        pytest.skip("Odd number of ranks, skip last rank")

    peer_rank = self_rank + 1 if self_rank % 2 == 0 else self_rank - 1
    count = 8

    # Source and destination buffers; both must be in symmetric windows for put_signal.
    send_data = _allocate_buffer(
        [peer_rank * 100 + i for i in range(count)],
        "float32",
        allocator,
    )
    recv_data = _allocate_buffer([0] * count, "float32", allocator)

    # Register send then recv (same order on all ranks for symmetric handles).
    send_win = nccl_comm.register_window(
        send_data, flags=nccl.WindowFlag.CollSymmetric
    )
    recv_win = nccl_comm.register_window(
        recv_data, flags=nccl.WindowFlag.CollSymmetric
    )
    if send_win is None or recv_win is None:
        pytest.skip("Window registration not supported.")

    nccl_comm.put_signal(
        local_buffer=send_data, peer=peer_rank, peer_window=recv_win
    )
    nccl_comm.wait_signal(nccl.WaitSignalDesc(peer_rank))

    _sync(allocator)

    # Peer wrote its payload into this rank's recv window (peer sent [self_rank*100+i]).
    expected = np.array(
        [self_rank * 100 + i for i in range(count)], dtype=np.float32
    )
    result = _to_numpy(recv_data)
    assert np.allclose(result, expected), (
        f"{allocator}: expected {expected}, got {result}"
    )


@requires_nccl_version("2.29.3")
@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("allocator", ["interop.cupy"])
def test_put_signal_with_offset(nccl_comm, rank_info, allocator):
    """Test put_signal with non-zero peer window offset."""
    self_rank = rank_info.nccl_rank
    nranks = rank_info.nccl_size

    if nranks % 2 != 0 and self_rank == nranks - 1:
        pytest.skip("Odd number of ranks, skip last rank")

    peer_rank = self_rank + 1 if self_rank % 2 == 0 else self_rank - 1
    count = 6
    offset = 3
    total_count = count + offset + 2

    send_list = [self_rank * 1000 + i for i in range(count)]
    send_data = _allocate_buffer(send_list, "float32", allocator)
    recv_data = _allocate_buffer([-1] * total_count, "float32", allocator)

    send_win = nccl_comm.register_window(
        send_data, flags=nccl.WindowFlag.CollSymmetric
    )
    recv_win = nccl_comm.register_window(
        recv_data, flags=nccl.WindowFlag.CollSymmetric
    )
    if send_win is None or recv_win is None:
        pytest.skip("Window registration not supported.")

    nccl_comm.put_signal(
        local_buffer=send_data,
        peer=peer_rank,
        peer_window=recv_win,
        peer_window_offset=offset,
    )
    nccl_comm.wait_signal(nccl.WaitSignalDesc(peer_rank))

    _sync(allocator)

    expected = np.full(total_count, -1, dtype=np.float32)
    expected[offset : offset + count] = np.array(
        [peer_rank * 1000 + i for i in range(count)], dtype=np.float32
    )
    result = _to_numpy(recv_data)
    assert np.allclose(result, expected), (
        f"{allocator}: expected {expected}, got {result}"
    )


@requires_nccl_version("2.29.3")
@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("allocator", ["interop.cupy"])
def test_put_signal_multiple(nccl_comm, rank_info, allocator):
    """Test multiple put_signal calls in a group, wait for all with op_count."""
    self_rank = rank_info.nccl_rank
    nranks = rank_info.nccl_size

    if nranks % 2 != 0 and self_rank == nranks - 1:
        pytest.skip("Odd number of ranks, skip last rank")

    peer_rank = self_rank + 1 if self_rank % 2 == 0 else self_rank - 1
    count = 4
    num_puts = 3

    send_data = _allocate_buffer(
        [peer_rank * 100 + i for i in range(count)], "float32", allocator
    )
    recv_data = _allocate_buffer([0] * (count * num_puts), "float32", allocator)

    send_win = nccl_comm.register_window(
        send_data, flags=nccl.WindowFlag.CollSymmetric
    )
    recv_win = nccl_comm.register_window(
        recv_data, flags=nccl.WindowFlag.CollSymmetric
    )
    if send_win is None or recv_win is None:
        pytest.skip("Window registration not supported.")

    with nccl.group():
        for p in range(num_puts):
            nccl_comm.put_signal(
                local_buffer=send_data,
                peer=peer_rank,
                peer_window=recv_win,
                peer_window_offset=p * count,
            )
    nccl_comm.wait_signal(
        nccl.WaitSignalDesc(peer_rank, num_puts), stream=0
    )

    _sync(allocator)

    # Peer sent [self_rank*100+i] into our recv window.
    expected_chunk = np.array(
        [self_rank * 100 + i for i in range(count)], dtype=np.float32
    )
    result = _to_numpy(recv_data)
    for p in range(num_puts):
        chunk = result[p * count : (p + 1) * count]
        assert np.allclose(chunk, expected_chunk), (
            f"{allocator} put {p}: expected {expected_chunk}, got {chunk}"
        )


@requires_nccl_version("2.29.3")
@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("allocator", ["interop.cupy"])
def test_put_signal_ping_pong(nccl_comm, rank_info, allocator):
    """Test ping-pong: rank0 put then wait, rank1 wait then put; two iterations with verification."""
    self_rank = rank_info.nccl_rank
    nranks = rank_info.nccl_size

    if nranks != 2:
        pytest.skip("Ping-pong test requires exactly 2 ranks")

    peer_rank = 1 - self_rank
    count = 8
    num_iterations = 2

    # Single send and recv buffers (symmetric reg requires fixed layout per rank).
    send_data = _allocate_buffer([0] * count, "float32", allocator)
    recv_data = _allocate_buffer([0] * count, "float32", allocator)

    send_win = nccl_comm.register_window(
        send_data, flags=nccl.WindowFlag.CollSymmetric
    )
    recv_win = nccl_comm.register_window(
        recv_data, flags=nccl.WindowFlag.CollSymmetric
    )
    if send_win is None or recv_win is None:
        pytest.skip("Window registration not supported.")

    for iteration in range(num_iterations):
        # Fill send buffer for this iteration.
        payload = [self_rank * 1000 + iteration * 10 + i for i in range(count)]
        if hasattr(send_data, "get"):  # CuPy
            send_data.set(np.array(payload, dtype=np.float32))
        else:
            send_data.copy_(torch.tensor(payload, dtype=torch.float32, device="cuda"))
        _sync(allocator)

        if self_rank == 0:
            nccl_comm.put_signal(
                local_buffer=send_data,
                peer=peer_rank,
                peer_window=recv_win,
                stream=0,
            )
            nccl_comm.wait_signal(
                nccl.WaitSignalDesc(peer_rank), stream=0
            )
        else:
            nccl_comm.wait_signal(
                nccl.WaitSignalDesc(peer_rank), stream=0
            )
            nccl_comm.put_signal(
                local_buffer=send_data,
                peer=peer_rank,
                peer_window=recv_win,
                stream=0,
            )
        _sync(allocator)

    # After two rounds: recv_data holds the last payload from peer (iteration 1); peer sent [self_rank*1000+10+i]
    expected = np.array(
        [peer_rank * 1000 + (num_iterations - 1) * 10 + i for i in range(count)],
        dtype=np.float32
    )
    result = _to_numpy(recv_data)
    assert np.allclose(result, expected), (
        f"{allocator}: expected {expected}, got {result}"
    )


# --- Buffer specification and slicing tests ---

@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("allocator", ["cupy", "torch", "interop.cupy", "interop.torch"])
def test_sliced_buffer(nccl_comm, rank_info, allocator):
    """Test sliced buffers and dtype override with reduce (AllReduce mode)."""

    # Create base array with rank-specific data
    base_data = [rank_info.nccl_rank * 100 + i for i in range(20)]
    full_array = _allocate_buffer(base_data, "float32", allocator)

    # Test 1: full_array - Use full array, count = 20
    recv_buf = _allocate_empty_buffer(20, "float32", allocator)
    expected = np.array([sum(r * 100 + i for r in range(rank_info.nccl_size)) for i in range(20)], dtype=np.float32)
    nccl_comm.reduce(full_array, recv_buf, nccl.SUM)
    _sync(allocator)
    result = _to_numpy(recv_buf)
    assert np.allclose(result, expected), f"{allocator} (full_array): expected {expected}, got {result}"

    # Test 2: slice_from_zero - Slice [0:10], count = 10
    send_slice = full_array[0:10]
    recv_buf = _allocate_empty_buffer(10, "float32", allocator)
    expected = np.array([sum(r * 100 + i for r in range(rank_info.nccl_size)) for i in range(10)], dtype=np.float32)
    nccl_comm.reduce(send_slice, recv_buf, nccl.SUM)
    _sync(allocator)
    result = _to_numpy(recv_buf)
    assert np.allclose(result, expected), f"{allocator} (slice_from_zero): expected {expected}, got {result}"

    # Test 3: slice_from_middle - Slice [5:15], count = 10
    send_slice = full_array[5:15]
    recv_buf = _allocate_empty_buffer(10, "float32", allocator)
    expected = np.array([sum(r * 100 + (i + 5) for r in range(rank_info.nccl_size)) for i in range(10)], dtype=np.float32)
    nccl_comm.reduce(send_slice, recv_buf, nccl.SUM)
    _sync(allocator)
    result = _to_numpy(recv_buf)
    assert np.allclose(result, expected), f"{allocator} (slice_from_middle): expected {expected}, got {result}"
