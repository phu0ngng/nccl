"""API tests for NCCL collective operations.

Tests all collective operations (all_reduce, broadcast, reduce, all_gather,
reduce_scatter, all_to_all, gather, scatter) with real NCCL across multiple ranks.
Validates data correctness by initializing with known values and checking results.
"""
import pytest
import numpy as np

import nccl.core as nccl

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
    assert np.array_equal(result, expected), f"{allocator}: expected {expected}, got {result}"

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
    assert np.array_equal(result, expected), f"{allocator} (group): expected {expected}, got {result}"


# --- Collective Tests ---

@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("allocator", ["cupy", "torch", "interop.cupy", "interop.torch"])
def test_all_reduce(nccl_comm, rank_info, allocator):
    """Test all_reduce with different buffer allocators."""
    count = 10

    # Prepare send data and expected result as lists
    send_list = [rank_info.nccl_rank * i for i in range(count)]
    expected_list = [(rank_info.nccl_size - 1) * rank_info.nccl_size / 2 * i for i in range(count)]

    # Allocate buffers
    send_data = _allocate_buffer(send_list, "float32", allocator)
    recv_data = _allocate_empty_buffer(count, "float32", allocator)
    expected = np.array(expected_list, dtype=np.float32)

    # Perform all_reduce
    nccl_comm.all_reduce(send_data, recv_data, nccl.SUM)
    _sync(allocator)
    result = _to_numpy(recv_data)
    assert np.array_equal(result, expected), f"{allocator}: expected {expected}, got {result}"

    # Perform all_reduce with explicit stream
    nccl_comm.all_reduce(send_data, recv_data, nccl.SUM, stream=0)
    _sync(allocator)
    result = _to_numpy(recv_data)
    assert np.array_equal(result, expected), f"{allocator} (stream=0): expected {expected}, got {result}"

    # Perform in-place all_reduce
    nccl_comm.all_reduce(send_data, send_data, nccl.SUM)
    _sync(allocator)
    result = _to_numpy(send_data)
    assert np.array_equal(result, expected), f"{allocator} (in-place): expected {expected}, got {result}"


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
    assert np.array_equal(result, expected), f"{allocator}: expected {expected}, got {result}"

    # Perform broadcast with explicit stream
    nccl_comm.broadcast(send_data, recv_data, root=root, stream=0)
    _sync(allocator)
    result = _to_numpy(recv_data)
    assert np.array_equal(result, expected), f"{allocator} (stream=0): expected {expected}, got {result}"

    # Perform in-place broadcast
    nccl_comm.broadcast(send_data, send_data, root=root)
    _sync(allocator)
    result = _to_numpy(send_data)
    assert np.array_equal(result, expected), f"{allocator} (in-place): expected {expected}, got {result}"

    # Perform broadcast with invalid sendbuf for non-root ranks
    if rank_info.nccl_rank == root:
        nccl_comm.broadcast(send_data, recv_data, root=root)
    else:
        nccl_comm.broadcast(None, recv_data, root=root)
    _sync(allocator)
    result = _to_numpy(recv_data)
    assert np.array_equal(result, expected), f"{allocator} (None sendbuf): expected {expected}, got {result}"


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
        assert np.array_equal(result, expected), f"{allocator}: expected {expected}, got {result}"

    # Perform reduce with explicit stream
    nccl_comm.reduce(send_data, recv_data, nccl.SUM, root=root, stream=0)
    _sync(allocator)
    if rank_info.nccl_rank == root:
        result = _to_numpy(recv_data)
        assert np.array_equal(result, expected), f"{allocator} (stream=0): expected {expected}, got {result}"

    # Perform reduce with invalid recvbuf for non-root ranks
    if rank_info.nccl_rank == root:
        nccl_comm.reduce(send_data, recv_data, nccl.SUM, root=root)
    else:
        nccl_comm.reduce(send_data, None, nccl.SUM, root=root)
    _sync(allocator)
    if rank_info.nccl_rank == root:
        result = _to_numpy(recv_data)
        assert np.array_equal(result, expected), f"{allocator} (None recvbuf): expected {expected}, got {result}"

    # Perform in-place reduce
    nccl_comm.reduce(send_data, send_data, nccl.SUM, root=root)
    _sync(allocator)
    if rank_info.nccl_rank == root:
        result = _to_numpy(send_data)
        assert np.array_equal(result, expected), f"{allocator} (in-place): expected {expected}, got {result}"


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("allocator", ["cupy", "torch", "interop.cupy", "interop.torch"])
def test_all_gather(nccl_comm, rank_info, allocator):
    """Test all_gather with different allocators."""
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

    # Perform all_gather
    nccl_comm.all_gather(send_data, recv_data)
    _sync(allocator)
    result = _to_numpy(recv_data)
    assert np.array_equal(result, expected), f"{allocator}: expected {expected}, got {result}"

    # Perform all_gather with explicit stream
    nccl_comm.all_gather(send_data, recv_data, stream=0)
    _sync(allocator)
    result = _to_numpy(recv_data)
    assert np.array_equal(result, expected), f"{allocator} (stream=0): expected {expected}, got {result}"

    # Perform in-place all_gather: sendbuf == recvbuf + rank * sendcount
    in_place_send_data = recv_data[rank_info.nccl_rank * count : (rank_info.nccl_rank + 1) * count]
    in_place_send_data[:] = send_data
    nccl_comm.all_gather(in_place_send_data, recv_data)
    _sync(allocator)
    result = _to_numpy(recv_data)
    assert np.array_equal(result, expected), f"{allocator} (in-place): expected {expected}, got {result}"


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
    assert np.array_equal(result, expected), f"{allocator}: expected {expected}, got {result}"

    # Perform reduce_scatter with explicit stream
    nccl_comm.reduce_scatter(send_data, recv_data, nccl.SUM, stream=0)
    _sync(allocator)
    result = _to_numpy(recv_data)
    assert np.array_equal(result, expected), f"{allocator} (stream=0): expected {expected}, got {result}"

    # Perform in-place reduce_scatter: recvbuf == sendbuf + rank * recvcount
    in_place_recv = send_data[rank_info.nccl_rank * count : (rank_info.nccl_rank + 1) * count]
    nccl_comm.reduce_scatter(send_data, in_place_recv, nccl.SUM)
    _sync(allocator)
    result = _to_numpy(in_place_recv)
    assert np.array_equal(result, expected), f"{allocator} (in-place): expected {expected}, got {result}"


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("allocator", ["cupy", "torch", "interop.cupy", "interop.torch"])
def test_all_to_all(nccl_comm, rank_info, allocator):
    """Test all_to_all with different allocators."""
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

    # Perform all_to_all
    nccl_comm.all_to_all(send_data, recv_data)
    _sync(allocator)
    result = _to_numpy(recv_data)
    assert np.array_equal(result, expected), f"{allocator}: expected {expected}, got {result}"

    # Perform all_to_all with explicit stream
    nccl_comm.all_to_all(send_data, recv_data, stream=0)
    _sync(allocator)
    result = _to_numpy(recv_data)
    assert np.array_equal(result, expected), f"{allocator} (stream=0): expected {expected}, got {result}"


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
        assert np.array_equal(result, expected), f"{allocator}: expected {expected}, got {result}"

    # Perform gather with explicit stream
    nccl_comm.gather(send_data, recv_data, root=root, stream=0)
    _sync(allocator)
    if rank_info.nccl_rank == root:
        result = _to_numpy(recv_data)
        assert np.array_equal(result, expected), f"{allocator} (stream=0): expected {expected}, got {result}"

    # Perform gather with invalid recvbuf for non-root ranks
    if rank_info.nccl_rank == root:
        nccl_comm.gather(send_data, recv_data, root=root)
    else:
        nccl_comm.gather(send_data, None, root=root)
    _sync(allocator)
    if rank_info.nccl_rank == root:
        result = _to_numpy(recv_data)
        assert np.array_equal(result, expected), f"{allocator} (None recvbuf): expected {expected}, got {result}"

    # Perform in-place gather: sendbuf == recvbuf + rank * sendcount
    # Need fresh recv_data for in-place test
    in_place_recv_data = _allocate_empty_buffer(count * rank_info.nccl_size, "float32", allocator)
    in_place_send_data = in_place_recv_data[rank_info.nccl_rank * count : (rank_info.nccl_rank + 1) * count]
    in_place_send_data[:] = send_data
    nccl_comm.gather(in_place_send_data, in_place_recv_data, root=root)
    _sync(allocator)
    if rank_info.nccl_rank == root:
        result = _to_numpy(in_place_recv_data)
        assert np.array_equal(result, expected), f"{allocator} (in-place): expected {expected}, got {result}"


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
    assert np.array_equal(result, expected), f"{allocator}: expected {expected}, got {result}"

    # Perform scatter with explicit stream
    nccl_comm.scatter(send_data, recv_data, root=root, stream=0)
    _sync(allocator)
    result = _to_numpy(recv_data)
    assert np.array_equal(result, expected), f"{allocator} (stream=0): expected {expected}, got {result}"

    # Perform scatter with invalid sendbuf for non-root ranks
    if rank_info.nccl_rank == root:
        nccl_comm.scatter(send_data, recv_data, root=root)
    else:
        nccl_comm.scatter(None, recv_data, root=root)
    _sync(allocator)
    result = _to_numpy(recv_data)
    assert np.array_equal(result, expected), f"{allocator} (None sendbuf): expected {expected}, got {result}"

    # Perform in-place scatter: recvbuf == sendbuf + rank * recvcount
    in_place_recv_data = send_data[rank_info.nccl_rank * count : (rank_info.nccl_rank + 1) * count]
    nccl_comm.scatter(send_data, in_place_recv_data, root=root)
    _sync(allocator)
    result = _to_numpy(in_place_recv_data)
    assert np.array_equal(result, expected), f"{allocator} (in-place): expected {expected}, got {result}"


# --- Buffer specification and slicing tests ---

@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("allocator", ["cupy", "torch", "interop.cupy", "interop.torch"])
def test_sliced_buffer(nccl_comm, rank_info, allocator):
    """Test sliced buffers and dtype override with all_reduce."""

    # Create base array with rank-specific data
    base_data = [rank_info.nccl_rank * 100 + i for i in range(20)]
    full_array = _allocate_buffer(base_data, "float32", allocator)

    # Test 1: full_array - Use full array, count = 20
    recv_buf = _allocate_empty_buffer(20, "float32", allocator)
    expected = np.array([sum(r * 100 + i for r in range(rank_info.nccl_size)) for i in range(20)], dtype=np.float32)
    nccl_comm.all_reduce(full_array, recv_buf, nccl.SUM)
    _sync(allocator)
    result = _to_numpy(recv_buf)
    assert np.array_equal(result, expected), f"{allocator} (full_array): expected {expected}, got {result}"

    # Test 2: slice_from_zero - Slice [0:10], count = 10
    send_slice = full_array[0:10]
    recv_buf = _allocate_empty_buffer(10, "float32", allocator)
    expected = np.array([sum(r * 100 + i for r in range(rank_info.nccl_size)) for i in range(10)], dtype=np.float32)
    nccl_comm.all_reduce(send_slice, recv_buf, nccl.SUM)
    _sync(allocator)
    result = _to_numpy(recv_buf)
    assert np.array_equal(result, expected), f"{allocator} (slice_from_zero): expected {expected}, got {result}"

    # Test 3: slice_from_middle - Slice [5:15], count = 10
    send_slice = full_array[5:15]
    recv_buf = _allocate_empty_buffer(10, "float32", allocator)
    expected = np.array([sum(r * 100 + (i + 5) for r in range(rank_info.nccl_size)) for i in range(10)], dtype=np.float32)
    nccl_comm.all_reduce(send_slice, recv_buf, nccl.SUM)
    _sync(allocator)
    result = _to_numpy(recv_buf)
    assert np.array_equal(result, expected), f"{allocator} (slice_from_middle): expected {expected}, got {result}"
