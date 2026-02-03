import os
import numpy as np
from mpi4py import MPI
import pytest

from cuda.core.experimental import Device

import nccl.bindings as nccl_bindings
import nccl.core as nccl
from conftest import requires_nccl_version

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


@pytest.mark.mpi
def test_init_without_config(uid_shared, rank_info):
    device = Device(rank_info.nccl_local_rank)
    device.set_current()

    comm = nccl.Communicator.init(nranks=rank_info.nccl_size, rank=rank_info.nccl_rank, unique_id=uid_shared)
    assert comm.nranks == rank_info.nccl_size
    assert comm.rank == rank_info.nccl_rank
    assert comm.device.device_id == rank_info.nccl_local_rank
    assert isinstance(comm.device, Device)
    comm.destroy()


@pytest.mark.mpi
def test_init_with_config(uid_shared, rank_info, get_nccl_debug_file):
    device = Device(rank_info.nccl_local_rank)
    device.set_current()

    cfg = nccl.NCCLConfig(
        blocking=True,
        min_ctas=2,
        max_ctas=4,
        cta_policy=nccl.CTAPolicy.Efficiency,
        split_share=False,
    )
    comm = nccl.Communicator.init(nranks=rank_info.nccl_size, rank=rank_info.nccl_rank, unique_id=uid_shared, config=cfg)
    assert comm.nranks == rank_info.nccl_size
    comm.destroy()

    nccl_debug_file = get_nccl_debug_file()
    assert nccl_debug_file is not None
    assert os.path.exists(nccl_debug_file)
    assert os.path.getsize(nccl_debug_file) > 0
    with open(nccl_debug_file) as f:
        debug_log = f.read()
    assert "Comm config Blocking set to 1" in debug_log
    assert "Comm config Min CTAs set to 2" in debug_log
    assert "Comm config Max CTAs set to 4" in debug_log
    assert "Comm config Split share set to 0" in debug_log


@pytest.mark.mpi
@pytest.mark.parametrize(
    "invalid_config, expected_exception, expected_status",
    [
        (nccl.NCCLConfig(net_name="NONEXIST"), nccl_bindings.NCCLError, nccl_bindings.Result.InvalidUsage),
    ]
)
def test_init_with_invalid_config(uid_shared, rank_info, invalid_config, expected_exception, expected_status):
    device = Device(rank_info.nccl_local_rank)
    device.set_current()

    # Ensure env does not override user net selection for net_name test
    os.environ.pop("NCCL_NET", None)
    comm = None

    # No group: error surfaces immediately; no comm to destroy
    with pytest.raises(expected_exception) as e:
        comm = nccl.Communicator.init(nranks=rank_info.nccl_size, rank=rank_info.nccl_rank, unique_id=uid_shared, config=invalid_config)
    assert e.value.status == expected_status
    assert comm is None

    # With group: handle is created, error at group end
    with pytest.raises(expected_exception) as e:
        with nccl.group():
            comm = nccl.Communicator.init(nranks=rank_info.nccl_size, rank=rank_info.nccl_rank, unique_id=uid_shared, config=invalid_config)
    assert e.value.status == expected_status
    assert comm is not None
    comm.destroy()


@requires_nccl_version("2.18.1")
@pytest.mark.mpi
@pytest.mark.parametrize(
    "color, key, expect_invalid, expected_nranks_fn",
    [
        # All in one group (color=0)
        (0, 0, False, lambda rank, size: size),
        # Split into two groups (even/odd)
        (lambda rank, size: int(rank % 2), lambda rank, size: rank, False, lambda rank, size: (size + 1 - (rank % 2)) // 2),
        # Same color, different keys
        (1, lambda rank, size: int(size - rank - 1), False, lambda rank, size: size),
        # NCCL_SPLIT_NOCOLOR should return None
        (nccl.NCCL_SPLIT_NOCOLOR, 0, True, lambda rank, size: 0),
    ]
)
def test_split(uid_shared, rank_info, color, key, expect_invalid, expected_nranks_fn):
    device = Device(rank_info.nccl_local_rank)
    device.set_current()

    base = nccl.Communicator.init(
        nranks=rank_info.nccl_size,
        rank=rank_info.nccl_rank,
        unique_id=uid_shared
    )

    # Compute color and key for this rank
    color_val = color(rank_info.nccl_rank, rank_info.nccl_size) if callable(color) else color
    key_val = key(rank_info.nccl_rank, rank_info.nccl_size) if callable(key) else key

    # Ensure color_val and key_val are int or None
    if color_val is not None and not isinstance(color_val, int):
        color_val = int(color_val)
    if key_val is not None and not isinstance(key_val, int):
        key_val = int(key_val)

    sub = base.split(color=color_val, key=key_val)
    if expect_invalid:
        assert not sub.is_valid
    else:
        assert isinstance(sub, nccl.Communicator)
        expected_nranks = expected_nranks_fn(rank_info.nccl_rank, rank_info.nccl_size)
        assert sub.nranks == expected_nranks
        assert 0 <= sub.rank < sub.nranks
        sub.destroy()

    base.destroy()


@requires_nccl_version("2.18.1")
@pytest.mark.mpi(min_size=4)
def test_split_with_value_validation(nccl_comm, rank_info):
    """Test split with value validation."""
    if not HAS_CUPY:
        pytest.skip("CuPy not installed, skip test_split_with_value_validation")

    send_data = nccl.cupy.empty(1, dtype="float32")
    recv_data = nccl.cupy.empty(1, dtype="float32")

    expected = np.zeros_like(recv_data.get())
    send_data[0] = nccl_comm.rank
    expected[0] = 0 + (rank_info.nccl_size - 1) * rank_info.nccl_size / 2

    nccl_comm.reduce(send_data, recv_data, nccl.SUM)
    cp.cuda.Stream.null.synchronize()
    result = recv_data.get()
    assert np.allclose(result, expected)

    # Split into two groups
    color = rank_info.nccl_rank % 2
    key = rank_info.nccl_rank
    sub = nccl_comm.split(color=color, key=key)
    assert isinstance(sub, nccl.Communicator)
    assert sub.is_valid
    if rank_info.nccl_rank % 2 == 0:
        first_rank = 0
        last_rank = rank_info.nccl_size - 2 if rank_info.nccl_size % 2 == 0 else rank_info.nccl_size - 1
        expected[0] = (first_rank + last_rank) * sub.nranks / 2
    else:
        first_rank = 1
        last_rank = rank_info.nccl_size - 1 if rank_info.nccl_size % 2 == 0 else rank_info.nccl_size - 2
        expected[0] = (first_rank + last_rank) * sub.nranks / 2

    sub.reduce(send_data, recv_data, nccl.SUM)
    cp.cuda.Stream.null.synchronize()
    result = recv_data.get()
    assert np.allclose(result, expected), f"rank {rank_info.nccl_rank}: expected {expected}, got {result}"

    sub.destroy()


@requires_nccl_version("2.27.3")
@pytest.mark.mpi(min_size=4)
@pytest.mark.parametrize(
    "exclude_ranks, config, flag, expect_error",
    [
        ([], None, nccl.CommShrinkFlag.Default, True),
        ([0], None, nccl.CommShrinkFlag.Default, False),
        ([1], None, nccl.CommShrinkFlag.Default, False),
        ([0,1,3], None, nccl.CommShrinkFlag.Default, False),
        ([], None, nccl.CommShrinkFlag.Abort, True),
        ([0], None, nccl.CommShrinkFlag.Abort, False),
        ([1], None, nccl.CommShrinkFlag.Abort, False),
        ([0,1,3], None, nccl.CommShrinkFlag.Abort, False),
    ]
)
def test_shrink(uid_shared, rank_info, exclude_ranks, config, flag, expect_error):
    # Set current device for this rank
    device = Device(rank_info.nccl_local_rank)
    device.set_current()

    base = nccl.Communicator.init(nranks=rank_info.nccl_size, rank=rank_info.nccl_rank, unique_id=uid_shared)

    if not rank_info.nccl_rank in exclude_ranks:
        if expect_error:
            with pytest.raises(nccl_bindings.NCCLError):
                shrunk = base.shrink(exclude_ranks=exclude_ranks, config=config, flag=flag)
        else:
            shrunk = base.shrink(exclude_ranks=exclude_ranks, config=config, flag=flag)
            assert isinstance(shrunk, nccl.Communicator)
            assert shrunk.nranks == rank_info.nccl_size - len(exclude_ranks)
            assert 0 <= shrunk.rank < shrunk.nranks
            shrunk.destroy()

    base.destroy()


@requires_nccl_version("2.27.3")
@pytest.mark.mpi(min_size=4)
def test_shrink_with_value_validation(nccl_comm, rank_info):
    """Test split with value validation."""
    if not HAS_CUPY:
        pytest.skip("CuPy not installed, skip test_split_with_value_validation")

    send_data = nccl.cupy.empty(1, dtype="float32")
    recv_data = nccl.cupy.empty(1, dtype="float32")

    expected = np.zeros_like(recv_data.get())
    send_data[0] = nccl_comm.rank
    expected[0] = 0 + (rank_info.nccl_size - 1) * rank_info.nccl_size / 2

    nccl_comm.reduce(send_data, recv_data, nccl.SUM)
    cp.cuda.Stream.null.synchronize()
    result = recv_data.get()
    assert np.allclose(result, expected)

    # Shrink group
    exclude_ranks = [0, 1]
    if not rank_info.nccl_rank in exclude_ranks:
        sub = nccl_comm.shrink(exclude_ranks=exclude_ranks)
        assert isinstance(sub, nccl.Communicator)
        assert sub.is_valid
        expected[0] = expected[0] - sum(exclude_ranks)

        sub.reduce(send_data, recv_data, nccl.SUM)
        cp.cuda.Stream.null.synchronize()
        result = recv_data.get()
        assert np.allclose(result, expected), f"rank {rank_info.nccl_rank}: expected {expected}, got {result}"

        sub.destroy()


@requires_nccl_version("2.23.4")
@pytest.mark.mpi(min_size=4)
def test_init_rank_scalable(rank_info):
    device = Device(rank_info.nccl_local_rank)
    device.set_current()

    comm = MPI.COMM_WORLD
    nranks = rank_info.nccl_size
    rank = rank_info.nccl_rank

    # Choose a small number of ids; match all ranks on the same value
    n_ids = 2 if nranks >= 2 else 1

    # Python port of rankHasRoot from test/perf/common.cu
    def rank_has_root(r: int, n_ranks: int, n_roots: int) -> bool:
        rmr = n_ranks % n_roots
        rpr = n_ranks // n_roots
        rlim = rmr * (rpr + 1)
        if r < rlim:
            return (r % (rpr + 1)) == 0
        else:
            return ((r - rlim) % rpr) == 0

    # Each selected root contributes one id
    uid = None
    if rank_has_root(rank, nranks, n_ids):
        uid = nccl.get_unique_id()

    if uid is not None:
        local_bytes = uid.as_bytes
    else:
        local_bytes = b""

    sendcount = len(local_bytes)
    counts = comm.allgather(sendcount)
    displs = [0]
    for c in counts[:-1]:
        displs.append(displs[-1] + c)
    total = sum(counts)
    recvbuf = bytearray(total)
    comm.Allgatherv([local_bytes, MPI.BYTE], [recvbuf, counts, displs, MPI.BYTE])

    # Reconstruct the global list of UniqueIds
    gathered_ids: list[nccl.UniqueId] = []
    for i, count in enumerate(counts):
        if count > 0:
            start = displs[i]
            end = start + count
            gathered_ids.append(nccl.UniqueId.from_bytes(bytes(recvbuf[start:end])))

    # Exactly n_ids ids expected
    assert len(gathered_ids) == n_ids

    comm_nccl = nccl.Communicator.init(
        nranks=rank_info.nccl_size,
        rank=rank_info.nccl_rank,
        unique_id=gathered_ids,
    )
    assert isinstance(comm_nccl, nccl.Communicator)
    assert comm_nccl.nranks == rank_info.nccl_size
    assert comm_nccl.rank == rank_info.nccl_rank
    comm_nccl.destroy()


@pytest.mark.mpi
def test_abort_finalize_idempotency(uid_shared, rank_info):
    # Set current device for this rank
    device = Device(rank_info.nccl_local_rank)
    device.set_current()
    # Test that abort, finalize, and destroy are idempotent and do not raise errors on repeated calls
    comm = nccl.Communicator.init(nranks=rank_info.nccl_size, rank=rank_info.nccl_rank, unique_id=uid_shared)
    # First call sequence
    comm.destroy()
    # Repeated calls should not raise
    comm.abort()
    comm.finalize()
    comm.destroy()

@pytest.mark.mpi
def test_properties_after_destroy_raise(uid_shared, rank_info):
    # Set current device for this rank
    device = Device(rank_info.nccl_local_rank)
    device.set_current()
    # Test that accessing properties after destroy raises exception
    comm = nccl.Communicator.init(nranks=rank_info.nccl_size, rank=rank_info.nccl_rank, unique_id=uid_shared)
    comm.destroy()
    # List of properties that should raise after destroy
    properties = [
        lambda: comm.nranks,
        lambda: comm.device,
        lambda: comm.rank,
    ]
    for prop in properties:
        with pytest.raises((nccl.NcclInvalid, RuntimeError)):
            _ = prop()


@pytest.mark.mpi
def test_get_async_error(nccl_comm, rank_info):
    """Test get_async_error returns Success during normal operation."""
    result = nccl_comm.get_async_error()
    assert result == nccl_bindings.Result.Success


@pytest.mark.mpi
def test_finalize(uid_shared, rank_info):
    """Test finalize() works and comm remains valid after finalize."""
    device = Device(rank_info.nccl_local_rank)
    device.set_current()

    comm = nccl.Communicator.init(nranks=rank_info.nccl_size, rank=rank_info.nccl_rank, unique_id=uid_shared)
    comm.finalize()

    # After finalize, comm is still valid and properties accessible
    assert comm.is_valid
    assert comm.nranks == rank_info.nccl_size

    comm.destroy()


@requires_nccl_version("2.19.3")
@pytest.mark.mpi
def test_register_buffer_api(nccl_comm):
    """Test register_buffer with real NCCL."""
    buf = nccl.mem_alloc(1024)

    handle = nccl_comm.register_buffer(buf)
    assert handle.is_valid

    handle.close()
    assert not handle.is_valid


@requires_nccl_version("2.27.3")
@pytest.mark.mpi
def test_register_window_api(nccl_comm):
    """Test register_window with real NCCL."""

    if not HAS_CUPY:
        pytest.skip("CuPy not installed")

    buf = nccl.cupy.empty(256, dtype='float32')  # 256 * 4 bytes = 1024 bytes

    win = nccl_comm.register_window(buf)
    assert win is None or win.is_valid

    if win is not None:
        win.close()
        assert not win.is_valid


@pytest.mark.mpi(min_size=4)
@pytest.mark.parametrize("scalar_type", [
    "int", "float", "numpy.ndarray_int", "numpy.ndarray_float", "NcclSupportedBuffer_int", "NcclSupportedBuffer_float"
])
def test_custom_op(nccl_comm, rank_info, scalar_type):
    """Test split with value validation."""
    if not HAS_CUPY:
        pytest.skip("CuPy not installed, skip test_split_with_value_validation")

    def _assert_result_matches(actual, expected_array, *, msg=None):
        if actual.dtype.kind in {"f", "c"} or expected_array.dtype.kind in {"f", "c"}:
            assert np.allclose(actual, expected_array), msg
        else:
            assert np.array_equal(actual, expected_array), msg

    if scalar_type == "int":
        send_data = nccl.cupy.empty(1, dtype="int32")
        recv_data = nccl.cupy.empty(1, dtype="int32")
        scalar = 13
        scalar_for_expected = 13
        scalar_dtype = nccl.INT32
    elif scalar_type == "float":
        send_data = nccl.cupy.empty(1, dtype="float64")
        recv_data = nccl.cupy.empty(1, dtype="float64")
        scalar = 1.7
        scalar_for_expected = 1.7
        scalar_dtype = nccl.FLOAT64
    elif scalar_type == "numpy.ndarray_int":
        send_data = nccl.cupy.empty(1, dtype="int32")
        recv_data = nccl.cupy.empty(1, dtype="int32")
        scalar = np.array([13], dtype="int32")
        scalar_for_expected = 13
        scalar_dtype = nccl.INT32
    elif scalar_type == "numpy.ndarray_float":
        send_data = nccl.cupy.empty(1, dtype="float32")
        recv_data = nccl.cupy.empty(1, dtype="float32")
        scalar = np.array([1.7], dtype="float32")
        scalar_for_expected = 1.7
        scalar_dtype = nccl.FLOAT32
    elif scalar_type == "NcclSupportedBuffer_int":
        send_data = nccl.cupy.empty(1, dtype="int32")
        recv_data = nccl.cupy.empty(1, dtype="int32")
        scalar = nccl.cupy.empty(1, dtype="int32")
        scalar[0] = 13
        scalar_for_expected = 13
        scalar_dtype = nccl.INT32
    elif scalar_type == "NcclSupportedBuffer_float":
        send_data = nccl.cupy.empty(1, dtype="float32")
        recv_data = nccl.cupy.empty(1, dtype="float32")
        scalar = nccl.cupy.empty(1, dtype="float32")
        scalar[0] = 1.7
        scalar_for_expected = 1.7
        scalar_dtype = nccl.FLOAT32
    else:
        pytest.skip(f"Unknown scalar type: {scalar_type}")

    expected = np.zeros_like(recv_data.get())
    send_data[0] = nccl_comm.rank
    expected[0] = 0 + (rank_info.nccl_size - 1) * rank_info.nccl_size / 2

    nccl_comm.reduce(send_data, recv_data, nccl.SUM)
    cp.cuda.Stream.null.synchronize()
    result = recv_data.get()
    _assert_result_matches(result, expected)

    op = nccl_comm.create_pre_mul_sum(scalar, datatype=scalar_dtype)
    expected = expected * scalar_for_expected
    nccl_comm.reduce(send_data, recv_data, op)
    cp.cuda.Stream.null.synchronize()
    result = recv_data.get()
    _assert_result_matches(result, expected, msg=f"rank {rank_info.nccl_rank}, scalar_type {scalar_type}: expected {expected}, got {result}")
    op.close()
