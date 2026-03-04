"""Unit tests for Communicator method argument conversions.

Tests cover:
- Communicator initialize() guard and cache reset
- Communicator grow() three calling patterns (new rank, existing root, existing non-root)
- NcclScalarSpec type conversion (int, float, np.ndarray, NcclSupportedBuffer)
- NcclBufferSpec handling in register_buffer/register_window
- Argument validation and error cases
"""
import numpy as np
import pytest

from nccl.core.communicator import Communicator
from nccl.core.typing import FLOAT32, FLOAT64, INT64
from nccl.core.constants import WindowFlag
from nccl.core.utils import UniqueId
from .mock import CAIBuf, DLPackBuf, View, FakeDevice
from nccl.core.typing import NcclInvalid


# --- initialize() Tests ---


def test_initialize_rejects_already_initialized():
    """initialize() raises NcclInvalid if communicator is already initialized."""
    comm = Communicator(0xC)
    uid = UniqueId.__new__(UniqueId)
    uid._internal = type("FakeUID", (), {"ptr": 0})()
    with pytest.raises(NcclInvalid, match="already initialized"):
        comm.initialize(nranks=2, rank=0, unique_id=uid)


def test_initialize_resets_cached_properties(monkeypatch):
    """initialize() clears cached nranks/rank/device/properties."""
    class B:
        @staticmethod
        def comm_init_rank_scalable(nranks, rank, n_id, comm_ids, config):
            return 0xABC

        unique_id_dtype = np.dtype([("internal", np.uint8, 128)])

    monkeypatch.setattr("nccl.core.communicator._nccl_bindings", B)

    comm = Communicator()
    # Manually set cached values to verify they get reset
    comm._resources = ["stale"]
    comm._nranks = 99
    comm._rank = 99
    comm._device = "stale"
    comm._comm_properties = "stale"

    uid = UniqueId.__new__(UniqueId)
    uid._internal = type("FakeUID", (), {"ptr": 0x123})()
    comm.initialize(nranks=2, rank=0, unique_id=uid)

    assert comm._resources == []
    assert comm._nranks is None
    assert comm._rank is None
    assert comm._device is None
    assert comm._comm_properties is None


# --- grow() Tests ---


def test_grow_rejects_new_rank_with_valid_comm():
    """grow() rejects new rank (rank != None) on an initialized communicator."""
    comm = Communicator(0xC)
    uid = UniqueId.__new__(UniqueId)
    uid._internal = type("FakeUID", (), {"ptr": 0x555})()
    with pytest.raises(NcclInvalid, match="New ranks must use a null communicator"):
        comm.grow(nranks=4, unique_id=uid, rank=3)


def test_grow_rejects_existing_rank_with_null_comm():
    """grow() rejects existing rank (rank=None) on a null communicator."""
    comm = Communicator()
    with pytest.raises(NcclInvalid, match="Existing ranks must use an initialized communicator"):
        comm.grow(nranks=4)


def test_grow_new_rank(monkeypatch):
    """New ranks: comm=NULL, uniqueId=&id, rank=assigned."""
    calls = {"grow": None}

    class B:
        @staticmethod
        def comm_grow(comm_ptr, nranks, uid_ptr, rank, config):
            calls["grow"] = (comm_ptr, nranks, uid_ptr, rank, config)
            return 0xFED

    monkeypatch.setattr("nccl.core.communicator._nccl_bindings", B)

    uid = UniqueId.__new__(UniqueId)
    uid._internal = type("FakeUID", (), {"ptr": 0x555})()

    comm = Communicator()
    new_comm = comm.grow(nranks=4, unique_id=uid, rank=3)
    assert new_comm.ptr == 0xFED
    assert calls["grow"] == (0, 4, 0x555, 3, 0)


def test_grow_existing_non_root(monkeypatch):
    """Existing non-root: comm=existing, uniqueId=NULL, rank=None → -1."""
    calls = {"grow": None}

    class B:
        @staticmethod
        def comm_grow(comm_ptr, nranks, uid_ptr, rank, config):
            calls["grow"] = (comm_ptr, nranks, uid_ptr, rank, config)
            return 0xABC

    monkeypatch.setattr("nccl.core.communicator._nccl_bindings", B)

    comm = Communicator.__new__(Communicator)
    comm._comm = 0xC
    comm._resources = []
    comm._nranks = None
    comm._device = None
    comm._rank = None
    comm._comm_properties = None

    # rank=None (default) should be converted to -1 for the C API
    new_comm = comm.grow(nranks=4)
    assert calls["grow"] == (0xC, 4, 0, -1, 0)


def test_grow_existing_root(monkeypatch):
    """Existing root: comm=existing, uniqueId=&id, rank=None → -1."""
    calls = {"grow": None}

    class B:
        @staticmethod
        def comm_grow(comm_ptr, nranks, uid_ptr, rank, config):
            calls["grow"] = (comm_ptr, nranks, uid_ptr, rank, config)
            return 0xABC

    monkeypatch.setattr("nccl.core.communicator._nccl_bindings", B)

    uid = UniqueId.__new__(UniqueId)
    uid._internal = type("FakeUID", (), {"ptr": 0x555})()

    comm = Communicator.__new__(Communicator)
    comm._comm = 0xC
    comm._resources = []
    comm._nranks = None
    comm._device = None
    comm._rank = None
    comm._comm_properties = None

    # rank=None (default) should be converted to -1 for the C API
    new_comm = comm.grow(nranks=4, unique_id=uid)
    assert calls["grow"] == (0xC, 4, 0x555, -1, 0)


def _setup_comm_with_mocked_bindings(monkeypatch, calls):
    """Setup communicator with mocked bindings and CustomRedOp."""
    class B:
        class ScalarResidence:
            HostImmediate = 0
            Device = 1
        @staticmethod
        def red_op_create_pre_mul_sum(scalar_ptr, dtype, residence, comm_ptr):
            calls["create"] = (scalar_ptr, dtype, residence, comm_ptr)
            return 0xDD
        @staticmethod
        def red_op_destroy(op, comm_ptr):
            pass

    monkeypatch.setattr("nccl.core.communicator._nccl_bindings", B)
    monkeypatch.setattr("nccl.core.resources._nccl_bindings", B)

    comm = Communicator.__new__(Communicator)
    comm._comm = 0xC
    comm._resources = []
    return comm, B


def test_scalar_spec_accepts_python_int(monkeypatch):
    """Test NcclScalarSpec with Python int (converted to int64 by NumPy)."""
    calls = {"create": None}
    comm, B = _setup_comm_with_mocked_bindings(monkeypatch, calls)

    # Python int infers INT64 (NumPy's natural dtype for int)
    op = comm.create_pre_mul_sum(42)
    assert calls["create"][1] == INT64.value
    assert calls["create"][2] == B.ScalarResidence.HostImmediate
    op.close()


def test_scalar_spec_accepts_python_float(monkeypatch):
    """Test NcclScalarSpec with Python float (converted to float64 by NumPy)."""
    calls = {"create": None}
    comm, B = _setup_comm_with_mocked_bindings(monkeypatch, calls)

    # Python float infers FLOAT64
    op = comm.create_pre_mul_sum(0.5)
    assert calls["create"][1] == FLOAT64.value
    assert calls["create"][2] == B.ScalarResidence.HostImmediate
    op.close()


def test_scalar_spec_accepts_numpy_array_host(monkeypatch):
    """Test NcclScalarSpec with NumPy array (host memory)."""
    calls = {"create": None}
    comm, B = _setup_comm_with_mocked_bindings(monkeypatch, calls)

    # NumPy array with explicit dtype
    scalar = np.array([0.25], dtype=np.float32)
    op = comm.create_pre_mul_sum(scalar)
    assert calls["create"][1] == FLOAT32.value
    assert calls["create"][2] == B.ScalarResidence.HostImmediate
    op.close()


def test_scalar_spec_accepts_device_buffer_dlpack(monkeypatch):
    """Test NcclScalarSpec with DLPack device buffer."""
    calls = {"create": None}
    comm, B = _setup_comm_with_mocked_bindings(monkeypatch, calls)
    monkeypatch.setattr(
        "nccl.core.buffer._resolve_buffer",
        lambda buf, s: View(
            ptr=0xBEEF, shape=(1,), dtype=np.dtype("float32"),
            device_id=0, strides=None, is_device_accessible=True,
            readonly=False, exporting_obj=buf
        )
    )

    # Device buffer with DLPack
    device_buf = DLPackBuf()
    op = comm.create_pre_mul_sum(device_buf)
    assert calls["create"][0] == 0xBEEF
    assert calls["create"][1] == FLOAT32.value
    assert calls["create"][2] == B.ScalarResidence.Device
    op.close()


def test_scalar_spec_accepts_device_buffer_cai(monkeypatch):
    """Test NcclScalarSpec with CUDA Array Interface device buffer."""
    calls = {"create": None}
    comm, B = _setup_comm_with_mocked_bindings(monkeypatch, calls)
    monkeypatch.setattr(
        "nccl.core.buffer._resolve_buffer",
        lambda buf, s: View(
            ptr=0xCAFE, shape=(1,), dtype=np.dtype("float32"),
            device_id=0, strides=None, is_device_accessible=True,
            readonly=False, exporting_obj=buf
        )
    )

    # Device buffer with CAI
    device_buf = CAIBuf()
    op = comm.create_pre_mul_sum(device_buf)
    assert calls["create"][0] == 0xCAFE
    assert calls["create"][1] == FLOAT32.value
    assert calls["create"][2] == B.ScalarResidence.Device
    op.close()


def test_scalar_spec_validates_single_element(monkeypatch):
    """Test NcclScalarSpec rejects multi-element arrays/buffers."""
    calls = {"create": None}
    comm, B = _setup_comm_with_mocked_bindings(monkeypatch, calls)

    # NumPy array with more than 1 element
    multi_elem = np.array([0.5, 0.6], dtype=np.float32)
    with pytest.raises(NcclInvalid, match="exactly 1 element"):
        comm.create_pre_mul_sum(multi_elem)

    # Device buffer with count > 1
    monkeypatch.setattr(
        "nccl.core.buffer._resolve_buffer",
        lambda buf, s: View(
            ptr=0xBEEF, shape=(2, 3), dtype=np.dtype("float32"),
            device_id=0, strides=None, is_device_accessible=True,
            readonly=False, exporting_obj=buf
        )
    )
    with pytest.raises(NcclInvalid, match="exactly 1 element"):
        comm.create_pre_mul_sum(CAIBuf())


def test_scalar_spec_explicit_datatype_override(monkeypatch):
    """Test NcclScalarSpec with explicit datatype parameter."""
    calls = {"create": None}
    comm, B = _setup_comm_with_mocked_bindings(monkeypatch, calls)

    # Python float with explicit FLOAT32 override
    op = comm.create_pre_mul_sum(0.5, datatype=FLOAT32)
    assert calls["create"][1] == FLOAT32.value
    op.close()


# --- Register Buffer Tests ---

def test_register_buffer_accepts_ncclbufferspec(monkeypatch):
    """Test register_buffer accepts NcclBufferSpec and derives size from buffer."""
    calls = {"reg": None}

    class B:
        @staticmethod
        def comm_register(comm_ptr, buf_ptr, size):
            calls["reg"] = (comm_ptr, buf_ptr, size)
            return 0xAA
        @staticmethod
        def comm_deregister(comm_ptr, handle):
            pass

    monkeypatch.setattr("nccl.core.communicator._nccl_bindings", B)
    monkeypatch.setattr("nccl.core.resources._nccl_bindings", B)
    monkeypatch.setattr(
        "nccl.core.buffer._resolve_buffer",
        lambda buf, s: View(
            ptr=0x1000, shape=(10,), dtype=np.dtype("float32"),
            device_id=0, strides=None, is_device_accessible=True,
            readonly=False, exporting_obj=buf
        )
    )

    comm = Communicator.__new__(Communicator)
    comm._comm = 0xC
    comm._device = FakeDevice(0)  # Mock device
    comm._rank = 0
    comm._nranks = 2
    comm._resources = []

    # Size derived from buffer (10 elements * 4 bytes = 40 bytes)
    handle = comm.register_buffer(CAIBuf())
    assert calls["reg"] == (0xC, 0x1000, 40)


def test_register_window_accepts_ncclbufferspec_and_flags(monkeypatch):
    """Test register_window accepts NcclBufferSpec with flags."""
    calls = {"reg": None}

    class B:
        @staticmethod
        def comm_window_register(comm_ptr, buf_ptr, size, flags):
            calls["reg"] = (comm_ptr, buf_ptr, size, flags)
            return 0xBB
        @staticmethod
        def comm_window_deregister(comm_ptr, handle):
            pass

    monkeypatch.setattr("nccl.core.communicator._nccl_bindings", B)
    monkeypatch.setattr("nccl.core.resources._nccl_bindings", B)
    monkeypatch.setattr(
        "nccl.core.buffer._resolve_buffer",
        lambda buf, s: View(
            ptr=0x2000, shape=(5,), dtype=np.dtype("float64"),
            device_id=0, strides=None, is_device_accessible=True,
            readonly=False, exporting_obj=buf
        )
    )

    comm = Communicator.__new__(Communicator)
    comm._comm = 0xC
    comm._device = FakeDevice(0)  # Mock device
    comm._rank = 0
    comm._nranks = 2
    comm._resources = []

    # Auto-detect size, default flags
    win = comm.register_window(CAIBuf())
    assert calls["reg"] == (0xC, 0x2000, 40, 0)  # 5 * 8 bytes = 40

    # With CollSymmetric flag
    calls["reg"] = None
    win2 = comm.register_window(DLPackBuf(), flags=WindowFlag.CollSymmetric)
    assert calls["reg"] == (0xC, 0x2000, 40, int(WindowFlag.CollSymmetric))


def test_register_window_returns_none_on_null_handle(monkeypatch):
    """register_window returns None and skips resource tracking on NULL handle."""

    class B:
        @staticmethod
        def comm_window_register(comm_ptr, buf_ptr, size, flags):
            return 0  # NULL handle
        @staticmethod
        def comm_window_deregister(comm_ptr, handle):
            pass

    monkeypatch.setattr("nccl.core.communicator._nccl_bindings", B)
    monkeypatch.setattr("nccl.core.resources._nccl_bindings", B)
    monkeypatch.setattr(
        "nccl.core.buffer._resolve_buffer",
        lambda buf, s: View(
            ptr=0x2000, shape=(5,), dtype=np.dtype("float64"),
            device_id=0, strides=None, is_device_accessible=True,
            readonly=False, exporting_obj=buf
        )
    )

    comm = Communicator.__new__(Communicator)
    comm._comm = 0xC
    comm._device = FakeDevice(0)
    comm._rank = 0
    comm._nranks = 2
    comm._resources = []

    win = comm.register_window(CAIBuf())
    assert win is None
    assert comm._resources == []


def test_buffer_device_validation(monkeypatch):
    """Test that buffers on wrong device raise NcclInvalid."""
    # Create communicator on device 0
    comm = Communicator.__new__(Communicator)
    comm._comm = 0xC
    comm._device = FakeDevice(0)  # Communicator on device 0
    comm._rank = 0
    comm._nranks = 2
    comm._resources = []

    # Mock _resolve_buffer to return buffer on device 1 (wrong device)
    monkeypatch.setattr(
        "nccl.core.buffer._resolve_buffer",
        lambda buf, s: View(
            ptr=0x1000, shape=(10,), dtype=np.dtype("float32"),
            device_id=1,  # Buffer on device 1, but comm is on device 0
            strides=None, is_device_accessible=True,
            readonly=False, exporting_obj=buf
        )
    )

    # Test that device mismatch raises NcclInvalid with clear message
    with pytest.raises(NcclInvalid, match="buffer is on device 1.*communicator is on device 0"):
        comm.register_buffer(CAIBuf())

    with pytest.raises(NcclInvalid, match="buffer is on device 1.*communicator is on device 0"):
        comm.register_window(CAIBuf())

    # Mock correct device (device 0) - should not raise
    monkeypatch.setattr(
        "nccl.core.buffer._resolve_buffer",
        lambda buf, s: View(
            ptr=0x1000, shape=(10,), dtype=np.dtype("float32"),
            device_id=0,  # Correct device
            strides=None, is_device_accessible=True,
            readonly=False, exporting_obj=buf
        )
    )

    # Mock the actual registration call to avoid errors
    class B:
        @staticmethod
        def comm_register(comm_ptr, buf_ptr, size):
            return 0xAA
        @staticmethod
        def comm_deregister(comm_ptr, handle):
            pass

    monkeypatch.setattr("nccl.core.communicator._nccl_bindings", B)
    monkeypatch.setattr("nccl.core.resources._nccl_bindings", B)

    # Should not raise with correct device
    handle = comm.register_buffer(CAIBuf())
    assert handle is not None


# --- init_all Argument Parsing Tests ---

def test_init_all_empty_list():
    """Test that init_all([]) returns empty list without calling into NCCL."""
    comms = Communicator.init_all([])
    assert isinstance(comms, list)
    assert len(comms) == 0


def test_init_all_zero_devices():
    """Test that init_all(0) returns empty list (range(0) is empty)."""
    comms = Communicator.init_all(0)
    assert isinstance(comms, list)
    assert len(comms) == 0


def test_init_all_negative_int():
    """Test that init_all(-1) returns empty list (range(-1) is empty)."""
    comms = Communicator.init_all(-1)
    assert comms == []


def test_init_all_rejects_invalid_type():
    """Test that init_all raises TypeError for non-int/sequence/None."""
    with pytest.raises(TypeError, match="devices must be an integer, sequence"):
        Communicator.init_all(1.5)

    with pytest.raises(TypeError, match="devices must be an integer, sequence"):
        Communicator.init_all("invalid")


# --- Struct Layout Validation Tests ---

def test_nccl_config_struct_layout():
    """Verify ncclConfig_t dtype matches C struct layout with manually calculated offsets."""
    from nccl.bindings import nccl as _nccl_bindings

    dtype = _nccl_bindings.config_dtype
    ptr_size = np.dtype(np.intp).itemsize
    assert ptr_size == 8, f"Tests assume 8-byte pointers, got {ptr_size}"

    # Manually calculated offsets and sizes based on C struct layout rules
    # struct ncclConfig_t {
    #   size_t size;              // uint64: offset 0, size 8
    #   uint32_t magic;           // uint32: offset 8, size 4
    #   uint32_t version;         // uint32: offset 12, size 4
    #   ncclConfig_v22900_t;      // int32 fields follow
    #   ...
    # }
    expected_layout = [
        # field_name                  type        offset    size
        ('size_',                     np.uint64,  0,        8),
        ('magic',                     np.uint32,  8,        4),
        ('version',                   np.uint32,  12,       4),
        ('blocking',                  np.int32,   16,       4),
        ('cga_cluster_size',          np.int32,   20,       4),
        ('min_ctas',                  np.int32,   24,       4),
        ('max_ctas',                  np.int32,   28,       4),
        ('net_name',                  np.intp,    32,       8),  # char* pointer
        ('split_share',               np.int32,   40,       4),
        ('traffic_class',             np.int32,   44,       4),
        ('comm_name',                 np.intp,    48,       8),  # char* pointer
        ('collnet_enable',            np.int32,   56,       4),
        ('cta_policy',                np.int32,   60,       4),
        ('shrink_share',              np.int32,   64,       4),
        ('nvls_ctas',                 np.int32,   68,       4),
        ('n_channels_per_net_peer',   np.int32,   72,       4),
        ('nvlink_centric_sched',      np.int32,   76,       4),
        ('graph_usage_mode',          np.int32,   80,       4),
        ('num_rma_ctx',               np.int32,   84,       4),
    ]
    expected_total_size = 88

    # Verify struct size
    assert dtype.itemsize == expected_total_size, \
        f"Struct size mismatch: {dtype.itemsize} != {expected_total_size}"

    # Verify all expected fields exist
    assert set(dtype.names) == set(name for name, _, _, _ in expected_layout), \
        f"Field name mismatch"

    # Verify field types, offsets, and sizes
    for field_name, expected_type, expected_offset, expected_size in expected_layout:
        actual_type = dtype.fields[field_name][0]
        actual_offset = dtype.fields[field_name][1]
        actual_size = actual_type.itemsize

        assert actual_type == np.dtype(expected_type), \
            f"Field '{field_name}' type mismatch: {actual_type} != {np.dtype(expected_type)}"
        assert actual_offset == expected_offset, \
            f"Field '{field_name}' offset mismatch: {actual_offset} != {expected_offset}"
        assert actual_size == expected_size, \
            f"Field '{field_name}' size mismatch: {actual_size} != {expected_size}"


def test_nccl_dev_resource_requirements_struct_layout():
    """Verify ncclDevResourceRequirements_t dtype matches C struct layout with manually calculated offsets."""
    from nccl.bindings import nccl as _nccl_bindings

    dtype = _nccl_bindings.dev_resource_requirements_dtype
    ptr_size = np.dtype(np.intp).itemsize
    assert ptr_size == 8, f"Tests assume 8-byte pointers, got {ptr_size}"

    # Manually calculated offsets and sizes based on C struct layout rules
    # struct ncclDevResourceRequirements_t {
    #   void *next;                           // ptr: offset 0, size 8
    #   uint64_t bufferSize;                  // uint64: offset 8, size 8
    #   uint64_t bufferAlign;                 // uint64: offset 16, size 8
    #   ncclDevResourceHandle_t *outBufferHandle; // ptr: offset 24, size 8
    #   int32_t ginSignalCount;               // int32: offset 32, size 4
    #   int32_t ginCounterCount;              // int32: offset 36, size 4
    #   ncclGinSignal_t *outGinSignalStart;   // ptr: offset 40, size 8
    #   ncclGinCounter_t *outGinCounterStart; // ptr: offset 48, size 8
    # }
    expected_layout = [
        # field_name               type        offset    size
        ('next',                   np.intp,    0,        8),
        ('buffer_size',            np.uint64,  8,        8),
        ('buffer_align',           np.uint64,  16,       8),
        ('out_buffer_handle',      np.intp,    24,       8),
        ('gin_signal_count',       np.int32,   32,       4),
        ('gin_counter_count',      np.int32,   36,       4),
        ('out_gin_signal_start',   np.intp,    40,       8),
        ('out_gin_counter_start',  np.intp,    48,       8),
    ]
    expected_total_size = 56

    # Verify struct size
    assert dtype.itemsize == expected_total_size, \
        f"Struct size mismatch: {dtype.itemsize} != {expected_total_size}"

    # Verify all expected fields exist
    assert set(dtype.names) == set(name for name, _, _, _ in expected_layout), \
        f"Field name mismatch"

    # Verify field types, offsets, and sizes
    for field_name, expected_type, expected_offset, expected_size in expected_layout:
        actual_type = dtype.fields[field_name][0]
        actual_offset = dtype.fields[field_name][1]
        actual_size = actual_type.itemsize

        assert actual_type == np.dtype(expected_type), \
            f"Field '{field_name}' type mismatch: {actual_type} != {np.dtype(expected_type)}"
        assert actual_offset == expected_offset, \
            f"Field '{field_name}' offset mismatch: {actual_offset} != {expected_offset}"
        assert actual_size == expected_size, \
            f"Field '{field_name}' size mismatch: {actual_size} != {expected_size}"


def test_nccl_comm_properties_struct_layout():
    """Verify ncclCommProperties_t dtype matches C struct layout with manually calculated offsets."""
    from nccl.bindings import nccl as _nccl_bindings

    dtype = _nccl_bindings.comm_properties_dtype

    # Manually calculated offsets and sizes based on C struct layout rules
    # struct ncclCommProperties_t {
    #   size_t size;                // uint64: offset 0, size 8
    #   uint32_t magic;             // uint32: offset 8, size 4
    #   uint32_t version;           // uint32: offset 12, size 4
    #   int rank;                   // int32: offset 16, size 4
    #   int nRanks;                 // int32: offset 20, size 4
    #   int cudaDev;                // int32: offset 24, size 4
    #   int nvmlDev;                // int32: offset 28, size 4
    #   bool deviceApiSupport;      // uint8: offset 32, size 1
    #   bool multimemSupport;       // uint8: offset 33, size 1
    #   // [padding 2 bytes: offset 34-35]
    #   ncclGinType_t ginType;      // int32 (enum): offset 36, size 4
    #   int nLsaTeams;              // int32: offset 40, size 4
    #   bool hostRmaSupport;        // uint8: offset 44, size 1
    #   // [padding 3 bytes: offset 45-47]
    #   ncclGinType_t railedGinType; // int32 (enum): offset 48, size 4
    #   // [padding 4 bytes to align struct to 8-byte boundary]
    # }
    expected_layout = [
        # field_name             type        offset    size
        ('size_',                np.uint64,  0,        8),
        ('magic',                np.uint32,  8,        4),
        ('version',              np.uint32,  12,       4),
        ('rank',                 np.int32,   16,       4),
        ('n_ranks',              np.int32,   20,       4),
        ('cuda_dev',             np.int32,   24,       4),
        ('nvml_dev',             np.int32,   28,       4),
        ('device_api_support',   np.uint8,   32,       1),
        ('multimem_support',     np.uint8,   33,       1),
        # 2 bytes padding here (offset 34-35)
        ('gin_type',             np.int32,   36,       4),
        ('n_lsa_teams',          np.int32,   40,       4),
        ('host_rma_support',     np.uint8,   44,       1),
        # 3 bytes padding here (offset 45-47)
        ('railed_gin_type',      np.int32,   48,       4),
        # 4 bytes padding to align struct to 8-byte boundary
    ]
    expected_total_size = 56

    # Verify struct size
    assert dtype.itemsize == expected_total_size, \
        f"Struct size mismatch: {dtype.itemsize} != {expected_total_size}"

    # Verify all expected fields exist
    assert set(dtype.names) == set(name for name, _, _, _ in expected_layout), \
        f"Field name mismatch"

    # Verify field types, offsets, and sizes
    for field_name, expected_type, expected_offset, expected_size in expected_layout:
        actual_type = dtype.fields[field_name][0]
        actual_offset = dtype.fields[field_name][1]
        actual_size = actual_type.itemsize

        assert actual_type == np.dtype(expected_type), \
            f"Field '{field_name}' type mismatch: {actual_type} != {np.dtype(expected_type)}"
        assert actual_offset == expected_offset, \
            f"Field '{field_name}' offset mismatch: {actual_offset} != {expected_offset}"
        assert actual_size == expected_size, \
            f"Field '{field_name}' size mismatch: {actual_size} != {expected_size}"


def test_nccl_team_struct_layout():
    """Verify ncclTeam_t dtype matches C struct layout with manually calculated offsets."""
    from nccl.bindings import nccl as _nccl_bindings

    dtype = _nccl_bindings.team_dtype

    # Manually calculated offsets and sizes based on C struct layout rules
    # struct ncclTeam {
    #   int nRanks;   // int32: offset 0, size 4
    #   int rank;     // int32: offset 4, size 4
    #   int stride;   // int32: offset 8, size 4
    # }
    expected_layout = [
        # field_name    type       offset    size
        ('n_ranks',    np.int32,  0,        4),
        ('rank',       np.int32,  4,        4),
        ('stride',     np.int32,  8,        4),
    ]
    expected_total_size = 12

    # Verify struct size
    assert dtype.itemsize == expected_total_size, \
        f"Struct size mismatch: {dtype.itemsize} != {expected_total_size}"

    # Verify all expected fields exist
    assert set(dtype.names) == set(name for name, _, _, _ in expected_layout), \
        f"Field name mismatch"

    # Verify field types, offsets, and sizes
    for field_name, expected_type, expected_offset, expected_size in expected_layout:
        actual_type = dtype.fields[field_name][0]
        actual_offset = dtype.fields[field_name][1]
        actual_size = actual_type.itemsize

        assert actual_type == np.dtype(expected_type), \
            f"Field '{field_name}' type mismatch: {actual_type} != {np.dtype(expected_type)}"
        assert actual_offset == expected_offset, \
            f"Field '{field_name}' offset mismatch: {actual_offset} != {expected_offset}"
        assert actual_size == expected_size, \
            f"Field '{field_name}' size mismatch: {actual_size} != {expected_size}"


def test_nccl_dev_comm_requirements_struct_layout():
    """Verify ncclDevCommRequirements_t dtype matches C struct layout with manually calculated offsets."""
    from nccl.bindings import nccl as _nccl_bindings

    dtype = _nccl_bindings.dev_comm_requirements_dtype
    ptr_size = np.dtype(np.intp).itemsize
    assert ptr_size == 8, f"Tests assume 8-byte pointers, got {ptr_size}"

    # Manually calculated offsets and sizes based on C struct layout rules
    # struct ncclDevCommRequirements_t {
    #   size_t size;                                  // uint64: offset 0, size 8
    #   unsigned int magic;                           // uint32: offset 8, size 4
    #   unsigned int version;                         // uint32: offset 12, size 4
    #   ncclDevResourceRequirements_t* resourceRequirementsList;  // ptr: offset 16, size 8
    #   ncclTeamRequirements_t* teamRequirementsList; // ptr: offset 24, size 8
    #   bool lsaMultimem;                             // uint8: offset 32, size 1
    #   // [padding 3 bytes: offset 33-35]
    #   int barrierCount;                             // int32: offset 36, size 4
    #   int lsaBarrierCount;                          // int32: offset 40, size 4
    #   int railGinBarrierCount;                      // int32: offset 44, size 4
    #   int lsaLLA2ABlockCount;                       // int32: offset 48, size 4
    #   int lsaLLA2ASlotCount;                        // int32: offset 52, size 4
    #   bool ginForceEnable;                          // uint8: offset 56, size 1
    #   // [padding 3 bytes: offset 57-59]
    #   int ginContextCount;                          // int32: offset 60, size 4
    #   int ginSignalCount;                           // int32: offset 64, size 4
    #   int ginCounterCount;                          // int32: offset 68, size 4
    #   ncclGinConnectionType_t ginConnectionType;    // int32: offset 72, size 4
    #   bool ginExclusiveContexts;                    // uint8: offset 76, size 1
    #   // [padding 3 bytes: offset 77-79]
    #   int ginQueueDepth;                            // int32: offset 80, size 4
    #   // [padding 4 bytes to align struct to 8-byte boundary]
    # }
    expected_layout = [
        # field_name                     type        offset    size
        ('size_',                        np.uint64,  0,        8),
        ('magic',                        np.uint32,  8,        4),
        ('version',                      np.uint32,  12,       4),
        ('resource_requirements_list',   np.intp,    16,       8),
        ('team_requirements_list',       np.intp,    24,       8),
        ('lsa_multimem',                 np.uint8,   32,       1),
        # 3 bytes padding
        ('barrier_count',                np.int32,   36,       4),
        ('lsa_barrier_count',            np.int32,   40,       4),
        ('rail_gin_barrier_count',       np.int32,   44,       4),
        ('lsa_ll_a2a_block_count',       np.int32,   48,       4),
        ('lsa_ll_a2a_slot_count',        np.int32,   52,       4),
        ('gin_force_enable',             np.uint8,   56,       1),
        # 3 bytes padding
        ('gin_context_count',            np.int32,   60,       4),
        ('gin_signal_count',             np.int32,   64,       4),
        ('gin_counter_count',            np.int32,   68,       4),
        ('gin_connection_type',          np.int32,   72,       4),
        ('gin_exclusive_contexts',       np.uint8,   76,       1),
        # 3 bytes padding
        ('gin_queue_depth',              np.int32,   80,       4),
        # 4 bytes padding to align struct to 8-byte boundary
    ]
    expected_total_size = 88

    # Verify struct size
    assert dtype.itemsize == expected_total_size, \
        f"Struct size mismatch: {dtype.itemsize} != {expected_total_size}"

    # Verify all expected fields exist
    assert set(dtype.names) == set(name for name, _, _, _ in expected_layout), \
        f"Field name mismatch"

    # Verify field types, offsets, and sizes
    for field_name, expected_type, expected_offset, expected_size in expected_layout:
        actual_type = dtype.fields[field_name][0]
        actual_offset = dtype.fields[field_name][1]
        actual_size = actual_type.itemsize

        assert actual_type == np.dtype(expected_type), \
            f"Field '{field_name}' type mismatch: {actual_type} != {np.dtype(expected_type)}"
        assert actual_offset == expected_offset, \
            f"Field '{field_name}' offset mismatch: {actual_offset} != {expected_offset}"
        assert actual_size == expected_size, \
            f"Field '{field_name}' size mismatch: {actual_size} != {expected_size}"


def test_nccl_team_requirements_struct_layout():
    """Verify ncclTeamRequirements_t dtype matches C struct layout with manually calculated offsets."""
    from nccl.bindings import nccl as _nccl_bindings

    dtype = _nccl_bindings.team_requirements_dtype
    ptr_size = np.dtype(np.intp).itemsize
    assert ptr_size == 8, f"Tests assume 8-byte pointers, got {ptr_size}"

    # Manually calculated offsets and sizes based on C struct layout rules
    # struct ncclTeamRequirements_t {
    #   ncclTeamRequirements_t* next;         // ptr: offset 0, size 8
    #   ncclTeam_t team;                      // struct: offset 8, size 12 (3 int32)
    #   bool multimem;                        // uint8: offset 20, size 1
    #   // [padding 3 bytes: offset 21-23]
    #   ncclMultimemHandle_t* outMultimemHandle; // ptr: offset 24, size 8
    # }
    expected_layout = [
        # field_name              type                          offset    size
        ('next',                  np.intp,                      0,        8),
        ('team',                  _nccl_bindings.team_dtype,    8,        12),  # nested struct, 12 bytes
        ('multimem',              np.uint8,                     20,       1),
        # 3 bytes padding (offset 21-23)
        ('out_multimem_handle',   np.intp,                      24,       8),
    ]
    expected_total_size = 32

    # Verify struct size
    assert dtype.itemsize == expected_total_size, \
        f"Struct size mismatch: {dtype.itemsize} != {expected_total_size}"

    # Verify all expected fields exist
    assert set(dtype.names) == set(name for name, _, _, _ in expected_layout), \
        f"Field name mismatch"

    # Verify field types, offsets, and sizes
    for field_name, expected_type, expected_offset, expected_size in expected_layout:
        actual_type = dtype.fields[field_name][0]
        actual_offset = dtype.fields[field_name][1]
        actual_size = actual_type.itemsize

        # For nested struct, just check the size matches
        if field_name == 'team':
            assert actual_size == expected_size, \
                f"Field '{field_name}' size mismatch: {actual_size} != {expected_size}"
        else:
            assert actual_type == np.dtype(expected_type), \
                f"Field '{field_name}' type mismatch: {actual_type} != {np.dtype(expected_type)}"

        assert actual_offset == expected_offset, \
            f"Field '{field_name}' offset mismatch: {actual_offset} != {expected_offset}"
        assert actual_size == expected_size, \
            f"Field '{field_name}' size mismatch: {actual_size} != {expected_size}"



def test_nccl_dev_comm_struct_layout():
    """Verify ncclDevComm_t dtype matches device struct layout with manually calculated offsets."""
    from nccl.bindings import nccl as _nccl_bindings

    dtype = _nccl_bindings.dev_comm_dtype
    ptr_size = np.dtype(np.intp).itemsize
    assert ptr_size == 8, f"Tests assume 8-byte pointers, got {ptr_size}"

    # Manually calculated offsets and sizes based on C struct layout rules
    # This is a device-side struct from nccl_device/impl/comm__types.h
    # Note: Some fields are nested structs, we validate the scalar fields and their offsets
    expected_layout = [
        # field_name                type        offset    size
        ('rank',                    np.int32,   0,        4),
        ('n_ranks',                 np.int32,   4,        4),
        ('n_ranks_rcp32',           np.uint32,  8,        4),
        ('lsa_rank',                np.int32,   12,       4),
        ('lsa_size',                np.int32,   16,       4),
        ('lsa_size_rcp32',          np.uint32,  20,       4),
        ('window_table',            np.intp,    24,       8),  # ncclDevCommWindowTable_t pointer
        ('resource_window',         np.intp,    32,       8),  # ncclWindow_t pointer
        # resource_window_inlined at offset 40, size 72 (nested struct)
        # lsa_multimem at offset 112, size 8 (nested struct)
        # lsa_barrier at offset 120, size 8 (nested struct)
        # rail_gin_barrier at offset 128, size 8 (nested struct)
        ('gin_connection_count',    np.uint8,   136,      1),
        # gin_net_device_types at offset 137, size 4 (array)
        # padding 3 bytes (offset 141-143)
        # gin_handles at offset 144, size 32 (pointer array)
        ('gin_signal_base',         np.uint32,  176,      4),
        ('gin_signal_count',        np.int32,   180,      4),
        ('gin_counter_base',        np.uint32,  184,      4),
        ('gin_counter_count',       np.int32,   188,      4),
        ('gin_signal_shadows',      np.intp,    192,      8),  # uint64_t* pointer
        ('gin_context_count',       np.uint32,  200,      4),
        ('gin_context_base',        np.uint32,  204,      4),
        ('gin_is_railed',           np.uint8,   208,      1),
        # padding 7 bytes (offset 209-215)
        ('abort_flag',              np.intp,    216,      8),  # uint32_t* pointer
    ]
    expected_total_size = 224

    # Verify struct size
    assert dtype.itemsize == expected_total_size, \
        f"Struct size mismatch: {dtype.itemsize} != {expected_total_size}"

    # Verify all expected scalar fields exist (nested structs and arrays are separate)
    scalar_field_names = {name for name, _, _, _ in expected_layout}
    assert scalar_field_names.issubset(set(dtype.names)), \
        f"Missing fields: {scalar_field_names - set(dtype.names)}"

    # Verify field types, offsets, and sizes for scalar fields
    for field_name, expected_type, expected_offset, expected_size in expected_layout:
        actual_type = dtype.fields[field_name][0]
        actual_offset = dtype.fields[field_name][1]
        actual_size = actual_type.itemsize

        assert actual_type == np.dtype(expected_type), \
            f"Field '{field_name}' type mismatch: {actual_type} != {np.dtype(expected_type)}"
        assert actual_offset == expected_offset, \
            f"Field '{field_name}' offset mismatch: {actual_offset} != {expected_offset}"
        assert actual_size == expected_size, \
            f"Field '{field_name}' size mismatch: {actual_size} != {expected_size}"
