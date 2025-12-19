"""Unit tests for Communicator method argument conversions.

Tests cover:
- NcclScalarSpec type conversion (int, float, np.ndarray, NcclSupportedBuffer)
- NcclBufferSpec handling in register_buffer/register_window
- Argument validation and error cases
"""
import numpy as np
import pytest

from nccl.core.communicator import Communicator
from nccl.core.typing import FLOAT32, FLOAT64, INT64
from nccl.core.constants import WindowFlag
from .mock import CAIBuf, DLPackBuf, View, FakeDevice
from nccl.core.typing import NcclInvalid


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

