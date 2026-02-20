import pytest

from nccl.core.resources import RegisteredBufferHandle, RegisteredWindowHandle, CustomRedOp
from nccl.core.constants import WindowFlag
from nccl.core.typing import FLOAT32
from nccl.bindings import ScalarResidence


def test_registered_buffer_handle_register_and_close(monkeypatch):
    calls = {"reg": [], "dereg": []}
    class B:
        @staticmethod
        def comm_register(comm_ptr, buf_ptr, size):
            calls["reg"].append((comm_ptr, buf_ptr, size))
            return 0xAA
        @staticmethod
        def comm_deregister(comm_ptr, handle):
            calls["dereg"].append((comm_ptr, handle))
    monkeypatch.setattr("nccl.core.resources._nccl_bindings", B)

    r = RegisteredBufferHandle(0xC, 0xB, 16)
    assert r.handle == 0xAA
    r.close()
    # Idempotent
    r.close()
    assert calls["reg"] == [(0xC, 0xB, 16)]
    assert calls["dereg"] == [(0xC, 0xAA)]
    with pytest.raises(RuntimeError):
        _ = r.handle


def test_registered_window_handle_flags_and_close(monkeypatch):
    calls = {"reg": [], "dereg": []}
    class B:
        @staticmethod
        def comm_window_register(comm_ptr, buf_ptr, size, flags):
            calls["reg"].append((comm_ptr, buf_ptr, size, flags))
            return 0xBB
        @staticmethod
        def comm_window_deregister(comm_ptr, handle):
            calls["dereg"].append((comm_ptr, handle))
    monkeypatch.setattr("nccl.core.resources._nccl_bindings", B)

    w = RegisteredWindowHandle(0xC, 0xB, 32, WindowFlag.CollSymmetric)
    assert w.handle == 0xBB
    w.close(); w.close()
    assert calls["reg"] == [(0xC, 0xB, 32, int(WindowFlag.CollSymmetric))]
    assert calls["dereg"] == [(0xC, 0xBB)]


def test_custom_redop_lifecycle(monkeypatch):
    calls = {"create": [], "destroy": []}
    class B:
        @staticmethod
        def red_op_create_pre_mul_sum(scalar_ptr, dtype, residence, comm_ptr):
            calls["create"].append((scalar_ptr, dtype, residence, comm_ptr))
            return 0xDD
        @staticmethod
        def red_op_destroy(op, comm_ptr):
            calls["destroy"].append((op, comm_ptr))
    monkeypatch.setattr("nccl.core.resources._nccl_bindings", B)

    # Use real NcclDataType and ScalarResidence
    r = CustomRedOp(0xC, 0x5CA1A, FLOAT32, ScalarResidence.HostImmediate)
    assert int(r) == 0xDD
    assert r.op == 0xDD
    r.close(); r.close()
    assert calls["create"] == [(0x5CA1A, FLOAT32.value, ScalarResidence.HostImmediate, 0xC)]
    assert calls["destroy"] == [(0xDD, 0xC)]
    with pytest.raises(RuntimeError):
        _ = r.op



