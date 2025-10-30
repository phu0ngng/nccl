import numpy as np
import pytest

from nccl.core.buffer import NcclBuffer
from nccl.core.typing import FLOAT32, FLOAT64, INT8, INT32, INT64, NcclInvalid
from .mock import View, CAIBuf, DLPackBuf, FakeBuffer


def test_with_raw_buffer(monkeypatch):
    """Test NcclBufferSpec with raw Buffer from cuda.core.experimental."""
    seen = {"buf": None}
    def _cap(buf, s):
        seen["buf"] = buf
        return View(ptr=0xDEF, shape=(4, 2), dtype=np.dtype("float64"), device_id=0, strides=None, is_device_accessible=True, readonly=False, exporting_obj=buf)
    monkeypatch.setattr("nccl.core.buffer._resolve_buffer", _cap)

    # Monkeypatch Buffer type so isinstance checks in typing layer recognize FakeBuffer
    monkeypatch.setattr("nccl.core.typing.Buffer", FakeBuffer)

    buf = FakeBuffer(ptr=0x1234, size=64)
    b = NcclBuffer(buf, None)
    assert b.ptr == 0xDEF
    assert b.count == 8
    assert seen["buf"] is buf


def test_with_dlpack_protocol(monkeypatch):
    seen = {"buf": None}
    def _cap(buf, s):
        seen["buf"] = buf
        return View(ptr=0xABC, shape=(8, 4), dtype=np.dtype("float32"), device_id=0, strides=None, is_device_accessible=True, readonly=False, exporting_obj=buf)
    monkeypatch.setattr("nccl.core.buffer._resolve_buffer", _cap)
    dl = DLPackBuf()
    b = NcclBuffer(dl, None)
    assert b.ptr == 0xABC
    assert b.count == 32
    assert int(b.dtype.value) == int(FLOAT32.value)
    assert seen["buf"] is dl


def test_with_cuda_array_interface(monkeypatch):
    seen = {"buf": None}
    def _cap(buf, s):
        seen["buf"] = buf
        return View(ptr=0xABC, shape=(8, 4), dtype=np.dtype("float32"), device_id=0, strides=None, is_device_accessible=True, readonly=False, exporting_obj=buf)
    monkeypatch.setattr("nccl.core.buffer._resolve_buffer", _cap)

    cai = CAIBuf()
    b = NcclBuffer(cai, None)
    assert b.ptr == 0xABC
    assert b.count == 32
    assert int(b.dtype.value) == int(FLOAT32.value)
    assert seen["buf"] is cai
