"""Unit tests for CUDA device and stream helpers."""
import pytest

from nccl.core.cuda import get_cuda_device, get_cuda_stream, get_device_id, get_stream_ptr, CudaDeviceContext
from .mock import FakeStream, FakeDevice


def test_get_cuda_device_with_none_device_and_int(monkeypatch):
    """Test get_cuda_device accepts None, Device, and int."""
    monkeypatch.setattr("nccl.core.cuda.Device", FakeDevice)

    # None → current device (Device())
    dev = get_cuda_device(None)
    assert isinstance(dev, FakeDevice)
    assert dev.device_id == 0

    # Device instance (returns as-is)
    fake_dev = FakeDevice(3)
    result = get_cuda_device(fake_dev)  # type: ignore[arg-type]
    assert result is fake_dev

    # int → Device(int)
    dev = get_cuda_device(5)
    assert isinstance(dev, FakeDevice)
    assert dev.device_id == 5


def test_get_cuda_stream_with_various_inputs(monkeypatch):
    """Test get_cuda_stream with None, int, Stream, and protocol objects."""
    monkeypatch.setattr("nccl.core.cuda.Device", FakeDevice)
    monkeypatch.setattr("nccl.core.cuda.Stream", FakeStream)

    # Add default_stream to FakeDevice
    FakeDevice.default_stream = FakeStream(0)

    # None → device.default_stream
    stream = get_cuda_stream(None)
    assert isinstance(stream, FakeStream)

    # int → Stream.from_handle(handle)
    stream = get_cuda_stream(123)
    assert isinstance(stream, FakeStream)
    assert int(stream.handle) == 123

    # Stream instance → return as-is
    fake_stream = FakeStream(456)
    result = get_cuda_stream(fake_stream)  # type: ignore[arg-type]
    assert result is fake_stream


def test_cuda_device_context_switch_and_restore(monkeypatch):
    """Test CudaDeviceContext switches device on enter and restores on exit."""
    monkeypatch.setattr("nccl.core.cuda.Device", FakeDevice)

    ctx = CudaDeviceContext(FakeDevice(2))  # type: ignore[arg-type]
    with ctx:
        pass

    # Verify set_current was called
    assert ctx._device.set_calls == [2]  # type: ignore[attr-defined]
    assert ctx._old_device.set_calls == [0]  # type: ignore[attr-defined]


def test_get_device_id_with_various_inputs(monkeypatch):
    """Test get_device_id with int, Device, and None."""
    monkeypatch.setattr("nccl.core.cuda.Device", FakeDevice)

    # int → return as-is
    device_id = get_device_id(3)
    assert device_id == 3

    # Device instance → extract device_id
    fake_dev = FakeDevice(5)
    device_id = get_device_id(fake_dev)  # type: ignore[arg-type]
    assert device_id == 5

    # None → get current device_id
    device_id = get_device_id(None)
    assert isinstance(device_id, int)
    assert device_id == 0  # FakeDevice() returns device_id=0


def test_get_stream_ptr_with_various_inputs(monkeypatch):
    """Test get_stream_ptr with None, int, Stream, and protocol objects."""
    monkeypatch.setattr("nccl.core.cuda.Stream", FakeStream)

    # None → return 0 (default stream)
    stream_ptr = get_stream_ptr(None)
    assert stream_ptr == 0

    # int → return as-is
    stream_ptr = get_stream_ptr(123)
    assert stream_ptr == 123

    # Stream instance → extract handle
    fake_stream = FakeStream(456)
    stream_ptr = get_stream_ptr(fake_stream)  # type: ignore[arg-type]
    assert stream_ptr == 456

    # Protocol object with __cuda_stream__
    class StreamProtocol:
        def __cuda_stream__(self):
            return (0, 789)  # (device, stream_ptr)

    stream_obj = StreamProtocol()
    stream_ptr = get_stream_ptr(stream_obj)  # type: ignore[arg-type]
    assert stream_ptr == 789
