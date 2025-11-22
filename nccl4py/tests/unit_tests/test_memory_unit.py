import sys

from nccl.core.memory import NcclMemoryResource, get_memory_resource
from .mock import FakeDevice, FakeBuf, TrackedClose


def test_memory_resource_allocate_deallocate_and_free_all(monkeypatch):
    calls = {"alloc": [], "free": []}

    class _Binds:
        @staticmethod
        def mem_alloc(size):
            calls["alloc"].append(size)
            return 0xDEADBEEF
        @staticmethod
        def mem_free(ptr):
            calls["free"].append(ptr)

    monkeypatch.setattr(sys.modules["nccl.core.memory"], "_nccl_bindings", _Binds)
    monkeypatch.setattr("nccl.core.memory.Buffer", type("_B", (), {"from_handle": staticmethod(lambda ptr, size, mr: FakeBuf(ptr))}))
    # Mock Device to avoid GPU initialization
    monkeypatch.setattr("nccl.core.memory.Device", FakeDevice)
    # Mock CudaDeviceContext to avoid GPU initialization
    monkeypatch.setattr("nccl.core.memory.CudaDeviceContext", lambda d: type("Ctx", (), {"__enter__": lambda s: s, "__exit__": lambda s, *a: None})())

    # NcclMemoryResource now takes device_id (int) instead of Device
    mr = NcclMemoryResource(0)
    buf = mr.allocate(123)
    assert calls["alloc"] == [123]

    # Closing buffer triggers deallocate via Buffer.close; emulate by calling mr.deallocate
    mr.deallocate(buf.ptr, 123)
    assert calls["free"] == [0xDEADBEEF]


def test_get_memory_resource_singleton_per_device(monkeypatch):
    # Monkeypatch Device to avoid GPU
    monkeypatch.setattr("nccl.core.memory.Device", FakeDevice)
    # get_memory_resource now takes device_id (int) instead of Device
    mr0a = get_memory_resource(0)
    mr0b = get_memory_resource(0)
    assert mr0a is mr0b

    # Different device IDs should return different instances
    mr1 = get_memory_resource(1)
    assert mr1 is not mr0a



