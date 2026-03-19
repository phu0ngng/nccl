"""
Free-threading stress tests for nccl4py.

These tests exercise concurrent access to nccl4py's Python API using
pytest-run-parallel. Only APIs documented as thread-safe are tested here.
"""

import threading
import unittest.mock

import pytest

import nccl.core.memory as _mem_mod
from nccl.core.memory import NcclMemoryResource, get_memory_resource

from .mock import FakeDevice, FakeBuf


# Module-level patches which are applied once before any test runs so that the
# individual tests do not use the monkeypatch fixture which is thread-unsafe.


class _MockBinds:
    _alloc_lock = threading.Lock()
    _counter = 0

    @classmethod
    def mem_alloc(cls, size):
        with cls._alloc_lock:
            cls._counter += 1
            return cls._counter

    @staticmethod
    def mem_free(ptr):
        pass


_FakeBuffer = type("_B", (), {"from_handle": staticmethod(lambda ptr, size, mr: FakeBuf(ptr))})
_FakeCudaCtx = lambda d: type("Ctx", (), {"__enter__": lambda s: s, "__exit__": lambda s, *a: None})()


@pytest.fixture(scope="module", autouse=True)
def _patch_memory_deps_module():
    """Patch cuda.core dependencies once for the entire module."""
    with (
        unittest.mock.patch.object(_mem_mod, "_nccl_bindings", _MockBinds),
        unittest.mock.patch.object(_mem_mod, "Buffer", _FakeBuffer),
        unittest.mock.patch.object(_mem_mod, "Device", FakeDevice),
        unittest.mock.patch.object(_mem_mod, "CudaDeviceContext", _FakeCudaCtx),
    ):
        yield


# Memory resource tests

@pytest.mark.parallel_threads_limit(16)
def test_get_memory_resource_concurrent_same_device():
    """
    Concurrent calls to get_memory_resource for the same device must always
    return the same thread-local instance — no duplicates, no crashes.
    """
    mr_a = get_memory_resource(0)
    mr_b = get_memory_resource(0)
    assert mr_a is mr_b


@pytest.mark.parallel_threads_limit(16)
def test_get_memory_resource_concurrent_different_devices():
    """
    Concurrent calls across different device IDs must not interfere with each
    other's thread-local caches.
    """
    mr0 = get_memory_resource(0)
    mr1 = get_memory_resource(1)
    assert mr0 is not mr1
    assert mr0.device_id == 0
    assert mr1.device_id == 1


@pytest.mark.parallel_threads_limit(8)
def test_memory_resource_allocate_concurrent():
    """
    Concurrent allocations on the same NcclMemoryResource must not crash or
    return overlapping pointers.
    """
    mr = NcclMemoryResource(0)
    buf = mr.allocate(64)
    assert buf.ptr != 0
