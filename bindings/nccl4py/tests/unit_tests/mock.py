from typing import Any


import numpy as np


class View:
    def __init__(
        self,
        ptr: int,
        shape: tuple,
        dtype: np.dtype,
        device_id: int,
        strides: tuple | None,
        is_device_accessible: bool,
        readonly: bool,
        exporting_obj: object | None,
    ) -> None:
        # Aligns with cuda.core.utils.StridedMemoryView
        self.ptr = ptr
        self.shape = shape
        self.strides = strides
        self.dtype = dtype
        self.device_id = device_id
        self.is_device_accessible = is_device_accessible
        self.readonly = readonly
        self.exporting_obj = exporting_obj


class FakeBuffer:
    """Mock for cuda.core.Buffer with DLPack protocol.

    This makes FakeBuffer a valid NcclSupportedBuffer (implements SupportsDLPack).
    """
    def __init__(self, ptr: int, size: int):
        self.ptr = ptr
        self.size = size

    def close(self):
        pass

    def __dlpack__(self, /, *, stream=None):
        return object()

    def __dlpack_device__(self):
        return (1, 0)  # CUDA device 0



class CAIBuf:
    @property
    def __cuda_array_interface__(self):
        return {
            "shape": (8, 4),
            "typestr": "<f4",
            "data": (0xABC, False),
            "version": 3,
            "strides": None,
        }


class DLPackBuf:
    def __dlpack__(self, /, *, stream=None):
        return object()
    def __dlpack_device__(self):
        return (1, 0)


class CUstream:
    """Mock for cuda.bindings.driver.CUstream (wraps int stream handle)."""
    def __init__(self, value: int):
        self._value = value

    def __int__(self):
        return self._value


class FakeStream:
    """Mock for cuda.core.Stream.

    Satisfies both Stream interface (.handle returns CUstream) and IsStreamT protocol (__cuda_stream__).
    """
    def __init__(self, handle: int):
        self._handle_int = handle

    @classmethod
    def from_handle(cls, handle: int):
        return cls(handle)

    @property
    def handle(self):
        """Return CUstream object (matches cuda.core.Stream.handle)."""
        return CUstream(self._handle_int)


class StreamProtoGood:
    """Valid implementation of IsStreamT protocol."""
    def __init__(self, handle: int):
        self._h = handle
    def __cuda_stream__(self):
        return (0, self._h)


class StreamProtoBadTuple:
    """Invalid IsStreamT: __cuda_stream__ returns wrong type."""
    def __cuda_stream__(self):
        return 123


class StreamProtoNotCallable:
    """Invalid IsStreamT: __cuda_stream__ is not callable."""
    __cuda_stream__ = 5


class FakeDevice:
    """Mock for cuda.core.Device."""
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.set_calls = []
    def set_current(self):
        self.set_calls.append(self.device_id)


class FakeBuf:
    """Simple mock buffer for memory resource tests."""
    def __init__(self, ptr):
        self.ptr = ptr
    def close(self):
        pass


class TrackedClose:
    def __init__(self):
        self.closed = 0
    def close(self):
        self.closed += 1
