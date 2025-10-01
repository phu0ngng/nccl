import numpy as _np
from packaging.version import Version as _Version

from .. import bindings as _nccl_bindings
from .._version import version


class Version:
    def __init__(self, nccl_version: int) -> None:
        v = nccl_version
        if v >= 10000:
            major = v // 10000
            minor = (v % 10000) // 100
            patch = v % 100
        else:
            major = v // 1000
            minor = (v % 1000) // 100
            patch = v % 100

        self.nccl_version = _Version(f"{major}.{minor}.{patch}")
        self.nccl4py_version = _Version(version)

    def __repr__(self) -> str:
        return f"""
Versions:
    NCCL4Py version: {self.nccl4py_version}
    NCCL Library version: {self.nccl_version}
"""


def get_version() -> Version:
    v = int(_nccl_bindings.get_version())
    return Version(v)


class UniqueId:
    def __init__(self) -> None:
        self._internal = _nccl_bindings.UniqueId()

    @staticmethod
    def from_bytes(b: bytes) -> "UniqueId":
        try:
            buf = b if isinstance(b, (bytes, bytearray)) else b.tobytes()
        except Exception as e:
            raise TypeError("'b' must be a bytes-like object") from e

        expected = int(_nccl_bindings.unique_id_dtype.itemsize)
        if len(buf) != expected:
            raise ValueError(f"unique id must be {expected} bytes, got {len(buf)}")

        arr = _np.frombuffer(buf, dtype=_nccl_bindings.unique_id_dtype, count=1)

        uid = UniqueId.__new__(UniqueId)
        uid._internal = _nccl_bindings.UniqueId.from_data(arr)
        return uid

    @property
    def ptr(self) -> _nccl_bindings.UniqueId.ptr:
        return self._internal.ptr

    @property
    def as_ndarray(self) -> _np.ndarray:
        return self._internal._data

    @property
    def as_bytes(self) -> bytes:
        return self._internal._data.tobytes()


def get_unique_id(empty: bool = False) -> UniqueId:
    uid = UniqueId()
    if empty:
        return uid
    _nccl_bindings.get_unique_id(uid.ptr)
    return uid
