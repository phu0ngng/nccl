import ctypes
import numpy as np
import pytest

from nccl.core.utils import Version, get_version, UniqueId, get_unique_id


class _Binds:
    @staticmethod
    def get_version():
        return 2 * 10000 + 28 * 100 + 1


def test_version_decode_and_get_version(monkeypatch):
    v = Version(1 * 1000 + 5 * 100 + 0)
    assert str(v.nccl_version) == "1.5.0"
    monkeypatch.setattr("nccl.core.utils._nccl_bindings", _Binds)
    got = get_version()
    assert str(got.nccl_version) == "2.28.1"


@pytest.mark.parametrize(
    "enc,expected_str",
    [
        (1 * 1000 + 5 * 100 + 0, "1.5.0"),        # old scheme, < 2.8
        (2 * 1000 + 7 * 100 + 99, "2.7.99"),      # old scheme, < 2.8
        (2 * 1000 + 8 * 100 + 99, "2.8.99"),      # old scheme boundary
        (2 * 10000 + 9 * 100 + 0, "2.9.0"),       # new scheme boundary
        (3 * 10000 + 0 * 100 + 0, "3.0.0"),       # new scheme major bump
    ],
)
def test_version_decode_from_int(enc, expected_str):
    v = Version(enc)
    assert str(v.nccl_version) == expected_str


@pytest.mark.parametrize(
    "a_enc,b_enc,rel",
    [
        (1500, 2000, "<"),
        (2799, 2800, "<"),
        (2750, 2751, "<"),
        (2899, 20900, "<"),
        (20909, 30000, "<"),
        (30101, 30102, "<"),
        (30200, 30102, ">"),
        (30102, 30102, "=="),
    ],
)
def test_version_comparisons(a_enc, b_enc, rel):
    va = Version(a_enc).nccl_version
    vb = Version(b_enc).nccl_version
    if rel == "<":
        assert va < vb
        assert va <= vb
        assert va != vb
        assert not (va > vb)
        assert not (va >= vb)
    elif rel == ">":
        assert va > vb
        assert va >= vb
        assert va != vb
        assert not (va < vb)
        assert not (va <= vb)
    elif rel == "==":
        assert va == vb
        assert not (va != vb)
        assert va <= vb
        assert va >= vb


def test_eq_ne_with_non_version():
    a = Version(3 * 10000 + 1 * 100 + 2).nccl_version
    assert (a == "3.1.2") is False
    assert (a != "3.1.2") is True


class _UidBinds:
    class UniqueId:
        def __init__(self):
            # 128 bytes as simple uint8 array
            import numpy as _np
            self._data = _np.zeros((1,), dtype=[("internal", "u1", (128,))])
        @property
        def ptr(self):
            # Provide a valid address so core can safely use ptr-based accessors.
            return int(self._data.ctypes.data)
        @staticmethod
        def from_data(arr):
            u = _UidBinds.UniqueId()
            # copy data
            u._data[...] = arr[...]
            return u

    unique_id_dtype = np.dtype([("internal", "u1", (128,))])

    @staticmethod
    def get_unique_id(ptr):
        # Fill the pointed-to ncclUniqueId buffer with a deterministic non-zero pattern.
        # This simulates NCCL writing the unique ID into user-provided storage.
        pattern = bytes((0xA5 ^ i) & 0xFF for i in range(128))
        ctypes.memmove(int(ptr), pattern, len(pattern))


def test_unique_id_as_bytes(monkeypatch):
    """Test UniqueId bytes view for empty, filled, and from_bytes."""
    monkeypatch.setattr("nccl.core.utils._nccl_bindings", _UidBinds)

    uid_empty = get_unique_id(empty=True)
    assert len(uid_empty.as_bytes) == 128
    assert uid_empty.as_bytes == bytes(128)

    uid = get_unique_id()
    assert len(uid.as_bytes) == 128
    expected = bytes((0xA5 ^ i) & 0xFF for i in range(128))
    assert uid.as_bytes == expected

    uid1 = get_unique_id()
    uid2 = get_unique_id()
    assert uid1.as_bytes == uid2.as_bytes  # deterministic mock pattern

    rt = UniqueId.from_bytes(uid1.as_bytes)
    assert rt.as_bytes == uid1.as_bytes


def test_unique_id_as_ndarray(monkeypatch):
    """Test UniqueId ndarray view for empty, filled, and from_bytes."""
    monkeypatch.setattr("nccl.core.utils._nccl_bindings", _UidBinds)

    uid_empty = get_unique_id(empty=True)
    assert np.array_equal(uid_empty.as_ndarray["internal"][0], np.zeros(128, dtype=np.uint8))

    uid = get_unique_id()
    expected = np.frombuffer(bytes((0xA5 ^ i) & 0xFF for i in range(128)), dtype=np.uint8)
    assert np.array_equal(uid.as_ndarray["internal"][0], expected)

    uid1 = get_unique_id()
    uid2 = get_unique_id()
    assert np.array_equal(uid1.as_ndarray["internal"][0], uid2.as_ndarray["internal"][0])

    rt = UniqueId.from_bytes(uid1.as_bytes)
    assert np.array_equal(rt.as_ndarray["internal"][0], uid1.as_ndarray["internal"][0])


def test_from_bytes_rejects_wrong_length(monkeypatch):
    """Test UniqueId.from_bytes validates length."""
    monkeypatch.setattr("nccl.core.utils._nccl_bindings", _UidBinds)

    with pytest.raises(ValueError):
        UniqueId.from_bytes(bytes(127))
