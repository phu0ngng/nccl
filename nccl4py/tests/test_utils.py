import os
import re
from pathlib import Path
import pytest
import ctypes
import numpy as np
import nccl.core as core
from nccl import __version__


# ---------------- Version tests (edge cases and comparisons) ----------------
def test_get_version_values():
    v = core.get_version()
    # nccl library version is numeric x.y.z
    major, minor, patch = map(int, str(v.nccl_version).split("."))
    assert major >= 0 and minor >= 0 and patch >= 0
    # nccl4py version should be parseable as x.y.z
    m2, n2, p2 = map(int, str(v.nccl4py_version).split("."))
    assert m2 >= 0 and n2 >= 0 and p2 >= 0

    assert str(v.nccl4py_version) == __version__

    # Independently validate against the banner embedded in libnccl.so.
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    libs: list[Path] = []
    for d in filter(None, ld.split(":")):
        p = Path(d)
        if not p.is_dir():
            continue
        exact = p / "libnccl.so"
        if exact.exists():
            libs.append(exact)
        else:
            libs.extend(p.glob("libnccl.so.*"))

    if not libs:
        pytest.skip("libnccl.so not found in LD_LIBRARY_PATH")

    # Scan file contents via mmap to avoid loading the whole binary into memory.
    import mmap
    pat = re.compile(rb"NCCL version\s+(\d+\.\d+\.\d+).*\+cuda\d+\.\d+")
    expected = None
    for lib in libs:
        real = lib.resolve()
        try:
            with open(real, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                m = pat.search(mm)
                if m:
                    expected = m.group(1).decode("ascii")
                    break
        except Exception:
            continue

    if expected is None:
        pytest.skip("Embedded NCCL banner not found in libnccl.so; cannot independently verify version")

    assert str(v.nccl_version) == expected

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
    v = core.Version(enc)
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
    va = core.Version(a_enc).nccl_version
    vb = core.Version(b_enc).nccl_version
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
    a = core.Version(3 * 10000 + 1 * 100 + 2).nccl_version
    assert (a == "3.1.2") is False
    assert (a != "3.1.2") is True


# ---------------- UniqueId tests ----------------
def test_get_unique_id_empty_and_filled_bytes():
    uid_empty = core.get_unique_id(empty=True)
    assert len(uid_empty.as_bytes) == 128

    uid = core.get_unique_id()
    assert len(uid.as_bytes) == 128
    assert any(uid.as_bytes)


def test_unique_id_from_bytes_roundtrip_and_distinctness():
    uid1 = core.get_unique_id()
    uid2 = core.get_unique_id()
    assert uid1.as_bytes != uid2.as_bytes

    rt = core.UniqueId.from_bytes(uid1.as_bytes)
    assert rt.as_bytes == uid1.as_bytes


def test_ptr_memory_matches_as_bytes():
    uid = core.get_unique_id()
    raw = ctypes.string_at(uid.ptr, 128)
    assert raw == uid.as_bytes


def test_ndarray_mutation_reflects_in_bytes():
    uid = core.get_unique_id(empty=True)
    pattern = np.arange(128, dtype=np.uint8)
    arr = uid.as_ndarray
    arr["internal"][0][:] = pattern.view(np.int8)
    assert uid.as_bytes == bytes(pattern)


def test_from_bytes_rejects_wrong_length():
    wrong = bytes(127)
    try:
        core.UniqueId.from_bytes(wrong)
        assert False, "Expected ValueError for wrong length"
    except ValueError:
        pass
