import os
import re
from pathlib import Path
import pytest
import ctypes
import numpy as np
import nccl.bindings as nccl_bindings
import nccl.core as core
from nccl import __version__


# ---------------- Version tests ----------------
@pytest.mark.mpi(min_size=1)
def test_get_version_values(uid_shared, rank_info, get_nccl_debug_file):
    """Test get_version and validate against NCCL debug log."""
    from cuda.core import Device

    device = Device(rank_info.nccl_local_rank)
    device.set_current()

    v = core.get_version()
    # nccl library version is numeric x.y.z
    major, minor, patch = map(int, str(v.nccl_version).split("."))
    assert major >= 0 and minor >= 0 and patch >= 0
    # nccl4py version should be parseable as x.y.z
    m2, n2, p2 = map(int, str(v.nccl4py_version).split("."))
    assert m2 >= 0 and n2 >= 0 and p2 >= 0

    assert str(v.nccl4py_version) == __version__

    # Initialize a communicator to trigger NCCL version logging
    comm = core.Communicator.init(
        nranks=rank_info.nccl_size,
        rank=rank_info.nccl_rank,
        unique_id=uid_shared
    )
    comm.destroy()

    # Read version from NCCL debug log
    nccl_debug_file = get_nccl_debug_file()
    if nccl_debug_file is None or not os.path.exists(nccl_debug_file):
        pytest.skip("NCCL debug file not available")

    with open(nccl_debug_file) as f:
        debug_log = f.read()

    # Look for "NCCL version X.Y.Z+cudaA.B" in debug log
    pat = re.compile(r"NCCL version\s+(\d+\.\d+\.\d+)\+cuda\d+\.\d+")
    m = pat.search(debug_log)
    if m:
        expected = m.group(1)
        assert str(v.nccl_version) == expected, f"Version mismatch: API returned {v.nccl_version}, debug log shows {expected}"

# ---------------- UniqueId tests ----------------
@pytest.mark.mpi(min_size=1)
def test_ptr_memory_matches_as_bytes():
    """API test: Verify ptr points to actual memory matching as_bytes."""
    uid = core.get_unique_id()
    raw = ctypes.string_at(uid.ptr, 128)
    assert raw == uid.as_bytes


@pytest.mark.mpi(min_size=1)
def test_unique_id_distinctness():
    """API test: Verify different get_unique_id() calls produce distinct IDs."""
    uid1 = core.get_unique_id()
    uid2 = core.get_unique_id()
    assert uid1.as_bytes != uid2.as_bytes


@pytest.mark.mpi(min_size=1)
def test_get_error_string():
    """Test get_error_string returns exact messages for all Result values."""
    # Map from C++ ncclGetErrorString (src/init.cc:2639-2652)
    expected_messages = {
        nccl_bindings.Result.Success: "no error",
        nccl_bindings.Result.UnhandledCudaError: "unhandled cuda error (run with NCCL_DEBUG=INFO for details)",
        nccl_bindings.Result.SystemError: "unhandled system error (run with NCCL_DEBUG=INFO for details)",
        nccl_bindings.Result.InternalError: "internal error - please report this issue to the NCCL developers",
        nccl_bindings.Result.InvalidArgument: "invalid argument (run with NCCL_DEBUG=WARN for details)",
        nccl_bindings.Result.InvalidUsage: "invalid usage (run with NCCL_DEBUG=WARN for details)",
        nccl_bindings.Result.RemoteError: "remote process exited or there was a network error",
        nccl_bindings.Result.InProgress: "NCCL operation in progress",
    }

    for result, expected in expected_messages.items():
        msg = core.get_error_string(result)
        assert msg == expected, f"Result.{result.name}: expected '{expected}', got '{msg}'"

    # Test with unknown/invalid result code
    msg = core.get_error_string(9999)
    assert msg == "unknown result code"
