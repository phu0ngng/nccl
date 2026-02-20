import os
from pathlib import Path
import shutil
import socket
import warnings
from dataclasses import dataclass
from packaging.version import Version as _Version
from mpi4py import MPI
import pytest

import nccl.core as nccl


# Global NCCL library version for test skipping
# Initialized once at test collection time
NCCL_LIB_VERSION = nccl.get_version().nccl_version


def get_cuda_device_count():
    """
    Get the number of CUDA devices available on the node.

    Returns:
        int: Number of CUDA devices, or 0 if unable to determine
    """
    try:
        from cuda.core import system
        return system.get_num_devices()
    except (ImportError, AttributeError, RuntimeError) as e:
        # Fallback if cuda.core is not available or no CUDA devices
        warnings.warn(
            f"Unable to determine CUDA device count: {type(e).__name__}: {e}. "
            "Tests requiring multiple devices will be skipped.",
            UserWarning
        )
        return 0


@pytest.fixture(scope="session", autouse=True)
def setup_nccl(tmp_path_factory):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    root_dir = str(tmp_path_factory.mktemp("nccl_logs")) if rank == 0 else None
    root_dir = comm.bcast(root_dir, root=0)
    root_dir = Path(root_dir)

    logdir = root_dir / f"rank{rank}"
    logdir.mkdir(parents=True, exist_ok=True)

    comm.Barrier()

    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,ENV,GRAPH,BOOTSTRAP"
    os.environ["NCCL_DEBUG_FILE"] = str(logdir / "nccl4py_test.%h.%p.log")

    yield logdir


@pytest.fixture()
def get_nccl_debug_file():
    comm = MPI.COMM_WORLD

    hostname = socket.gethostname()
    process_id = os.getpid()

    nccl_debug_file_pattern = os.environ.get("NCCL_DEBUG_FILE", "")
    nccl_debug_file = nccl_debug_file_pattern.replace("%h", hostname).replace("%p", str(process_id))

    def _get_nccl_debug_file():
        comm.Barrier()

        return nccl_debug_file

    return _get_nccl_debug_file


@pytest.fixture()
def uid_shared():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    uid = nccl.get_unique_id(empty=(rank != 0))
    comm.Bcast([uid.as_ndarray, MPI.BYTE], root=0)
    return uid


@dataclass(frozen=True)
class RankInfo:
    mpi_rank: int
    mpi_size: int
    nccl_rank: int
    nccl_size: int
    nccl_local_rank: int

@pytest.fixture(scope="session")
def rank_info():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    host = socket.gethostname()
    hosts = comm.allgather(host)
    local_rank = sum(1 for h in hosts[:rank] if h == host)
    return RankInfo(mpi_rank=rank, mpi_size=size, nccl_rank=rank, nccl_size=size, nccl_local_rank=local_rank)


@pytest.fixture()
def nccl_comm(uid_shared, rank_info):
    """Create and destroy NCCL communicator for each test."""
    from cuda.core import Device

    device = Device(rank_info.nccl_local_rank)
    device.set_current()

    comm = nccl.Communicator.init(
        nranks=rank_info.nccl_size,
        rank=rank_info.nccl_rank,
        unique_id=uid_shared
    )

    yield comm

    comm.destroy()


def requires_nccl_version(min_version):
    """
    Helper to create skipif marker for NCCL version requirements.

    Args:
        min_version (str): Minimum required NCCL version (e.g., "2.19.0")

    Returns:
        pytest.mark.skipif: Pytest marker that skips test if NCCL version is too old

    Example:
        @requires_nccl_version("2.19.0")
        def test_window_api(nccl_comm):
            # Test code that requires NCCL >= 2.19.0
            ...
    """
    return pytest.mark.skipif(
        NCCL_LIB_VERSION < _Version(min_version),
        reason=f"Requires NCCL >= {min_version} (found {NCCL_LIB_VERSION})"
    )


def requires_min_devices(min_devices):
    """
    Helper to create skipif marker for minimum CUDA device requirements.

    Args:
        min_devices (int): Minimum number of CUDA devices required

    Returns:
        pytest.mark.skipif: Pytest marker that skips test if not enough devices available

    Example:
        @requires_min_devices(2)
        def test_multi_device(nccl_comm):
            # Test code that requires at least 2 CUDA devices
            ...
    """
    device_count = get_cuda_device_count()
    return pytest.mark.skipif(
        device_count < min_devices,
        reason=f"Requires at least {min_devices} CUDA device(s) (found {device_count})"
    )

