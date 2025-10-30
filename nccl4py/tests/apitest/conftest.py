import os
import shutil
import socket
from dataclasses import dataclass
from mpi4py import MPI
import pytest

import nccl.core as nccl


@pytest.fixture(scope="session", autouse=True)
def setup_nccl(tmp_path_factory):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    logdir = tmp_path_factory.mktemp("nccl_logs") if rank == 0 else None
    logdir = comm.bcast(logdir, root=0)

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
    from cuda.core.experimental import Device

    device = Device(rank_info.nccl_local_rank)
    device.set_current()

    comm = nccl.Communicator.init(
        nranks=rank_info.nccl_size,
        rank=rank_info.nccl_rank,
        unique_id=uid_shared
    )

    yield comm

    comm.destroy()
