import pytest
import sys

from nccl.core.group import group, group_start, group_end, group_simulate_end, GroupSimInfo


def test_group_context_calls_start_end(monkeypatch):
    calls = []
    class B:
        @staticmethod
        def group_start(): calls.append("start")
        @staticmethod
        def group_end(): calls.append("end")
        @staticmethod
        def group_simulate_end(x): calls.append(("sim", x))

    monkeypatch.setattr(sys.modules["nccl.core.group"], "_nccl_bindings", B)

    with group():
        pass
    assert calls == ["start", "end"]


def test_group_context_end_on_exception(monkeypatch):
    calls = []
    class B:
        @staticmethod
        def group_start(): calls.append("start")
        @staticmethod
        def group_end(): calls.append("end")
    monkeypatch.setattr(sys.modules["nccl.core.group"], "_nccl_bindings", B)

    with pytest.raises(RuntimeError):
        with group():
            raise RuntimeError("boom")
    assert calls == ["start", "end"]


def test_group_simulate_end_with_and_without_info(monkeypatch):
    rec = []
    class B:
        @staticmethod
        def group_simulate_end(p): rec.append(p)
    monkeypatch.setattr(sys.modules["nccl.core.group"], "_nccl_bindings", B)

    group_simulate_end(None)
    assert rec[-1] is None
    # Fake GroupSimInfo with ptr
    class G:
        def __init__(self): self._sim_info = type("S", (), {"ptr": 123})()
        @property
        def ptr(self): return int(self._sim_info.ptr)
    group_simulate_end(G())
    assert rec[-1] == 123



