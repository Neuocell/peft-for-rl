from __future__ import annotations

from scripts.analysis import record_gpu_memory


def test_nvidia_smi_csv_sample(monkeypatch) -> None:
    monkeypatch.setattr(
        record_gpu_memory.subprocess,
        "check_output",
        lambda *args, **kwargs: "0, 32550, 83\n1, 32600, 79\n",
    )
    assert record_gpu_memory.sample() == [(0, 32550, 83), (1, 32600, 79)]
