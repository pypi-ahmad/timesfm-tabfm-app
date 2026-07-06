"""Tests for TimesFM LoRA adapter registry behavior."""

from __future__ import annotations

from pathlib import Path

from app_core.lora_jobs import LoRAJobStatus
from app_core.timesfm_lora_adapters import (
    ensure_adapter_registered_from_job,
    list_lora_adapters,
    resolve_lora_adapter_path,
)


def _build_status(output_dir: Path) -> LoRAJobStatus:
    return LoRAJobStatus(
        job_id="abc123def456",
        status="completed",
        command="python -m app_core.timesfm_lora_runner",
        created_at="2026-01-01T00:00:00+00:00",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:02+00:00",
        return_code=0,
        output_dir=str(output_dir),
        log_path=str(output_dir / "job.log"),
        config={
            "adapter_name": "retail_adapter",
            "backend": "torch",
            "base_model_id": "google/timesfm-2.5-200m-pytorch",
        },
        mode="in_app",
        backend="torch",
        base_model_id="google/timesfm-2.5-200m-pytorch",
        adapter_path=str(output_dir / "adapter"),
        metrics_path=str(output_dir / "eval_metrics.json"),
        dataset_fingerprint="deadbeef",
    )


def test_ensure_adapter_registered_from_job_and_resolve_path(tmp_path: Path) -> None:
    output_dir = tmp_path / "run_1"
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eval_metrics.json").write_text('{"mae": 0.1}', encoding="utf-8")
    registry_path = tmp_path / "adapters.json"

    record = ensure_adapter_registered_from_job(
        job=_build_status(output_dir),
        registry_path=registry_path,
    )
    assert record is not None
    resolved = resolve_lora_adapter_path(
        registry_path=registry_path,
        adapter_id=record.adapter_id,
    )
    assert resolved == str(adapter_dir.resolve())
    listed = list_lora_adapters(registry_path=registry_path)
    assert len(listed) == 1
    assert listed[0].source_job_id == "abc123def456"


def test_ensure_adapter_registered_from_job_is_idempotent(tmp_path: Path) -> None:
    output_dir = tmp_path / "run_2"
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    registry_path = tmp_path / "adapters.json"
    status = _build_status(output_dir)

    first = ensure_adapter_registered_from_job(job=status, registry_path=registry_path)
    second = ensure_adapter_registered_from_job(job=status, registry_path=registry_path)
    assert first is not None
    assert second is not None
    assert first.adapter_id == second.adapter_id
