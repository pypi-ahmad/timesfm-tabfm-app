"""Managed local TimesFM LoRA job orchestration."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODE_EXTERNAL_SCRIPT = "external_script"
MODE_IN_APP = "in_app"
SUPPORTED_JOB_MODES = {MODE_EXTERNAL_SCRIPT, MODE_IN_APP}


class LoRAJobError(RuntimeError):
    """Raised when LoRA jobs cannot be started, queried, or stopped."""


@dataclass(frozen=True)
class LoRAJobConfig:
    """Serializable LoRA training/evaluation configuration."""

    script_path: str
    output_dir: str
    mode: str = MODE_EXTERNAL_SCRIPT
    dataset_path: str | None = None
    train_dataset_path: str | None = None
    validation_dataset_path: str | None = None
    eval_only: bool = False
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 5e-5
    lora_r: int = 8
    lora_alpha: int = 16
    context_len: int = 64
    horizon_len: int = 13
    backend: str = "torch"
    base_model_id: str = ""
    adapter_name: str = "timesfm_lora_adapter"
    dataset_fingerprint: str = ""
    dataset_spec: dict[str, Any] | None = None
    retention_policy: str = "delete_raw"
    adapter_path: str | None = None
    metrics_path: str | None = None
    extra_args: str = ""


@dataclass(frozen=True)
class LoRAJobStatus:
    """Current status snapshot for one LoRA job."""

    job_id: str
    status: str
    command: str
    created_at: str
    started_at: str
    finished_at: str | None
    return_code: int | None
    output_dir: str
    log_path: str
    config: dict[str, Any]
    mode: str = MODE_EXTERNAL_SCRIPT
    backend: str = "torch"
    base_model_id: str = ""
    adapter_path: str | None = None
    metrics_path: str | None = None
    dataset_fingerprint: str = ""


_PROCESS_REGISTRY: dict[str, subprocess.Popen[bytes]] = {}


def _utc_now_label() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_registry(registry_path: Path) -> dict[str, Any]:
    if not registry_path.exists():
        return {}
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LoRAJobError(f"LoRA registry must be a JSON object: {registry_path}")
    return payload


def _write_registry(registry_path: Path, payload: dict[str, Any]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _normalize_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in SUPPORTED_JOB_MODES:
        raise LoRAJobError(
            "LoRA mode must be one of: external_script, in_app."
        )
    return normalized


def _normalize_job_entry(entry: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(entry)
    normalized.setdefault("mode", MODE_EXTERNAL_SCRIPT)
    normalized.setdefault("backend", "torch")
    normalized.setdefault("base_model_id", "")
    normalized.setdefault("adapter_path", None)
    normalized.setdefault("metrics_path", None)
    normalized.setdefault("dataset_fingerprint", "")
    normalized.setdefault("config", {})
    normalized.setdefault("created_at", normalized.get("started_at", _utc_now_label()))
    normalized.setdefault("started_at", normalized["created_at"])
    normalized.setdefault("finished_at", None)
    normalized.setdefault("return_code", None)
    normalized.setdefault("status", "running")
    return normalized


def _build_command(config: LoRAJobConfig) -> list[str]:
    mode = _normalize_mode(config.mode)
    if mode == MODE_EXTERNAL_SCRIPT:
        command = [sys.executable, config.script_path]
    else:
        command = [sys.executable, "-m", "app_core.timesfm_lora_runner"]

    if config.eval_only:
        command.append("--eval_only")
    command.extend(["--output_dir", config.output_dir])
    command.extend(["--epochs", str(config.epochs)])
    command.extend(["--batch_size", str(config.batch_size)])
    command.extend(["--lr", str(config.learning_rate)])
    command.extend(["--lora_r", str(config.lora_r)])
    command.extend(["--lora_alpha", str(config.lora_alpha)])
    command.extend(["--context_len", str(config.context_len)])
    command.extend(["--horizon_len", str(config.horizon_len)])
    command.extend(["--backend", config.backend])
    if config.base_model_id.strip():
        command.extend(["--base_model_id", config.base_model_id.strip()])
    if config.adapter_name.strip():
        command.extend(["--adapter_name", config.adapter_name.strip()])
    if config.dataset_path:
        command.extend(["--dataset_path", config.dataset_path])
    if config.train_dataset_path:
        command.extend(["--train_dataset_path", config.train_dataset_path])
    if config.validation_dataset_path:
        command.extend(["--validation_dataset_path", config.validation_dataset_path])
    if config.dataset_fingerprint.strip():
        command.extend(["--dataset_fingerprint", config.dataset_fingerprint.strip()])
    if config.extra_args.strip():
        command.extend(shlex.split(config.extra_args.strip()))
    return command


def start_lora_job(config: LoRAJobConfig, registry_path: Path) -> LoRAJobStatus:
    """Start a managed local LoRA job and persist metadata."""
    mode = _normalize_mode(config.mode)
    if mode == MODE_EXTERNAL_SCRIPT:
        script = Path(config.script_path)
        if not script.exists():
            raise LoRAJobError(f"LoRA script was not found: {config.script_path}")

    jobs = _read_registry(registry_path)
    job_id = uuid.uuid4().hex[:12]
    output_dir = Path(config.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"lora-job-{job_id}.log"
    command = _build_command(config)
    env = os.environ.copy()
    src_path = str(Path(__file__).resolve().parents[1])
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = (
        f"{src_path}:{existing_pythonpath}" if existing_pythonpath else src_path
    )

    log_file = log_path.open("wb")
    try:
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(output_dir),
            env=env,
        )
    finally:
        log_file.close()
    _PROCESS_REGISTRY[job_id] = process

    now = _utc_now_label()
    adapter_path = config.adapter_path or str(output_dir / "adapter")
    metrics_path = config.metrics_path or str(output_dir / "eval_metrics.json")
    entry = {
        "job_id": job_id,
        "status": "running",
        "command": " ".join(command),
        "created_at": now,
        "started_at": now,
        "finished_at": None,
        "return_code": None,
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "config": asdict(config),
        "mode": mode,
        "backend": str(config.backend).strip().lower(),
        "base_model_id": config.base_model_id,
        "adapter_path": adapter_path,
        "metrics_path": metrics_path,
        "dataset_fingerprint": config.dataset_fingerprint,
    }
    jobs[job_id] = entry
    _write_registry(registry_path, jobs)
    return LoRAJobStatus(**entry)


def refresh_lora_job(job_id: str, registry_path: Path) -> LoRAJobStatus:
    """Refresh one LoRA job status from process state and registry metadata."""
    jobs = _read_registry(registry_path)
    if job_id not in jobs:
        raise LoRAJobError(f"LoRA job does not exist: {job_id}")
    entry = _normalize_job_entry(jobs[job_id])

    process = _PROCESS_REGISTRY.get(job_id)
    if process is not None and entry["status"] == "running":
        return_code = process.poll()
        if return_code is not None:
            entry["status"] = "completed" if return_code == 0 else "failed"
            entry["return_code"] = int(return_code)
            entry["finished_at"] = _utc_now_label()
            _PROCESS_REGISTRY.pop(job_id, None)
            jobs[job_id] = entry
            _write_registry(registry_path, jobs)
    else:
        jobs[job_id] = entry
        _write_registry(registry_path, jobs)

    return LoRAJobStatus(**entry)


def stop_lora_job(job_id: str, registry_path: Path) -> LoRAJobStatus:
    """Terminate a running LoRA job."""
    jobs = _read_registry(registry_path)
    if job_id not in jobs:
        raise LoRAJobError(f"LoRA job does not exist: {job_id}")
    entry = _normalize_job_entry(jobs[job_id])
    process = _PROCESS_REGISTRY.get(job_id)
    if process is None:
        return refresh_lora_job(job_id=job_id, registry_path=registry_path)

    process.terminate()
    try:
        return_code = process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        return_code = process.wait(timeout=5)
    _PROCESS_REGISTRY.pop(job_id, None)
    entry["status"] = "stopped"
    entry["return_code"] = int(return_code)
    entry["finished_at"] = _utc_now_label()
    jobs[job_id] = entry
    _write_registry(registry_path, jobs)
    return LoRAJobStatus(**entry)


def list_lora_jobs(registry_path: Path) -> list[LoRAJobStatus]:
    """List all persisted LoRA jobs sorted by creation time descending."""
    jobs = _read_registry(registry_path)
    statuses = [LoRAJobStatus(**_normalize_job_entry(entry)) for entry in jobs.values()]
    return sorted(statuses, key=lambda item: item.created_at, reverse=True)
