"""Registry utilities for TimesFM LoRA adapters."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app_core.lora_jobs import LoRAJobStatus


class LoRAAdapterError(RuntimeError):
    """Raised when adapter registry operations fail."""


@dataclass(frozen=True)
class LoRAAdapterRecord:
    """One registered LoRA adapter artifact."""

    adapter_id: str
    name: str
    adapter_path: str
    backend: str
    base_model_id: str
    source_job_id: str
    created_at: str
    metrics_path: str | None
    dataset_fingerprint: str
    metadata: dict[str, Any]


def _read_registry(registry_path: Path) -> dict[str, Any]:
    if not registry_path.exists():
        return {}
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise LoRAAdapterError(f"Adapter registry must be a JSON object: {registry_path}")
    return payload


def _write_registry(registry_path: Path, payload: dict[str, Any]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def list_lora_adapters(
    registry_path: Path,
    backend: str | None = None,
) -> list[LoRAAdapterRecord]:
    """List adapters sorted by creation time descending."""
    normalized_backend = backend.strip().lower() if backend else None
    entries = []
    for payload in _read_registry(registry_path).values():
        record = LoRAAdapterRecord(**payload)
        if normalized_backend and record.backend.strip().lower() != normalized_backend:
            continue
        entries.append(record)
    return sorted(entries, key=lambda item: item.created_at, reverse=True)


def register_lora_adapter(
    registry_path: Path,
    *,
    name: str,
    adapter_path: str,
    backend: str,
    base_model_id: str,
    source_job_id: str,
    metrics_path: str | None = None,
    dataset_fingerprint: str = "",
    metadata: dict[str, Any] | None = None,
) -> LoRAAdapterRecord:
    """Register one adapter artifact in the adapter registry."""
    path = Path(adapter_path)
    if not path.exists():
        raise LoRAAdapterError(f"Adapter path does not exist: {adapter_path}")

    created_at = datetime.now(timezone.utc).isoformat()
    adapter_id = f"adapter_{uuid.uuid4().hex[:12]}"
    record = LoRAAdapterRecord(
        adapter_id=adapter_id,
        name=name.strip() or adapter_id,
        adapter_path=str(path.resolve()),
        backend=backend.strip().lower(),
        base_model_id=base_model_id.strip(),
        source_job_id=source_job_id.strip(),
        created_at=created_at,
        metrics_path=str(Path(metrics_path).resolve()) if metrics_path else None,
        dataset_fingerprint=dataset_fingerprint,
        metadata=metadata or {},
    )
    registry = _read_registry(registry_path)
    registry[record.adapter_id] = asdict(record)
    _write_registry(registry_path, registry)
    return record


def ensure_adapter_registered_from_job(
    *,
    job: LoRAJobStatus,
    registry_path: Path,
) -> LoRAAdapterRecord | None:
    """Create adapter record from completed job if not already registered."""
    if job.status != "completed":
        return None
    if not job.adapter_path:
        return None
    adapter_path = Path(job.adapter_path)
    if not adapter_path.exists():
        return None

    existing_by_job = [
        item
        for item in list_lora_adapters(registry_path=registry_path)
        if item.source_job_id == job.job_id
    ]
    if existing_by_job:
        return existing_by_job[0]

    config = dict(job.config or {})
    adapter_name = str(config.get("adapter_name", "")).strip() or f"job_{job.job_id}"
    metrics_path = job.metrics_path if job.metrics_path and Path(job.metrics_path).exists() else None
    metadata = {
        "mode": job.mode,
        "output_dir": job.output_dir,
        "job_created_at": job.created_at,
        "job_finished_at": job.finished_at,
    }
    return register_lora_adapter(
        registry_path=registry_path,
        name=adapter_name,
        adapter_path=str(adapter_path),
        backend=job.backend or str(config.get("backend", "torch")),
        base_model_id=job.base_model_id or str(config.get("base_model_id", "")),
        source_job_id=job.job_id,
        metrics_path=metrics_path,
        dataset_fingerprint=job.dataset_fingerprint or str(config.get("dataset_fingerprint", "")),
        metadata=metadata,
    )


def resolve_lora_adapter_path(
    *,
    registry_path: Path,
    adapter_id: str,
) -> str:
    """Resolve adapter path by ID and ensure the path exists."""
    for record in list_lora_adapters(registry_path=registry_path):
        if record.adapter_id == adapter_id:
            if not Path(record.adapter_path).exists():
                raise LoRAAdapterError(
                    f"Adapter exists in registry but path is missing: {record.adapter_path}"
                )
            return record.adapter_path
    raise LoRAAdapterError(f"Adapter was not found: {adapter_id}")
