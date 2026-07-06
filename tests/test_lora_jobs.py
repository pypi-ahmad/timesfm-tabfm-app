"""Tests for managed LoRA job orchestration."""

from __future__ import annotations

import time
from pathlib import Path

from app_core.lora_jobs import (
    LoRAJobConfig,
    list_lora_jobs,
    refresh_lora_job,
    start_lora_job,
)


def _create_fake_lora_script(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "import argparse",
                "import time",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--output_dir')",
                "parser.add_argument('--epochs')",
                "parser.add_argument('--batch_size')",
                "parser.add_argument('--lr')",
                "parser.add_argument('--lora_r')",
                "parser.add_argument('--lora_alpha')",
                "parser.add_argument('--context_len')",
                "parser.add_argument('--horizon_len')",
                "parser.add_argument('--dataset_path', default=None)",
                "parser.add_argument('--eval_only', action='store_true')",
                "parser.parse_known_args()",
                "time.sleep(0.1)",
                "print('ok')",
            ]
        ),
        encoding="utf-8",
    )


def test_start_and_refresh_lora_job(tmp_path: Path) -> None:
    script_path = tmp_path / "finetune_lora.py"
    _create_fake_lora_script(script_path)
    registry_path = tmp_path / "registry.json"
    output_dir = tmp_path / "outputs"

    status = start_lora_job(
        config=LoRAJobConfig(
            script_path=str(script_path),
            output_dir=str(output_dir),
            epochs=1,
            batch_size=4,
        ),
        registry_path=registry_path,
    )
    assert status.status == "running"

    for _ in range(10):
        refreshed = refresh_lora_job(status.job_id, registry_path=registry_path)
        if refreshed.status in {"completed", "failed"}:
            break
        time.sleep(0.05)
    assert refreshed.status == "completed"
    assert refreshed.return_code == 0

    listed = list_lora_jobs(registry_path=registry_path)
    assert any(item.job_id == status.job_id for item in listed)


def test_start_and_refresh_in_app_lora_job(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    output_dir = tmp_path / "in_app_outputs"
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    train_path.write_text(
        "\n".join(
            [
                "timestamp,value,entity_id",
                "2024-01-01,1.0,GLOBAL",
                "2024-01-02,2.0,GLOBAL",
                "2024-01-03,3.0,GLOBAL",
                "2024-01-04,4.0,GLOBAL",
                "2024-01-05,5.0,GLOBAL",
            ]
        ),
        encoding="utf-8",
    )
    validation_path.write_text(
        "\n".join(
            [
                "timestamp,value,entity_id",
                "2024-01-06,6.0,GLOBAL",
                "2024-01-07,7.0,GLOBAL",
            ]
        ),
        encoding="utf-8",
    )

    status = start_lora_job(
        config=LoRAJobConfig(
            mode="in_app",
            script_path="unused.py",
            output_dir=str(output_dir),
            train_dataset_path=str(train_path),
            validation_dataset_path=str(validation_path),
            epochs=1,
            batch_size=2,
            base_model_id="google/timesfm-2.5-200m-pytorch",
            adapter_name="test_adapter",
        ),
        registry_path=registry_path,
    )
    assert status.mode == "in_app"
    assert status.status == "running"

    for _ in range(30):
        refreshed = refresh_lora_job(status.job_id, registry_path=registry_path)
        if refreshed.status in {"completed", "failed"}:
            break
        time.sleep(0.05)
    assert refreshed.status == "completed"
    assert refreshed.return_code == 0
    assert refreshed.adapter_path is not None
    assert Path(refreshed.adapter_path).exists()
