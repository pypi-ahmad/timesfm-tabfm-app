"""Tests for TimesFM LoRA transactional dataset preprocessing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app_core.timesfm_lora_data import (
    RETENTION_DELETE_RAW,
    RETENTION_KEEP_RAW,
    TransactionalDatasetSpec,
    materialize_lora_dataset_from_csv_bytes,
)
from app_core.validators import ValidationError


def _frame_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def test_materialize_lora_dataset_univariate_deletes_raw(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=12, freq="D"),
            "value": [100 + idx for idx in range(12)],
        }
    )
    spec = TransactionalDatasetSpec(
        timestamp_column="timestamp",
        value_column="value",
        validation_ratio=0.25,
        min_points_per_entity=6,
    )
    materialized = materialize_lora_dataset_from_csv_bytes(
        csv_bytes=_frame_to_csv_bytes(frame),
        output_dir=tmp_path / "dataset",
        spec=spec,
        retention_policy=RETENTION_DELETE_RAW,
    )

    assert Path(materialized.train_path).exists()
    assert Path(materialized.validation_path).exists()
    assert Path(materialized.metadata_path).exists()
    assert materialized.retained_raw_path is None
    assert materialized.rows_total == 12
    assert materialized.rows_train == 9
    assert materialized.rows_validation == 3
    assert materialized.entity_count == 1


def test_materialize_lora_dataset_panel_keeps_raw(tmp_path: Path) -> None:
    rows = []
    for entity_id in ["STORE_A", "STORE_B"]:
        for idx in range(8):
            rows.append(
                {
                    "ts": f"2024-01-{idx + 1:02d}",
                    "sales": 100 + idx,
                    "store": entity_id,
                    "promo_index": idx % 2,
                }
            )
    frame = pd.DataFrame(rows)
    spec = TransactionalDatasetSpec(
        timestamp_column="ts",
        value_column="sales",
        entity_column="store",
        feature_columns=["promo_index"],
        validation_ratio=0.25,
        min_points_per_entity=5,
    )
    materialized = materialize_lora_dataset_from_csv_bytes(
        csv_bytes=_frame_to_csv_bytes(frame),
        output_dir=tmp_path / "panel_dataset",
        spec=spec,
        retention_policy=RETENTION_KEEP_RAW,
    )

    assert materialized.entity_count == 2
    assert materialized.rows_total == 16
    assert materialized.retained_raw_path is not None
    assert Path(materialized.retained_raw_path).exists()


def test_materialize_lora_dataset_raises_on_missing_columns(tmp_path: Path) -> None:
    frame = pd.DataFrame({"timestamp": ["2024-01-01"], "value": [1.0]})
    spec = TransactionalDatasetSpec(
        timestamp_column="timestamp",
        value_column="value",
        entity_column="ticker",
        min_points_per_entity=3,
    )
    with pytest.raises(ValidationError, match="missing required columns"):
        materialize_lora_dataset_from_csv_bytes(
            csv_bytes=_frame_to_csv_bytes(frame),
            output_dir=tmp_path / "invalid_dataset",
            spec=spec,
            retention_policy=RETENTION_DELETE_RAW,
        )
