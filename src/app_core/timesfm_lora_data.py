"""TimesFM LoRA dataset normalization and materialization utilities."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app_core.validators import ValidationError

RETENTION_DELETE_RAW = "delete_raw"
RETENTION_KEEP_RAW = "keep_raw"


@dataclass(frozen=True)
class TransactionalDatasetSpec:
    """User-selected schema mapping and split policy for transactional data."""

    timestamp_column: str = "timestamp"
    value_column: str = "value"
    entity_column: str | None = None
    feature_columns: list[str] = field(default_factory=list)
    validation_ratio: float = 0.2
    min_points_per_entity: int = 20


@dataclass(frozen=True)
class MaterializedLoRADataset:
    """Materialized LoRA-ready train/validation dataset artifact paths."""

    normalized_path: str
    train_path: str
    validation_path: str
    metadata_path: str
    fingerprint: str
    rows_total: int
    rows_train: int
    rows_validation: int
    entity_count: int
    retained_raw_path: str | None


def _normalize_feature_columns(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    normalized = df.copy()
    for feature_column in feature_columns:
        normalized[feature_column] = pd.to_numeric(
            normalized[feature_column],
            errors="coerce",
        )
    return normalized


def normalize_transactional_dataframe(
    dataframe: pd.DataFrame,
    spec: TransactionalDatasetSpec,
) -> pd.DataFrame:
    """Validate and normalize transactional dataframe to canonical schema."""
    if not 0.01 <= float(spec.validation_ratio) <= 0.5:
        raise ValidationError("LoRA validation ratio must be between 0.01 and 0.5.")
    if int(spec.min_points_per_entity) < 3:
        raise ValidationError("LoRA min_points_per_entity must be at least 3.")

    required = {spec.timestamp_column, spec.value_column}
    if spec.entity_column:
        required.add(spec.entity_column)
    missing_required = sorted(required.difference(set(dataframe.columns)))
    if missing_required:
        raise ValidationError(
            f"LoRA dataset is missing required columns: {missing_required}."
        )

    missing_features = sorted(
        column for column in spec.feature_columns if column not in dataframe.columns
    )
    if missing_features:
        raise ValidationError(
            f"LoRA dataset is missing selected feature columns: {missing_features}."
        )

    selected_columns = [
        spec.timestamp_column,
        spec.value_column,
        *( [spec.entity_column] if spec.entity_column else [] ),
        *spec.feature_columns,
    ]
    normalized = dataframe[selected_columns].copy()
    normalized[spec.timestamp_column] = pd.to_datetime(
        normalized[spec.timestamp_column],
        errors="coerce",
    )
    normalized[spec.value_column] = pd.to_numeric(
        normalized[spec.value_column],
        errors="coerce",
    )
    normalized = _normalize_feature_columns(
        df=normalized,
        feature_columns=spec.feature_columns,
    )

    if spec.entity_column:
        normalized[spec.entity_column] = normalized[spec.entity_column].astype(str).str.strip()
        normalized = normalized[normalized[spec.entity_column] != ""]
        normalized["entity_id"] = normalized[spec.entity_column]
    else:
        normalized["entity_id"] = "GLOBAL"

    normalized = normalized.dropna(subset=[spec.timestamp_column, spec.value_column]).copy()
    normalized = normalized.rename(
        columns={
            spec.timestamp_column: "timestamp",
            spec.value_column: "value",
        }
    )

    selected_after = ["timestamp", "value", "entity_id", *spec.feature_columns]
    normalized = normalized[selected_after]
    normalized = normalized.drop_duplicates(subset=["entity_id", "timestamp"], keep="last")
    normalized = normalized.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)

    if normalized.empty:
        raise ValidationError(
            "LoRA dataset is empty after parsing timestamp/value and filtering invalid rows."
        )

    counts = normalized.groupby("entity_id").size()
    valid_entities = counts[counts >= int(spec.min_points_per_entity)].index.tolist()
    if not valid_entities:
        raise ValidationError(
            "No entity has enough points for LoRA split. "
            f"Minimum points per entity: {int(spec.min_points_per_entity)}."
        )
    normalized = normalized[normalized["entity_id"].isin(valid_entities)].copy()
    normalized = normalized.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
    return normalized


def split_transactional_dataset(
    normalized_df: pd.DataFrame,
    validation_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split normalized data by entity using a chronological holdout."""
    train_frames: list[pd.DataFrame] = []
    validation_frames: list[pd.DataFrame] = []
    for _, entity_df in normalized_df.groupby("entity_id", sort=True):
        entity_df = entity_df.sort_values("timestamp").reset_index(drop=True)
        split_index = int(len(entity_df) * (1.0 - float(validation_ratio)))
        split_index = max(2, min(split_index, len(entity_df) - 1))
        train_frames.append(entity_df.iloc[:split_index].copy())
        validation_frames.append(entity_df.iloc[split_index:].copy())

    train_df = pd.concat(train_frames, ignore_index=True)
    validation_df = pd.concat(validation_frames, ignore_index=True)
    if train_df.empty or validation_df.empty:
        raise ValidationError("LoRA train/validation split failed due to insufficient rows.")
    return train_df, validation_df


def _dataset_fingerprint(normalized_df: pd.DataFrame, spec: TransactionalDatasetSpec) -> str:
    payload = normalized_df.to_csv(index=False).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(payload)
    digest.update(json.dumps(asdict(spec), sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def materialize_lora_dataset_from_csv_bytes(
    csv_bytes: bytes,
    output_dir: Path,
    spec: TransactionalDatasetSpec,
    retention_policy: str = RETENTION_DELETE_RAW,
) -> MaterializedLoRADataset:
    """Parse, normalize, split, and persist LoRA training datasets."""
    if retention_policy not in {RETENTION_DELETE_RAW, RETENTION_KEEP_RAW}:
        raise ValidationError(
            "LoRA retention policy must be one of: delete_raw, keep_raw."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_upload_path = output_dir / "raw_upload.csv"
    raw_upload_path.write_bytes(csv_bytes)

    try:
        dataframe = pd.read_csv(io.BytesIO(csv_bytes))
    except Exception as exc:
        raise ValidationError("LoRA dataset CSV could not be parsed.") from exc

    normalized_df = normalize_transactional_dataframe(dataframe=dataframe, spec=spec)
    train_df, validation_df = split_transactional_dataset(
        normalized_df=normalized_df,
        validation_ratio=float(spec.validation_ratio),
    )
    fingerprint = _dataset_fingerprint(normalized_df=normalized_df, spec=spec)

    normalized_path = output_dir / "normalized_dataset.csv"
    train_path = output_dir / "train_dataset.csv"
    validation_path = output_dir / "validation_dataset.csv"
    metadata_path = output_dir / "dataset_metadata.json"
    normalized_df.to_csv(normalized_path, index=False)
    train_df.to_csv(train_path, index=False)
    validation_df.to_csv(validation_path, index=False)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fingerprint": fingerprint,
        "schema": asdict(spec),
        "rows": {
            "total": int(len(normalized_df)),
            "train": int(len(train_df)),
            "validation": int(len(validation_df)),
        },
        "entity_count": int(normalized_df["entity_id"].nunique()),
        "retention_policy": retention_policy,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    retained_raw_path: str | None = str(raw_upload_path)
    if retention_policy == RETENTION_DELETE_RAW:
        raw_upload_path.unlink(missing_ok=True)
        retained_raw_path = None

    return MaterializedLoRADataset(
        normalized_path=str(normalized_path),
        train_path=str(train_path),
        validation_path=str(validation_path),
        metadata_path=str(metadata_path),
        fingerprint=fingerprint,
        rows_total=int(len(normalized_df)),
        rows_train=int(len(train_df)),
        rows_validation=int(len(validation_df)),
        entity_count=int(normalized_df["entity_id"].nunique()),
        retained_raw_path=retained_raw_path,
    )
