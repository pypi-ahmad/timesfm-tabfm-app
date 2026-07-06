"""Utilities for loading many CSVs from direct upload and ZIP blobs."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Sequence

import pandas as pd

from app_core.validators import ValidationError


@dataclass(frozen=True)
class BatchInputItem:
    """In-memory batch input with deterministic filename."""

    name: str
    dataframe: pd.DataFrame


def _dedupe_name(name: str, seen: set[str]) -> str:
    if name not in seen:
        seen.add(name)
        return name

    stem, dot, ext = name.rpartition(".")
    stem = stem or ext
    ext = ext if dot else ""
    counter = 2
    while True:
        candidate = f"{stem}_{counter}.{ext}" if ext else f"{stem}_{counter}"
        if candidate not in seen:
            seen.add(candidate)
            return candidate
        counter += 1


def _parse_zip_csv_members(zip_name: str, zip_bytes: bytes) -> list[tuple[str, bytes]]:
    parsed: list[tuple[str, bytes]] = []
    try:
        archive = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile as exc:
        raise ValidationError(f"ZIP '{zip_name}' is invalid or corrupted.") from exc

    with archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            member_path = PurePosixPath(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValidationError(f"Unsafe ZIP path detected: '{member.filename}'.")

            file_name = member_path.name
            if not file_name.lower().endswith(".csv"):
                raise ValidationError(
                    f"Non-CSV file '{member.filename}' found in ZIP '{zip_name}'."
                )
            parsed.append((file_name, archive.read(member)))
    return parsed


def load_batch_items_from_bytes(
    csv_files: Sequence[tuple[str, bytes]],
    zip_files: Sequence[tuple[str, bytes]],
    max_files: int = 25,
) -> list[BatchInputItem]:
    """Load and parse batch inputs from raw CSV and ZIP bytes."""
    merged_files: list[tuple[str, bytes]] = []
    merged_files.extend(csv_files)
    for zip_name, zip_bytes in zip_files:
        merged_files.extend(_parse_zip_csv_members(zip_name, zip_bytes))

    if not merged_files:
        return []

    if len(merged_files) > max_files:
        raise ValidationError(f"Maximum batch size is {max_files} files per run.")

    seen_names: set[str] = set()
    items: list[BatchInputItem] = []
    for original_name, file_bytes in merged_files:
        if not original_name.lower().endswith(".csv"):
            raise ValidationError(f"Only CSV files are supported, got '{original_name}'.")
        unique_name = _dedupe_name(PurePosixPath(original_name).name, seen_names)
        try:
            dataframe = pd.read_csv(io.BytesIO(file_bytes))
        except Exception as exc:
            raise ValidationError(f"Could not parse CSV '{original_name}'.") from exc
        items.append(BatchInputItem(name=unique_name, dataframe=dataframe))

    return items

