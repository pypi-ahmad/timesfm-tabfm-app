"""Helpers for writing markdown run artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


def append_markdown_run(
    output_path: str,
    section_title: str,
    body_markdown: str,
) -> None:
    """Append one timestamped run section to the markdown output file."""
    path = Path(output_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    content = (
        f"\n\n## {section_title}\n"
        f"_Run at: {timestamp}_\n\n"
        f"{body_markdown.strip()}\n"
    )
    if not path.exists():
        path.write_text("# Unified App Outputs\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(content)

