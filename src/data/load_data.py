"""Minimal safe data-loading helpers used by audit scripts."""

from __future__ import annotations

import json
import zipfile
from io import TextIOWrapper
from pathlib import Path
from typing import Any

import pandas as pd


def list_zip_members(zip_path: Path) -> list[dict[str, Any]]:
    """Return member names and sizes for a ZIP archive without extracting it."""

    with zipfile.ZipFile(zip_path) as archive:
        return [
            {"name": member.filename, "size": member.file_size}
            for member in archive.infolist()
            if not member.is_dir()
        ]


def zip_member_exists(zip_path: Path, member_name: str) -> bool:
    """Check whether a ZIP archive contains ``member_name``."""

    with zipfile.ZipFile(zip_path) as archive:
        return member_name in archive.namelist()


def read_json_from_zip(zip_path: Path, member_name: str) -> Any:
    """Read a JSON member from a ZIP archive without extracting it."""

    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member_name) as raw_file:
            return json.load(TextIOWrapper(raw_file, encoding="utf-8"))


def read_csv_from_zip(zip_path: Path, member_name: str, sep: str | None = None) -> pd.DataFrame:
    """Read a CSV/TSV member from a ZIP archive without extracting it."""

    separator = _infer_separator(member_name, sep)
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member_name) as raw_file:
            return pd.read_csv(TextIOWrapper(raw_file, encoding="utf-8"), sep=separator)


def read_table(path: Path, sep: str | None = None) -> pd.DataFrame:
    """Read a small CSV/TSV table using a separator inferred from the suffix."""

    return pd.read_csv(path, sep=_infer_separator(str(path), sep))


def count_nonempty_lines(path: Path) -> int:
    """Count non-empty text lines without interpreting the file schema."""

    count = 0
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                count += 1
    return count


def read_json(path: Path) -> Any:
    """Read a JSON file."""

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _infer_separator(path_or_name: str, sep: str | None) -> str:
    if sep is not None:
        return sep
    return "\t" if path_or_name.lower().endswith((".tsv", ".txt")) else ","
