"""Utility helpers for dashboard serialization."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any


def to_serializable(value: Any) -> Any:
    """Recursively convert dataclasses and datetimes into JSON-friendly structures."""

    if is_dataclass(value):
        return {key: to_serializable(val) for key, val in asdict(value).items()}
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(item) for item in value]
    return value


__all__ = ["to_serializable"]
