"""Shared CLI utilities."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
from typing import Any


def serialisable(obj: Any) -> Any:
    """Recursively convert dataclass / date / set to JSON-serialisable types."""
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, set):
        return sorted(obj)
    if hasattr(obj, "__dataclass_fields__"):
        return {k: serialisable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [serialisable(i) for i in obj]
    if isinstance(obj, dict):
        return {k: serialisable(v) for k, v in obj.items()}
    return obj
