"""Spot loaders.

Two entry points:

load_spots(path)
    Raw CSV ingestion from the original spot-survey file.
    Handles semicolons, encoding issues, column-name aliases, and parses
    row/position from the spot_id string.  Used during Phase 1 normalization.

load_spots_normalized(path)
    JSON loader for the Phase 1 output (spots_normalized.json).
    Every field is already typed and named; no column resolution needed.
    Used by the Phase 3 allocation CLI.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import ftfy
import pandas as pd

from saronsdal.models.normalized import Spot
from saronsdal.spatial.topology_loader import normalise_spot_id

logger = logging.getLogger(__name__)

# Expected column names in the spot file.
_SPOT_COLUMN_ALIASES = {
    "area":                  ["area", "Area"],
    "spot_id":               ["spot_id", "Spot_id", "spot id", "Spot ID"],
    "length":                ["length", "Length"],
    "hillyness":             ["hillyness", "hilliness", "Hillyness"],
    "no_bobil_nor_caravan":  ["no_bobil_nor_caravan"],
    "end_row":               ["end_row"],
    "not_spot":              ["not_spot"],
    "no_bobil":              ["no_bobil"],
    "reserved":              ["reserved"],
    "length_norm":           ["length_norm"],
}


def _resolve_spot_columns(columns: List[str]) -> dict:
    cols_lower = {c.lower().strip(): c for c in columns}
    result = {}
    for field, aliases in _SPOT_COLUMN_ALIASES.items():
        result[field] = next(
            (cols_lower[a.lower()] for a in aliases if a.lower() in cols_lower),
            None,
        )
    return result


def _parse_bool_flag(val: str) -> bool:
    """
    Treat "1", "true", "yes", "ja" as True; everything else (including "") as False.
    """
    return str(val).strip().lower() in ("1", "true", "yes", "ja")


def _parse_float(val: str, default: float = 0.0) -> float:
    try:
        return float(str(val).replace(",", ".").strip())
    except (ValueError, AttributeError):
        return default


def _parse_int(val: str, default: int = 0) -> int:
    try:
        return int(float(str(val).replace(",", ".").strip()))
    except (ValueError, AttributeError):
        return default


def _parse_spot_id(spot_id: str) -> Tuple[str, int]:
    """
    Derive (row, position) from a spot ID string like "A12" or "B3".

    Returns:
        (row, position) e.g. ("A", 12)

    Raises:
        ValueError if the spot_id cannot be parsed.
    """
    match = re.match(r"^([A-Za-z]+)(\d+)$", spot_id.strip())
    if not match:
        raise ValueError(f"Cannot parse spot_id '{spot_id}': expected letters+digits")
    row = match.group(1).upper()
    position = int(match.group(2))
    return row, position


def load_spots(path: Path) -> List[Spot]:
    """
    Load the spot survey CSV and return a list of Spot objects.

    The file is semicolon-delimited.  Boolean flags default to False when the
    cell is empty.  Rows where spot_id is missing or cannot be parsed are
    skipped with a warning.

    Args:
        path: Path to spots.csv (or spots.csv.csv).

    Returns:
        List of Spot objects; order matches the CSV row order.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Spot file not found: {path}")

    # Spots file uses semicolons as separator.
    try:
        df = pd.read_csv(
            path,
            sep=";",
            dtype=str,
            encoding="utf-8-sig",
            keep_default_na=False,
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path,
            sep=";",
            dtype=str,
            encoding="latin-1",
            keep_default_na=False,
        )

    # Encoding repair on column names and values.
    df.columns = [ftfy.fix_text(str(c)).strip() for c in df.columns]
    for col in df.columns:
        df[col] = df[col].apply(lambda v: ftfy.fix_text(str(v)).strip() if v else "")

    col_map = _resolve_spot_columns(df.columns.tolist())

    if col_map.get("spot_id") is None:
        raise ValueError(
            f"Cannot find 'spot_id' column in {path.name}. "
            f"Columns: {df.columns.tolist()}"
        )
    if col_map.get("area") is None:
        raise ValueError(
            f"Cannot find 'area' column in {path.name}."
        )

    spots: List[Spot] = []
    skipped = 0

    for idx, row in df.iterrows():
        spot_id_raw = row.get(col_map["spot_id"], "").strip() if col_map["spot_id"] else ""
        area_raw = row.get(col_map["area"], "").strip() if col_map["area"] else ""

        if not spot_id_raw or not area_raw:
            logger.warning(
                "Spot row %d: missing spot_id or area — skipped (spot_id=%r, area=%r)",
                idx,
                spot_id_raw,
                area_raw,
            )
            skipped += 1
            continue

        try:
            row_letter, position = _parse_spot_id(spot_id_raw)
        except ValueError as exc:
            logger.warning("Spot row %d: %s — skipped", idx, exc)
            skipped += 1
            continue

        length_raw = row.get(col_map["length"], "") if col_map.get("length") else ""
        length_m = _parse_float(length_raw, default=0.0)
        if length_m == 0.0 and length_raw:
            logger.warning(
                "Spot %s/%s: could not parse length %r, using 0.0",
                area_raw,
                spot_id_raw,
                length_raw,
            )

        hilliness_raw = row.get(col_map["hillyness"], "0") if col_map.get("hillyness") else "0"
        hilliness = _parse_int(hilliness_raw, default=0)

        # Optional length_norm field.
        length_norm_raw = row.get(col_map["length_norm"], "") if col_map.get("length_norm") else ""
        length_norm: Optional[float] = None
        if length_norm_raw:
            try:
                length_norm = float(length_norm_raw.replace(",", "."))
            except ValueError:
                pass

        def _flag(field_name: str) -> bool:
            col = col_map.get(field_name)
            if col is None:
                return False
            return _parse_bool_flag(row.get(col, ""))

        spot = Spot(
            spot_id=normalise_spot_id(spot_id_raw),
            section=ftfy.fix_text(area_raw),
            row=row_letter,
            position=position,
            length_m=length_m,
            width_m=3.0,                             # fixed: every spot is 3m wide
            hilliness=hilliness,
            is_end_of_row=_flag("end_row"),
            is_not_spot=_flag("not_spot"),
            is_reserved=_flag("reserved"),
            no_motorhome=_flag("no_bobil"),
            no_caravan_nor_motorhome=_flag("no_bobil_nor_caravan"),
            coordinates=None,                        # populated in a later phase
            length_norm=length_norm,
        )
        spots.append(spot)

    logger.info(
        "Loaded %d spots from %s (%d skipped)",
        len(spots),
        path.name,
        skipped,
    )
    return spots


# ---------------------------------------------------------------------------
# Normalized JSON loader (Phase 3 input)
# ---------------------------------------------------------------------------

#: Fields that must be present in every entry of spots_normalized.json.
_REQUIRED_SPOT_FIELDS = (
    "spot_id",
    "section",
    "row",
    "position",
    "length_m",
    "hilliness",
    "is_end_of_row",
    "is_not_spot",
    "is_reserved",
    "no_motorhome",
    "no_caravan_nor_motorhome",
    "width_m",
)


def load_spots_normalized(path: Path) -> List[Spot]:
    """Load spots from a Phase 1 normalized JSON file (spots_normalized.json).

    Each entry in the JSON array maps directly to a Spot dataclass — no column
    resolution, no bool-flag parsing, no spot_id decomposition required.

    Entries with missing required fields are skipped with a warning rather than
    raising, so a partial file still yields usable results.

    Args:
        path: Path to spots_normalized.json.

    Returns:
        List of Spot objects in file order.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError:        if the file is not a JSON array.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Normalized spot file not found: {path}")

    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    if not isinstance(raw, list):
        raise ValueError(
            f"{path.name} must be a JSON array of spot objects, got {type(raw).__name__}"
        )

    spots: List[Spot] = []
    skipped = 0

    for i, d in enumerate(raw):
        missing = [f for f in _REQUIRED_SPOT_FIELDS if f not in d]
        if missing:
            logger.warning(
                "spots_normalized entry %d: missing fields %s — skipped",
                i,
                missing,
            )
            skipped += 1
            continue

        # coordinates is stored as [x, y] or null
        raw_coords = d.get("coordinates")
        coords: Optional[Tuple[float, float]] = None
        if raw_coords is not None:
            try:
                coords = (float(raw_coords[0]), float(raw_coords[1]))
            except (TypeError, IndexError, ValueError):
                logger.warning(
                    "spots_normalized entry %d (%s): unparseable coordinates %r — set to None",
                    i,
                    d.get("spot_id", "?"),
                    raw_coords,
                )

        try:
            spot = Spot(
                spot_id=str(d["spot_id"]),
                section=str(d["section"]),
                row=str(d["row"]),
                position=int(d["position"]),
                length_m=float(d["length_m"]),
                hilliness=int(d["hilliness"]),
                is_end_of_row=bool(d["is_end_of_row"]),
                is_not_spot=bool(d["is_not_spot"]),
                is_reserved=bool(d["is_reserved"]),
                no_motorhome=bool(d["no_motorhome"]),
                no_caravan_nor_motorhome=bool(d["no_caravan_nor_motorhome"]),
                width_m=float(d.get("width_m", 3.0)),
                coordinates=coords,
                length_norm=float(d["length_norm"]) if d.get("length_norm") is not None else None,
            )
            spots.append(spot)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "spots_normalized entry %d (%s): type error — %s — skipped",
                i,
                d.get("spot_id", "?"),
                exc,
            )
            skipped += 1

    logger.info(
        "Loaded %d normalized spots from %s (%d skipped)",
        len(spots),
        path.name,
        skipped,
    )
    return spots
