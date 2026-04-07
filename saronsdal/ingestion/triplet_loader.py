"""Triplet loader — parses sirvoy_room_ids.csv into Triplet objects.

The 'Rom' column of sirvoy_room_ids.csv is the authoritative list of valid
allocation units.  Each camping row defines exactly one Triplet.  Non-camping
entries (Internatet, Campinghytte, Sovesal, Helsebua, etc.) are silently
skipped.

Supported 'Rom' formats
-----------------------
Standard triplet with length annotation::
    "Furulunden A16-18 (5m)"
    "Fjellterrassen D28-30 (7,8m)"

Standard triplet without annotation::
    "Bedehuset A01-03"
    "Furulunden E12-14"

Repeated row letter (Sirvoy variant)::
    "Bibelskolen A01-A03"
    "Elvebredden B21-B23 (6,9m)"

Space inside range (data quirk)::
    "Vårdalen A01- 03 (15,7m)"

Non-standard (parsed, flagged, not allocatable)::
    "Bibelskolen A10-11 ( bobil)"   → 2 spots → review_flag "non_triplet_spot_count_2"
    "Bibelskolen D22"               → 1 spot  → review_flag "non_triplet_spot_count_1"
    "Elvebredden C37-40 TELT (5m)"  → 4 spots → review_flag "non_triplet_spot_count_4"

Non-camping (skipped silently)::
    "Campinghytte 1 seng1"
    "Internatet 101 (seng1)"
    "Sovesal gutter seng01"
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ftfy
import pandas as pd

from saronsdal.models.normalized import Spot
from saronsdal.models.triplet import Triplet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Camping section names — non-camping entries are silently skipped
# ---------------------------------------------------------------------------

#: Section names that correspond to outdoor camping spots (caravan / motorhome
#: / tent).  Entries whose section name is not in this set are treated as
#: non-camping and excluded from the returned triplet list.
CAMPING_SECTIONS: frozenset[str] = frozenset({
    "Bedehuset",
    "Bibelskolen",
    "Egelandsletta",
    "Elvebredden",
    "Fjellterrassen",
    "Furulunden",
    "Furutoppen",
    "Vårdalen",
})


# ---------------------------------------------------------------------------
# Regular expressions
# ---------------------------------------------------------------------------

# Matches range rooms like "Furulunden A16-18 (5m)" or "Bibelskolen A01-A03".
# Handles:
#   - optional space around the hyphen ("A01- 03")
#   - optional repeated row letter before end number ("A01-A03")
#   - optional trailing annotation "(7,8m)", "TELT (5m)", etc.
_RANGE_RE = re.compile(
    r"^(?P<section>\S+)\s+"
    r"(?P<row>[A-Z])(?P<start>\d+)\s*-\s*(?P=row)?(?P<end>\d+)"
    r"(?:\s.*)?$"
)

# Matches single-spot entries like "Bibelskolen D22" or "Elvebredden C04".
_SINGLE_RE = re.compile(
    r"^(?P<section>\S+)\s+(?P<row>[A-Z])(?P<num>\d+)(?:\s.*)?$"
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_triplets(
    rooms_path: Path,
    spot_lookup: Dict[Tuple[str, str], Spot],
) -> List[Triplet]:
    """Parse sirvoy_room_ids.csv into a list of Triplet objects.

    Args:
        rooms_path:  Path to sirvoy_room_ids.csv (semicolon-delimited).
        spot_lookup: Mapping of ``(section, spot_id) → Spot`` built from
                     spots.csv data.  Used to resolve each triplet's anchor
                     spot for length and restriction flags.

    Returns:
        List of Triplet objects for all camping 'Rom' rows.  Includes both
        allocatable (standard 3-spot, anchor resolved) and non-allocatable
        (flagged) entries.  Use ``Triplet.is_allocatable`` to filter.

    Raises:
        FileNotFoundError: if rooms_path does not exist.
        ValueError:        if the 'Rom' column cannot be found.
    """
    rooms_path = Path(rooms_path)
    if not rooms_path.exists():
        raise FileNotFoundError(f"Sirvoy rooms file not found: {rooms_path}")

    try:
        df = pd.read_csv(
            rooms_path,
            sep=";",
            dtype=str,
            encoding="utf-8-sig",
            keep_default_na=False,
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            rooms_path,
            sep=";",
            dtype=str,
            encoding="latin-1",
            keep_default_na=False,
        )

    # Encoding repair on column names and cell values
    df.columns = [ftfy.fix_text(str(c)).strip() for c in df.columns]
    for col in df.columns:
        df[col] = df[col].apply(lambda v: ftfy.fix_text(str(v)).strip() if v else "")

    # Locate the 'Rom' column (case-insensitive)
    rom_col = next(
        (c for c in df.columns if c.strip().lower() == "rom"),
        None,
    )
    if rom_col is None:
        raise ValueError(
            f"Cannot find 'Rom' column in {rooms_path.name}. "
            f"Columns found: {df.columns.tolist()}"
        )

    triplets: List[Triplet] = []
    skipped_non_camping = 0

    for _, row_data in df.iterrows():
        room_id_raw = row_data.get(rom_col, "").strip()
        if not room_id_raw:
            continue

        triplet = _parse_room_id(room_id_raw, spot_lookup)
        if triplet is None:
            skipped_non_camping += 1
            continue
        triplets.append(triplet)

    allocatable = sum(1 for t in triplets if t.is_allocatable)
    flagged = len(triplets) - allocatable
    logger.info(
        "Loaded %d triplets from %s: %d allocatable, %d flagged, "
        "%d non-camping skipped",
        len(triplets), rooms_path.name, allocatable, flagged, skipped_non_camping,
    )
    return triplets


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_room_id(
    room_id: str,
    spot_lookup: Dict[Tuple[str, str], Spot],
) -> Optional[Triplet]:
    """Parse one 'Rom' string into a Triplet.

    Returns ``None`` for non-camping entries (not in CAMPING_SECTIONS).
    Returns a Triplet with ``review_flags`` for non-standard entries.
    """
    room_id = room_id.strip()

    # Try range pattern first: "Furulunden A16-18" / "Bibelskolen A01-A03"
    m = _RANGE_RE.match(room_id)
    if m:
        section = m.group("section")
        if section not in CAMPING_SECTIONS:
            return None
        row = m.group("row")
        start = int(m.group("start"))
        end = int(m.group("end"))
        spot_ids = [f"{row}{i}" for i in range(start, end + 1)]
        return _build_triplet(room_id, section, row, spot_ids, spot_lookup)

    # Try single-spot pattern: "Bibelskolen D22"
    m = _SINGLE_RE.match(room_id)
    if m:
        section = m.group("section")
        if section not in CAMPING_SECTIONS:
            return None
        row = m.group("row")
        num = int(m.group("num"))
        spot_ids = [f"{row}{num}"]
        return _build_triplet(room_id, section, row, spot_ids, spot_lookup)

    # Cannot parse — treat as non-camping
    return None


def _build_triplet(
    room_id: str,
    section: str,
    row: str,
    spot_ids: List[str],
    spot_lookup: Dict[Tuple[str, str], Spot],
) -> Triplet:
    """Construct a Triplet and populate review_flags for non-standard entries."""
    flags: List[str] = []

    if len(spot_ids) != 3:
        flags.append(f"non_triplet_spot_count_{len(spot_ids)}")

    first_spot = spot_lookup.get((section, spot_ids[0]))
    if first_spot is None:
        flags.append("first_spot_missing")
        logger.debug(
            "Triplet %r: anchor spot (%s, %s) not found in spot data",
            room_id, section, spot_ids[0],
        )

    return Triplet(
        room_id=room_id,
        section=section,
        row=row,
        spot_ids=spot_ids,
        first_spot=first_spot,
        review_flags=flags,
    )
