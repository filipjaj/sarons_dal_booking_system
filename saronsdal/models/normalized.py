"""Normalized domain models — the output of the normalization layer.

These models carry typed, parsed values along with confidence scores and review
flags.  They are the inputs to later phases (grouping, topology, optimization).

Design notes:
- Every model that was produced by parsing carries `parse_confidence` (0.0–1.0)
  and `review_flags: List[str]`.
- Fields that cannot be determined carry None, never a fake default.
- Coordinates are Optional and will be populated in a later phase.
- Directional allocation roles (primary vs support spot) are not assigned here;
  Phase 3 (topology) handles that.  The spot_count field on VehicleUnit already
  captures how many spots a booking needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import FrozenSet, List, Literal, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

VehicleType = Literal[
    "caravan",
    "motorhome",
    "camplet",
    "tent",
    "car",
    "unknown",
]

BookingSource = Literal["website", "reception", "other"]


# ---------------------------------------------------------------------------
# Dimension parsing result
# ---------------------------------------------------------------------------

@dataclass
class ParsedDimensions:
    """
    Result of parsing one or more raw dimension strings.

    All length/width values are in **metres**.
    `fortelt_width_m` is the lateral width extension from a fortelt (side awning);
    it does NOT add to body_length.
    """

    length_m: Optional[float]          # body length (vehicle or tent major axis)
    width_m: Optional[float]           # tent width (minor axis); None for caravans
    fortelt_width_m: Optional[float]   # parsed fortelt/awning lateral extension
    confidence: float                  # 0.0 – 1.0
    flags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Vehicle unit
# ---------------------------------------------------------------------------

@dataclass
class VehicleUnit:
    """
    Normalized description of the camping equipment a booking brings.

    Width rules (all in metres):
        - base width for caravan/motorhome = config.caravan_base_width_m (2.75)
        - if has_fortelt: total_width_m = body_width_m + fortelt_width_m
        - required_spot_count derived from total_width_m:
            (0, 3]  → 2 spots
            (3, 6]  → 3 spots
            (6, 9]  → 4 spots
        - for tents: width = min(dim1, dim2), length = max(dim1, dim2)
    """

    vehicle_type: VehicleType
    spec_size_hint: Optional[str]       # "large" | "small" — from spec string, if given

    body_length_m: Optional[float]      # vehicle/tent body length
    body_width_m: Optional[float]       # 2.75 for caravan/motorhome; tent minor axis
    fortelt_width_m: float              # 0.0 when absent
    total_width_m: Optional[float]      # body_width_m + fortelt_width_m
    required_spot_count: Optional[int]  # number of 3m-wide spots needed; None = unknown

    has_fortelt: bool
    has_markise: bool                   # awning (markise) — note only, no width model yet
    registration: Optional[str]         # cleaned registration number; None if not applicable

    parse_confidence: float
    review_flags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Spot request (parsed preferences)
# ---------------------------------------------------------------------------

@dataclass
class SectionRow:
    """
    A section + subsection-row preference extracted from booking text.

    "Vårdalen D" → SectionRow(section="Vårdalen", row="D")

    Kept as a soft preference signal — never a hard constraint.
    The optimizer uses it to prefer placing a booking in a specific row
    when the section is already fixed.
    """
    section: str    # canonical section name (e.g. "Vårdalen")
    row: str        # single uppercase letter (e.g. "D")


@dataclass
class SpotRequest:
    """
    Parsed placement preferences for a booking.

    Preferences are extracted but NOT resolved against actual spots or other
    bookings.  Resolution happens in later phases.
    """

    preferred_sections: List[str]           # normalized section names
    preferred_spot_ids: List[str]           # e.g. ["D25", "D26", "D27"]
    avoid_sections: List[str]
    amenity_flags: Set[str]                 # "flat", "near_toilet", "near_river", etc.
    raw_near_texts: List[str]               # raw extracted near-mention fragments

    parse_confidence: float
    review_flags: List[str] = field(default_factory=list)
    preferred_section_rows: List[SectionRow] = field(default_factory=list)
    # Soft row preference, e.g. SectionRow("Vårdalen", "D") from "Vårdalen D".
    # Never a hard constraint; Phase 3 uses it as a placement hint.


# ---------------------------------------------------------------------------
# Raw group signals (unresolved)
# ---------------------------------------------------------------------------

@dataclass
class RawGroupSignals:
    """
    Unresolved grouping signals extracted from a booking's text fields.

    These are raw strings only — no name matching, no graph construction.
    Group resolution happens in Phase 2.
    """

    organization: Optional[str]         # from Company field, encoding-repaired
    group_field: Optional[str]          # from the 2025-style "gruppe" column, if present
    near_text_fragments: List[str]      # raw text after "bo nærme", "bo med", etc.
    is_org_private: bool                # True if org is "privat" / empty / irrelevant


# ---------------------------------------------------------------------------
# Booking (fully normalized)
# ---------------------------------------------------------------------------

@dataclass
class Booking:
    """
    A fully normalized campsite booking.

    This is the primary output of Phase 1.  All fields are typed and validated.
    Later phases attach group links, allocation units, and assignment results.
    """

    # ------------------------------------------------------------------ identity
    booking_no: str
    booking_source: BookingSource

    # ------------------------------------------------------------------ dates
    check_in: Optional[date]
    check_out: Optional[date]

    # ------------------------------------------------------------------ person
    first_name: str
    last_name: str
    full_name: str
    num_guests: int                     # 0 when unparseable (flagged)
    language: str
    is_confirmed: bool

    # ------------------------------------------------------------------ equipment
    vehicle: VehicleUnit

    # ------------------------------------------------------------------ preferences
    request: SpotRequest

    # ------------------------------------------------------------------ group signals (unresolved)
    group_signals: RawGroupSignals

    # ------------------------------------------------------------------ quality
    # Aggregate confidence across vehicle + preference parsing.
    data_confidence: float

    # Booking-level review flags (union of all sub-model flags plus booking-level flags).
    review_flags: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------ raw text (preserved for Phase 2.5+)
    # Original free-text fields kept verbatim (ftfy-repaired) so that downstream
    # enrichment phases can scan full text, not just extracted fragments.
    raw_guest_message: str = ""
    raw_comment: str = ""
    raw_location_wish: str = ""
    city: str = ""                  # booking city (ftfy-repaired); "" when absent

    @property
    def needs_campsite(self) -> bool:
        """
        True if this booking plausibly requires a campsite spot.

        Returns False only when the booking's equipment type is clearly indoor
        (Internatet / Campinghytte) and body_length_m is None/0.
        Low-confidence Resepsjonen bookings are NOT excluded — they are flagged
        for manual review instead.
        """
        if self.vehicle.vehicle_type == "unknown" and self.vehicle.body_length_m is None:
            # No equipment type or dimensions — likely an indoor booking
            # (Internatet / Campinghytte) that leaked through ingestion.
            return False
        return True

    @property
    def is_low_confidence(self) -> bool:
        return self.data_confidence < 0.5


# ---------------------------------------------------------------------------
# Spot
# ---------------------------------------------------------------------------

@dataclass
class Spot:
    """
    One physical camping spot, as loaded from the spot survey CSV.

    Each spot is exactly 3m wide.  Lengths vary per spot.
    Coordinates are populated in a later phase when provided.
    """

    spot_id: str            # e.g. "A1", "B21"
    section: str            # normalized section name, e.g. "Furulunden"
    row: str                # letter prefix, e.g. "A"
    position: int           # numeric suffix, e.g. 1

    length_m: float
    hilliness: int          # 0 = flat, higher = steeper

    # ------------------------------------------------------------------ topology flags
    is_end_of_row: bool     # hard break — no contiguous allocation may cross this spot
    is_not_spot: bool       # individual obstruction; this spot is not allocatable
    is_reserved: bool       # keep the flag; do not discard reserved spots silently

    # ------------------------------------------------------------------ equipment restriction flags
    no_motorhome: bool              # mapped from "no_bobil"
    no_caravan_nor_motorhome: bool  # mapped from "no_bobil_nor_caravan"

    # ------------------------------------------------------------------ fields with defaults
    width_m: float = 3.0    # fixed; every spot is 3m wide

    # ------------------------------------------------------------------ future use
    coordinates: Optional[Tuple[float, float]] = None  # (x, y) in metres; Phase 3+

    # ------------------------------------------------------------------ raw fields preserved for inspection
    length_norm: Optional[float] = None  # raw normalised length value from CSV

    @property
    def is_allocatable(self) -> bool:
        """True if this spot can be assigned to a booking."""
        return not self.is_not_spot and not self.is_reserved
