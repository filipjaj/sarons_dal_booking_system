"""Vehicle classification and VehicleUnit construction.

Classification priority (highest to lowest confidence):
  1. EXTRAS specification row  (e.g. "Campingvogn" → caravan)
  2. Registration number pattern
  3. Keywords in length/regnr text
  4. Unknown

Width and required_spot_count are derived from the classified type, parsed
dimensions, and config values.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from saronsdal.models.normalized import ParsedDimensions, VehicleType, VehicleUnit
from saronsdal.models.raw import RawBooking
from saronsdal.normalization.length_parser import parse_dimensions

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "equipment.yaml"


@lru_cache(maxsize=1)
def _load_config() -> Dict[str, Any]:
    with open(_CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bool_field(raw: str) -> bool:
    """Normalise a Ja/Nei/yes/no/1/0 string to bool."""
    return raw.strip().lower() in ("ja", "yes", "1", "true")


_PLATE_PATTERN = re.compile(
    r"^[A-ZÆØÅ]{1,3}\s?\d{2,5}$",  # Norwegian and older/foreign plates
    re.IGNORECASE,
)


def _looks_like_plate(text: str) -> bool:
    """Return True if `text` looks like a Norwegian/European vehicle plate."""
    return bool(_PLATE_PATTERN.match(text.strip()))


def _classify_from_specs(spec_values: list, cfg: Dict) -> tuple[Optional[VehicleType], Optional[str], float]:
    """
    Attempt to classify vehicle type from EXTRAS specification values.

    Returns:
        (vehicle_type or None, spec_size_hint or None, confidence)
    """
    spec_map: Dict[str, str] = cfg.get("specification_map", {})
    size_hints: Dict[str, str] = cfg.get("specification_size_hint", {})
    indoor: list = cfg.get("indoor_specification_values", [])
    unknown_specs: list = cfg.get("unknown_specification_values", [])

    vehicle_type: Optional[VehicleType] = None
    size_hint: Optional[str] = None
    confidence = 0.0

    for sv in spec_values:
        sv_clean = sv.strip()

        if sv_clean in indoor:
            # Indoor accommodation booking — not a vehicle type.
            return "unknown", None, 0.3

        if sv_clean in unknown_specs:
            return "unknown", None, 0.2

        if sv_clean in spec_map:
            mapped = spec_map[sv_clean]
            vehicle_type = mapped  # type: ignore[assignment]
            size_hint = size_hints.get(sv_clean)
            confidence = 0.95
            break  # First matching EXTRAS spec wins.

    return vehicle_type, size_hint, confidence


def _classify_from_regnr(regnr: str, raw_length: str) -> tuple[Optional[VehicleType], float]:
    """
    Fallback classification from registration number and length text.

    Returns:
        (vehicle_type or None, confidence)
    """
    regnr_l = regnr.lower().strip()
    len_l = raw_length.lower().strip()

    # Explicit "telt" keyword → tent.
    if "telt" in regnr_l or "telt" in len_l:
        return "tent", 0.75

    # Camp-let.
    if re.search(r"\bcamp.?let\b", regnr_l + " " + len_l, re.IGNORECASE):
        return "camplet", 0.80

    # Plate present → caravan or motorhome (indistinguishable without more info).
    if regnr_l and regnr_l not in ("0", "", "-") and _looks_like_plate(regnr):
        return "caravan", 0.60   # Default to caravan; low confidence.

    # "Personbil" / "bil" → car (with possibly a tent).
    if regnr_l in ("personbil", "bil"):
        return "car", 0.55

    return None, 0.0


def _derive_spot_count(
    total_width_m: Optional[float],
    has_fortelt: bool,
    vehicle_type: Optional[VehicleType],
    cfg: Dict,
) -> Optional[int]:
    """
    Derive required spot count from total width and config thresholds.

    If total_width_m is None:
      - caravan/motorhome with fortelt → 3 (hardcoded business rule)
      - caravan/motorhome without fortelt → 2
      - tent → None (unknown until width is established)
      - other → None
    """
    narrow = cfg["spot_geometry"]["narrow_threshold_m"]   # 3.0
    medium = cfg["spot_geometry"]["medium_threshold_m"]   # 6.0
    wide   = cfg["spot_geometry"]["wide_threshold_m"]     # 9.0

    if total_width_m is not None:
        if total_width_m <= narrow:
            return 2
        if total_width_m <= medium:
            return 3
        if total_width_m <= wide:
            return 4
        return None  # Over 9m wide — needs manual review.

    # Width unknown.
    if vehicle_type in ("caravan", "motorhome"):
        return 3 if has_fortelt else 2
    return None  # Tent or unknown without dimensions.


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_vehicle(booking: RawBooking) -> VehicleUnit:
    """
    Produce a VehicleUnit from a merged RawBooking.

    Uses (in priority order):
      1. EXTRAS specification rows
      2. Registration number + length text heuristics
      3. Config defaults for width/spot count
    """
    cfg = _load_config()
    flags: list = []

    has_fortelt = _bool_field(booking.has_fortelt)
    has_markise = _bool_field(booking.has_markise)

    # --- Step 1: Parse dimensions. ------------------------------------------
    dims: ParsedDimensions = parse_dimensions(
        booking.raw_length,
        booking.raw_width,
    )
    flags.extend(dims.flags)

    # --- Step 2: Classify vehicle type. -------------------------------------
    extras = booking.extra_specification_values()
    vtype, size_hint, type_confidence = _classify_from_specs(extras, cfg)

    if vtype is None:
        vtype, type_confidence = _classify_from_regnr(booking.regnr, booking.raw_length)

    if vtype is None:
        vtype = "unknown"
        type_confidence = 0.0
        flags.append("low_confidence_vehicle_type")

    # Multiple conflicting EXTRAS specs is a data problem.
    unique_vehicle_specs = [
        s for s in extras
        if s in cfg.get("specification_map", {})
    ]
    if len(unique_vehicle_specs) > 1:
        flags.append("conflicting_specifications")
        type_confidence -= 0.30

    # Flag bookings from reception with no dimension data.
    is_reception = booking.booking_source.lower() == "resepsjonen"
    if is_reception and dims.length_m is None:
        flags.append("reception_placeholder_data")

    # --- Step 3: Derive body width. -----------------------------------------
    veh_cfg = cfg["vehicle"]
    base_width = veh_cfg["caravan_base_width_m"]  # 2.75 by default

    if vtype == "tent":
        # For tents: body_width_m comes from the parsed width dimension.
        body_width_m: Optional[float] = dims.width_m
        if body_width_m is None and dims.length_m is not None:
            # Single-dimension tent — cannot determine which axis is width.
            flags.append("tent_single_dimension")
            body_width_m = None
    elif vtype in ("caravan", "motorhome", "camplet"):
        body_width_m = base_width
    else:
        body_width_m = None  # Unknown / car — no width assumption.

    # --- Step 4: Fortelt width. ---------------------------------------------
    fw: float = 0.0
    if dims.fortelt_width_m is not None:
        fw = dims.fortelt_width_m
    elif has_fortelt:
        fw = veh_cfg["default_fortelt_width_m"]

    # --- Step 4b: Explicit total width from the new 2026 Sirvoy column. -----
    # "Total bredde på din enhet i meter (inkludert fortelt hvis du har det)"
    # This is the most authoritative width source when present and valid.
    # Plausible range: 1.0 m (narrow tent) – 9.0 m (4-spot unit).
    _MIN_WIDTH_M = 1.0
    _MAX_WIDTH_M = 9.0
    explicit_total_width_m: Optional[float] = None

    raw_tw = (booking.raw_total_width or "").strip()
    if raw_tw and raw_tw not in ("0", "0.0", "0,0", "-"):
        tw_dims = parse_dimensions(raw_tw, "")
        if tw_dims.length_m is not None:
            if _MIN_WIDTH_M <= tw_dims.length_m <= _MAX_WIDTH_M:
                explicit_total_width_m = tw_dims.length_m
                flags.append("total_width_from_explicit_field")
                # Propagate parse-quality flags so reviewers know if the width
                # value itself needed unit inference.
                for f in tw_dims.flags:
                    if f in ("assumed_mm", "assumed_cm", "assumed_cm_typo",
                             "approximate", "complex_dimension_string"):
                        flags.append(f"explicit_width_{f}")
            else:
                flags.append("width_out_of_bounds")
                # Explicit value present but invalid — treat as if absent;
                # fall through to the inferred path below.

    # --- Step 5: Total width. -----------------------------------------------
    total_width_m: Optional[float] = None
    if explicit_total_width_m is not None:
        # Customer-provided total (body + fortelt) takes precedence.
        total_width_m = explicit_total_width_m
    elif body_width_m is not None:
        # Inferred from body width + any fortelt extension.
        total_width_m = body_width_m + fw

    # --- Step 6: Required spot count. ---------------------------------------
    req_spots = _derive_spot_count(total_width_m, has_fortelt, vtype, cfg)
    if req_spots is None and vtype not in ("car", "unknown"):
        flags.append("unknown_spot_count")

    # --- Step 7: Aggregate confidence. -------------------------------------
    # Blend type confidence with dimension parse confidence.
    dim_weight = 0.4
    type_weight = 0.6
    if dims.confidence == 0.0 and type_confidence > 0.0:
        # If we have no dimension data but know the type, we can still flag but proceed.
        overall_conf = type_confidence * type_weight
    elif type_confidence == 0.0:
        overall_conf = dims.confidence * dim_weight
    else:
        overall_conf = (dims.confidence * dim_weight) + (type_confidence * type_weight)

    if "width_out_of_bounds" in flags:
        overall_conf = max(0.0, overall_conf - 0.15)

    # Clean registration string.
    regnr_clean: Optional[str] = None
    regnr_raw = booking.regnr.strip()
    if regnr_raw and regnr_raw not in ("0", "-", ""):
        regnr_clean = regnr_raw

    return VehicleUnit(
        vehicle_type=vtype,
        spec_size_hint=size_hint,
        body_length_m=dims.length_m,
        body_width_m=body_width_m,
        fortelt_width_m=fw,
        total_width_m=total_width_m,
        required_spot_count=req_spots,
        has_fortelt=has_fortelt,
        has_markise=has_markise,
        registration=regnr_clean,
        parse_confidence=round(max(0.0, min(1.0, overall_conf)), 2),
        review_flags=flags,
    )
