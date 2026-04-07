"""Extract placement preferences and raw group signals from booking text fields.

All extraction is raw — no name resolution, no graph construction.
Output goes into SpotRequest and RawGroupSignals.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import ftfy
import yaml

from saronsdal.models.normalized import RawGroupSignals, SectionRow, SpotRequest
from saronsdal.models.raw import RawBooking

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_SECTIONS_CONFIG_PATH = Path(__file__).parent.parent / "config" / "sections.yaml"
_GROUP_CONFIG_PATH = Path(__file__).parent.parent / "config" / "group_aliases.yaml"


@lru_cache(maxsize=1)
def _load_sections_cfg() -> Dict:
    with open(_SECTIONS_CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def _load_group_cfg() -> Dict:
    with open(_GROUP_CONFIG_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Section alias lookup
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_alias_to_section() -> Dict[str, str]:
    """Build a lowercase alias → canonical section name lookup."""
    cfg = _load_sections_cfg()
    mapping: Dict[str, str] = {}
    for key, sec in cfg.get("sections", {}).items():
        canonical = sec["canonical"]
        for alias in sec.get("aliases", []):
            mapping[alias.lower()] = canonical
    return mapping


def _is_section_name(text: str) -> bool:
    """Return True if text looks like a known section name."""
    cfg = _load_sections_cfg()
    known = {s.lower() for s in cfg.get("section_name_set", [])}
    return text.strip().lower() in known


# ---------------------------------------------------------------------------
# Spot ID extraction
# ---------------------------------------------------------------------------

_SPOT_ID_RE = re.compile(
    r"\b([A-Z]{1,2}\d{1,3})\b",
    re.IGNORECASE,
)

_SPOT_RANGE_RE = re.compile(
    r"\b([A-Z]{1,2})(\d{1,3})\s*[-–]\s*([A-Z]{0,2})(\d{1,3})\b",
    re.IGNORECASE,
)

_SPOT_RANGE_TIL_RE = re.compile(
    r"\b([A-Z]{1,2})(\d{1,3})\s+(?:til|to)\s+([A-Z]{0,2})(\d{1,3})\b",
    re.IGNORECASE,
)


def _extract_spot_ids(text: str) -> List[str]:
    """
    Extract explicit spot IDs from free text, including ranges.

    Examples:
        "D25-D27"  → ["D25", "D26", "D27"]
        "A1 til 5" → ["A1", "A2", "A3", "A4", "A5"]
        "B12"      → ["B12"]
    """
    ids: List[str] = []
    text_upper = text.upper()

    def _expand_range(prefix1: str, start: int, prefix2: str, end: int) -> List[str]:
        p = (prefix2 if prefix2 else prefix1).upper()
        return [f"{p}{i}" for i in range(start, end + 1)]

    # Range with dash or en-dash.
    for m in _SPOT_RANGE_RE.finditer(text_upper):
        p1, n1, p2, n2 = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
        ids.extend(_expand_range(p1, n1, p2, n2))

    # Range with "til".
    for m in _SPOT_RANGE_TIL_RE.finditer(text_upper):
        p1, n1, p2, n2 = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
        ids.extend(_expand_range(p1, n1, p2, n2))

    # Single IDs (only if not already covered by a range match).
    range_positions = set()
    for m in list(_SPOT_RANGE_RE.finditer(text_upper)) + list(
        _SPOT_RANGE_TIL_RE.finditer(text_upper)
    ):
        range_positions.update(range(m.start(), m.end()))

    for m in _SPOT_ID_RE.finditer(text_upper):
        if not any(p in range_positions for p in range(m.start(), m.end())):
            candidate = m.group(0).upper()
            # Filter out false positives that look like abbreviations in context.
            # Require the match to look like "letter(s) + digit(s)" with 1–3 digits.
            if re.match(r"^[A-Z]{1,2}\d{1,3}$", candidate):
                ids.append(candidate)

    # Deduplicate while preserving order.
    seen: set = set()
    result = []
    for sid in ids:
        if sid not in seen:
            seen.add(sid)
            result.append(sid)
    return result


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------

def _extract_sections(text: str) -> List[str]:
    """Return canonical section names mentioned in text."""
    lookup = _get_alias_to_section()
    found = []
    text_lower = text.lower()
    # Sort aliases by length descending so longer matches win.
    for alias in sorted(lookup.keys(), key=len, reverse=True):
        if alias in text_lower:
            canonical = lookup[alias]
            if canonical not in found:
                found.append(canonical)
    return found


# ---------------------------------------------------------------------------
# Section + row extraction
# ---------------------------------------------------------------------------

# Matches a single letter (not followed by a digit) after optional whitespace.
# "Vårdalen D"  → row "D"  (D not followed by digit → subsection row)
# "Elvebredden D25" → no match (D followed by digit 2 → spot ID, not row)
_SECTION_ROW_SUFFIX_RE = re.compile(r"\s+([A-Za-z])(?!\d)")


def _extract_section_rows(text: str) -> List[SectionRow]:
    """
    Find (section, row) pairs where a known section alias is immediately
    followed by a single letter that is NOT followed by a digit.

    Examples:
        "Vårdalen D"       → [SectionRow("Vårdalen", "D")]
        "Furulunden B"     → [SectionRow("Furulunden", "B")]
        "Elvebredden D25"  → []  (D25 is a spot ID, not a row)
        "Fjellterrassen"   → []  (no row letter present)
        "Furulunden B og Furulunden C" → two SectionRow objects
    """
    lookup = _get_alias_to_section()
    text_fixed = ftfy.fix_text(text)
    text_lower = text_fixed.lower()

    found: List[SectionRow] = []
    seen: Set[tuple] = set()

    for alias in sorted(lookup.keys(), key=len, reverse=True):
        alias_len = len(alias)
        start = 0
        while True:
            pos = text_lower.find(alias, start)
            if pos == -1:
                break
            suffix = text_fixed[pos + alias_len:]
            m = _SECTION_ROW_SUFFIX_RE.match(suffix)
            if m:
                row = m.group(1).upper()
                if "A" <= row <= "Z":
                    canonical = lookup[alias]
                    key = (canonical, row)
                    if key not in seen:
                        seen.add(key)
                        found.append(SectionRow(section=canonical, row=row))
            start = pos + 1

    return found


# ---------------------------------------------------------------------------
# "Near" / "together with" text extraction
# ---------------------------------------------------------------------------

_NEAR_PATTERNS = [
    re.compile(
        r"(?:bo\s+)?nær(?:heten\s+av|mest\s+mulig|e)?[:\s]+"
        r"([\w\sÆØÅæøå,.&'()/\-]+?)(?:\.|,|\n|$|\)|ønsker|takk|reiser)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:bo\s+)?(?:ved\s+siden\s+av|ved\s+siden)[:\s]+"
        r"([\w\sÆØÅæøå,.&'()/\-]+?)(?:\.|,|\n|$|\)|ønsker|takk)",
        re.IGNORECASE,
    ),
    re.compile(
        r"sammen\s+med[:\s]+"
        r"([\w\sÆØÅæøå,.&'()/\-]+?)(?:\.|,|\n|$|\)|ønsker|takk|gleder)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:bo\s+)?(?:i\s+)?nærheten\s+av[:\s]+"
        r"([\w\sÆØÅæøå,.&'()/\-]+?)(?:\.|,|\n|$|\)|ønsker|takk)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:ønsker\s+å\s+)?(?:stå|ligge|bo)\s+(?:med|nær|ved)[:\s]+"
        r"([\w\sÆØÅæøå,.&'()/\-]+?)(?:\.|,|\n|$|\)|ønsker|takk)",
        re.IGNORECASE,
    ),
]

# Words that are too generic to constitute a useful name fragment.
_GENERIC_STOPWORDS = {
    "gjengen", "familien", "fam", "de", "andre", "oss", "disse",
    "naboene", "alle", "vi", "dem", "oss selv", "hverandre",
    "gruppen", "gruppa",
}


def _extract_near_fragments(text: str) -> List[str]:
    """Extract raw text fragments following near-mention patterns."""
    fragments = []
    for pattern in _NEAR_PATTERNS:
        for m in pattern.finditer(text):
            frag = m.group(1).strip().rstrip(",.")
            # Require at least one capitalised word (probable name or org).
            if frag and re.search(r"[A-ZÆØÅ]", frag):
                # Filter out pure stopwords.
                lower_frag = frag.lower().strip()
                if lower_frag not in _GENERIC_STOPWORDS and len(frag) >= 4:
                    fragments.append(frag)
    # Deduplicate.
    seen: set = set()
    return [f for f in fragments if not (f.lower() in seen or seen.add(f.lower()))]


# ---------------------------------------------------------------------------
# Amenity flag extraction
# ---------------------------------------------------------------------------

_AMENITY_KEYWORDS: Dict[str, str] = {
    "flat":         r"\b(?:flat|jevn|slette)\b",
    "near_toilet":  r"\b(?:toalett|wc|sanitær)\b",
    "near_river":   r"\belv(?:en|a|e)?\b",
    "near_forest":  r"\bskog(?:en|a|e)?\b",
    "near_hall":    r"\b(?:møtesal|hall|arena|sal)\b",
}


def _extract_amenity_flags(text: str) -> Set[str]:
    flags: Set[str] = set()
    for flag, pattern in _AMENITY_KEYWORDS.items():
        if re.search(pattern, text, re.IGNORECASE):
            flags.add(flag)
    return flags


# ---------------------------------------------------------------------------
# Avoid-area extraction
# ---------------------------------------------------------------------------

_AVOID_PATTERNS = [
    re.compile(
        r"(?:ønsker\s+ikke|ikke\s+)?(?:å\s+bo|bo)\s+på\s+"
        r"([\w\sÆØÅæøå]+?)(?:\.|,|\n|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:ikke|unngå)\s+([\w\sÆØÅæøå]+?)(?:\.|,|\n|$)",
        re.IGNORECASE,
    ),
]


def _extract_avoid_sections(text: str) -> List[str]:
    avoid: List[str] = []
    for pattern in _AVOID_PATTERNS:
        for m in pattern.finditer(text):
            candidate = m.group(1).strip()
            sections = _extract_sections(candidate)
            avoid.extend(s for s in sections if s not in avoid)
    return avoid


# ---------------------------------------------------------------------------
# Organization / group field
# ---------------------------------------------------------------------------

def _clean_org(raw: str) -> Optional[str]:
    """Normalise the company/org field; return None if it's a private label."""
    cfg = _load_group_cfg()
    private = {p.lower() for p in cfg.get("private_labels", [])}
    cleaned = ftfy.fix_text(raw.strip())
    if not cleaned or cleaned.lower() in private:
        return None
    return cleaned


def _is_org_private(raw: str) -> bool:
    cfg = _load_group_cfg()
    private = {p.lower() for p in cfg.get("private_labels", [])}
    return not raw.strip() or raw.strip().lower() in private


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_spot_request(booking: RawBooking) -> SpotRequest:
    """
    Parse all placement preference fields and return a SpotRequest.

    Sources scanned (in priority order): raw_location_wish, guest_message, comment.
    """
    # Combine all text sources.
    combined = ftfy.fix_text(
        " ".join([
            booking.raw_location_wish,
            booking.guest_message,
            booking.comment,
        ])
    )
    flags: list = []

    preferred_sections = _extract_sections(combined)
    preferred_section_rows = _extract_section_rows(combined)
    preferred_spots = _extract_spot_ids(
        # Check location wish first for explicit spot IDs.
        booking.raw_location_wish + " " + booking.guest_message + " " + booking.comment
    )
    avoid_sections = _extract_avoid_sections(combined)
    amenity_flags = _extract_amenity_flags(combined)
    near_texts = _extract_near_fragments(combined)

    # Low confidence if nothing useful was extracted.
    confidence = 1.0
    if not preferred_sections and not preferred_spots and not near_texts:
        confidence = 0.5
        flags.append("no_placement_preference")

    return SpotRequest(
        preferred_sections=preferred_sections,
        preferred_section_rows=preferred_section_rows,
        preferred_spot_ids=preferred_spots,
        avoid_sections=avoid_sections,
        amenity_flags=amenity_flags,
        raw_near_texts=near_texts,
        parse_confidence=round(confidence, 2),
        review_flags=flags,
    )


def extract_group_signals(booking: RawBooking) -> RawGroupSignals:
    """
    Extract unresolved grouping signals from a booking.

    Does NOT resolve names, match against other bookings, or build any graph.
    """
    # Organization / company field.
    org = _clean_org(booking.company)
    is_private = _is_org_private(booking.company)

    # 2025-style group field: the RawBooking has no dedicated group_field slot
    # because the basic CSV schema does not always have it.  In future schema
    # versions it can be added to RawBooking; for now we check the raw_location_wish
    # and company fields together.
    #
    # When the group field IS present in the CSV, the merger should attach it
    # to a future `group_field` attribute on RawBooking.  For now we read it
    # from `raw_location_wish` only when it does not look like a section name.
    group_field_val: Optional[str] = None
    loc_wish = ftfy.fix_text(booking.raw_location_wish.strip())
    if loc_wish and not _is_section_name(loc_wish):
        # Could be a group name rather than a location wish.
        # Only use as group signal if it doesn't look like a section + spot combo.
        if not re.match(r"^[A-ZÆØÅ][a-zæøå]+\s+[A-Z]\d+", loc_wish):
            # Heuristic: if the wish text is short and doesn't contain section keywords
            # or spot IDs, treat it as a potential group signal.
            has_section = bool(_extract_sections(loc_wish))
            has_spots = bool(_extract_spot_ids(loc_wish))
            if not has_section and not has_spots and len(loc_wish) < 60:
                group_field_val = loc_wish if len(loc_wish) > 3 else None

    # Extract near-mention fragments from all text fields for group linking.
    combined = ftfy.fix_text(
        " ".join([booking.raw_location_wish, booking.guest_message, booking.comment])
    )
    near_texts = _extract_near_fragments(combined)

    return RawGroupSignals(
        organization=org,
        group_field=group_field_val,
        near_text_fragments=near_texts,
        is_org_private=is_private,
    )
