"""CLI entry point for Phase 3: spatial allocation.

Usage:
    python -m saronsdal.cli.allocate \\
        --bookings  output/bookings_normalized.json \\
        --spots     output/spots_normalized.json \\
        --topology  "Sarons Dal/Topology" \\
        --groups    output/resolved_groups.json \\
        --prefs     output/llm_suggestions/preference_enrichments.jsonl \\
        --subsec    output/llm_suggestions/subsection_resolutions.jsonl \\
        --output    output/allocation_results.json

Produces:
    allocation_results.json — one entry per booking with assigned spot, score,
                              and explanation dict.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import dataclasses
from datetime import date
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def _dataclass_default(cls, field_name: str) -> Any:
    """Return the default value for a dataclass field, or None if no default."""
    f = cls.__dataclass_fields__[field_name]
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return None


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------

from saronsdal.cli.utils import serialisable as _serialisable


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_bookings(path: Path):
    from saronsdal.models.normalized import Booking, VehicleUnit, SpotRequest, RawGroupSignals, SectionRow
    import dataclasses

    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    bookings = []
    from datetime import date as _date
    for d in raw:
        # Reconstruct Booking from dict; skip unknown fields gracefully
        try:
            # Dates
            for date_field in ("check_in", "check_out"):
                if isinstance(d.get(date_field), str):
                    try:
                        d[date_field] = _date.fromisoformat(d[date_field])
                    except (ValueError, TypeError):
                        d[date_field] = None

            # VehicleUnit
            v = d.get("vehicle", {})
            v.pop("__class__", None)
            vehicle = VehicleUnit(**{
                k: v.get(k) for k in VehicleUnit.__dataclass_fields__
            })

            # SpotRequest
            r = d.get("request", {})
            r.pop("__class__", None)
            if "amenity_flags" in r and isinstance(r["amenity_flags"], list):
                r["amenity_flags"] = set(r["amenity_flags"])
            rows = []
            for sr in r.get("preferred_section_rows", []):
                rows.append(SectionRow(section=sr["section"], row=sr["row"]))
            r["preferred_section_rows"] = rows
            request = SpotRequest(**{
                k: r.get(k, _dataclass_default(SpotRequest, k))
                for k in SpotRequest.__dataclass_fields__
            })

            # RawGroupSignals
            gs = d.get("group_signals", {})
            gs.pop("__class__", None)
            group_signals = RawGroupSignals(**{
                k: gs.get(k) for k in RawGroupSignals.__dataclass_fields__
            })

            booking = Booking(
                booking_no=d["booking_no"],
                booking_source=d.get("booking_source", "other"),
                check_in=d.get("check_in"),
                check_out=d.get("check_out"),
                first_name=d.get("first_name", ""),
                last_name=d.get("last_name", ""),
                full_name=d.get("full_name", ""),
                num_guests=d.get("num_guests", 1),
                language=d.get("language", ""),
                is_confirmed=d.get("is_confirmed", False),
                vehicle=vehicle,
                request=request,
                group_signals=group_signals,
                data_confidence=d.get("data_confidence", 0.0),
                review_flags=d.get("review_flags", []),
                raw_guest_message=d.get("raw_guest_message", ""),
                raw_comment=d.get("raw_comment", ""),
                raw_location_wish=d.get("raw_location_wish", ""),
                city=d.get("city", ""),
            )
            bookings.append(booking)
        except Exception as exc:
            logger.warning("Skipping booking %s: %s", d.get("booking_no", "?"), exc)

    logger.info("Loaded %d bookings from %s", len(bookings), path.name)
    return bookings


def _load_clusters(path: Optional[Path]):
    if not path:
        return []
    if not path.exists():
        logger.warning("--groups file not found: %s (no cluster data loaded)", path)
        return []
    from saronsdal.models.grouping import ResolvedCluster, AffinityEdge
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    clusters = []
    for d in data.get("clusters", []):
        try:
            clusters.append(ResolvedCluster(**{
                k: d.get(k) for k in ResolvedCluster.__dataclass_fields__
            }))
        except Exception:
            pass
    logger.info("Loaded %d clusters from %s", len(clusters), path.name)
    return clusters


def _load_pref_suggestions(path: Optional[Path]):
    if not path:
        return []
    if not path.exists():
        logger.warning("--prefs file not found: %s (no preference enrichments loaded)", path)
        return []
    from saronsdal.llm.schemas import PreferenceSuggestion, StructuredPreferences
    results = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                pd = d.get("preferences", {})
                prefs = StructuredPreferences(**{
                    k: pd.get(k, StructuredPreferences.__dataclass_fields__[k].default)
                    for k in StructuredPreferences.__dataclass_fields__
                })
                results.append(PreferenceSuggestion(
                    booking_no=d["booking_no"],
                    full_name=d.get("full_name", ""),
                    preferences=prefs,
                    confidence=d.get("confidence", 0.0),
                    raw_text=d.get("raw_text", ""),
                ))
            except Exception as exc:
                logger.warning("Skipping pref line: %s", exc)
    logger.info("Loaded %d preference suggestions from %s", len(results), path.name)
    return results


def _load_subsection_suggestions(path: Optional[Path]):
    if not path:
        return []
    if not path.exists():
        logger.warning("--subsec file not found: %s (no subsection suggestions loaded)", path)
        return []
    from saronsdal.llm.schemas import SubsectionSuggestion
    results = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                results.append(SubsectionSuggestion(
                    booking_no=d["booking_no"],
                    full_name=d.get("full_name", ""),
                    extracted_section=d.get("extracted_section", ""),
                    suggested_rows=d.get("suggested_rows", []),
                    suggested_spot_ids=d.get("suggested_spot_ids", []),
                    confidence=d.get("confidence", 0.0),
                    notes=d.get("notes", ""),
                ))
            except Exception as exc:
                logger.warning("Skipping subsection line: %s", exc)
    logger.info(
        "Loaded %d subsection suggestions from %s", len(results), path.name
    )
    return results


def _load_near_refs(path: Optional[Path]):
    """Load reference_resolutions.jsonl → list of NearTextSuggestion objects."""
    if not path:
        return []
    if not path.exists():
        logger.warning("--refs file not found: %s (no near-text reference links loaded)", path)
        return []
    from saronsdal.llm.schemas import NearTextSuggestion, ResolvedRef
    results = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                resolved_refs = []
                for r in d.get("resolved_refs", []):
                    resolved_refs.append(ResolvedRef(
                        raw_fragment=r.get("raw_fragment", ""),
                        matched_booking_no=r.get("matched_booking_no"),
                        match_type=r.get("match_type", "unresolved"),
                        confidence=float(r.get("confidence", 0.0)),
                    ))
                results.append(NearTextSuggestion(
                    booking_no=d["booking_no"],
                    full_name=d.get("full_name", ""),
                    resolved_refs=resolved_refs,
                    unresolved_fragments=d.get("unresolved_fragments", []),
                    notes=d.get("notes", ""),
                    place_refs=d.get("place_refs", []),
                ))
            except Exception as exc:
                logger.warning("Skipping near-ref line: %s", exc)
    logger.info(
        "Loaded %d near-text reference suggestions from %s", len(results), path.name
    )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Phase 3: greedy spot allocation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bookings",  required=True, type=Path,
                        help="bookings_normalized.json from Phase 1")
    parser.add_argument("--spots",     required=True, type=Path,
                        help="spots_normalized.json from Phase 1")
    parser.add_argument("--rooms",     required=True, type=Path,
                        help="sirvoy_room_ids.csv — authoritative Sirvoy room definitions")
    parser.add_argument("--topology",  required=True, type=Path,
                        help="Directory containing topology_grid_*.csv files")
    parser.add_argument("--groups",    type=Path, default=None,
                        help="resolved_groups.json from Phase 2 (optional)")
    parser.add_argument("--prefs",     type=Path, default=None,
                        help="preference_enrichments.jsonl from Phase 2.5 (optional)")
    parser.add_argument("--subsec",    type=Path, default=None,
                        help="subsection_resolutions.jsonl from Phase 2.5 (optional)")
    parser.add_argument("--refs",      type=Path, default=None,
                        help="reference_resolutions.jsonl from Phase 2.5 (optional)")
    parser.add_argument("--output",    type=Path, default=Path("allocation_results.json"),
                        help="Output file path")
    parser.add_argument("--debug",     action="store_true",
                        help="Enable DEBUG logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    # Load inputs
    from saronsdal.ingestion.spot_loader import load_spots_normalized
    from saronsdal.ingestion.triplet_loader import load_triplets
    from saronsdal.spatial.topology_loader import load_topology
    from saronsdal.allocation.allocator import allocate

    bookings   = _load_bookings(args.bookings)
    spots      = load_spots_normalized(args.spots)
    topo       = load_topology(args.topology)
    clusters   = _load_clusters(args.groups)
    prefs      = _load_pref_suggestions(args.prefs)
    subsecs    = _load_subsection_suggestions(args.subsec)
    near_refs  = _load_near_refs(args.refs)

    # Build triplet catalogue from sirvoy_room_ids.csv + spot data
    spot_lookup = {(s.section, s.spot_id): s for s in spots}
    triplets    = load_triplets(args.rooms, spot_lookup)
    allocatable = sum(1 for t in triplets if t.is_allocatable)

    logger.info(
        "Inputs: %d bookings, %d spots, %d triplets (%d allocatable), "
        "%d grid sections, %d clusters, "
        "%d pref suggestions, %d subsection suggestions, %d near-text refs",
        len(bookings), len(spots), len(triplets), allocatable,
        len(topo.sections()),
        len(clusters), len(prefs), len(subsecs), len(near_refs),
    )

    # Run allocation
    results = allocate(
        bookings=bookings,
        triplets=triplets,
        topo=topo,
        clusters=clusters or None,
        pref_suggestions=prefs or None,
        subsection_suggestions=subsecs or None,
        near_refs=near_refs or None,
    )

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = [_serialisable(r) for r in results]
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)

    assigned = sum(1 for r in results if r.is_assigned)
    print(f"Allocation complete: {assigned}/{len(results)} bookings assigned.")
    print(f"Results written to: {args.output}")

    if topo.unrecognised_cells:
        print(
            f"Warning: {len(topo.unrecognised_cells)} unrecognised topology cell(s) "
            f"— run with --debug to inspect."
        )


if __name__ == "__main__":
    main()
