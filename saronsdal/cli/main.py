"""CLI entry point for Phase 1 + Phase 2: ingestion, normalization, and group resolution.

Usage:
    python -m saronsdal.cli.main \\
        --bookings-basic   "Sarons Dal/bookings_basic.csv.csv" \\
        --bookings-spec    "Sarons Dal/bookings_specification.csv.csv" \\
        --spots            "Sarons Dal/spots.csv.csv" \\
        --output-dir       output/

Produces:
    output/bookings_normalized.json   — all normalized bookings
    output/spots_normalized.json      — all normalized spots
    output/review_report.csv          — bookings that need manual review
    output/org_values.txt             — unique org/group field values (for alias config)
    output/group_links.jsonl          — one AffinityEdge per line (Phase 2)
    output/resolved_groups.json       — clusters + large_clusters (Phase 2)
    output/ambiguous_group_cases.csv  — name references that could not be resolved (Phase 2)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, List

from saronsdal.grouping.affinity_graph import build_graph
from saronsdal.grouping.group_normalizer import GroupNormalizer
from saronsdal.grouping.group_resolver import resolve_groups
from saronsdal.grouping.name_reference_extractor import NameReferenceExtractor
from saronsdal.grouping.signal_aggregator import aggregate_all
from saronsdal.ingestion.merger import merge_bookings
from saronsdal.ingestion.spot_loader import load_spots
from saronsdal.models.grouping import AmbiguousGroupCase, LargeWeakCluster, ResolvedCluster
from saronsdal.models.normalized import Booking, Spot
from saronsdal.normalization.booking_normalizer import normalise_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

from saronsdal.cli.utils import serialisable as _serialisable


def _write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_serialisable(data), fh, ensure_ascii=False, indent=2)
    logger.info("Wrote %s", path)


# ---------------------------------------------------------------------------
# Review report
# ---------------------------------------------------------------------------

_REVIEW_COLUMNS = [
    "booking_no",
    "full_name",
    "booking_source",
    "check_in",
    "check_out",
    "vehicle_type",
    "body_length_m",
    "total_width_m",
    "required_spot_count",
    "data_confidence",
    "review_flags",
]


def _write_review_report(bookings: List[Booking], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flagged = [b for b in bookings if b.review_flags or b.is_low_confidence]
    flagged.sort(key=lambda b: b.data_confidence)

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_REVIEW_COLUMNS)
        writer.writeheader()
        for b in flagged:
            writer.writerow({
                "booking_no":          b.booking_no,
                "full_name":           b.full_name,
                "booking_source":      b.booking_source,
                "check_in":            b.check_in.isoformat() if b.check_in else "",
                "check_out":           b.check_out.isoformat() if b.check_out else "",
                "vehicle_type":        b.vehicle.vehicle_type,
                "body_length_m":       b.vehicle.body_length_m if b.vehicle.body_length_m else "",
                "total_width_m":       b.vehicle.total_width_m if b.vehicle.total_width_m else "",
                "required_spot_count": b.vehicle.required_spot_count if b.vehicle.required_spot_count else "",
                "data_confidence":     b.data_confidence,
                "review_flags":        "; ".join(b.review_flags),
            })
    logger.info(
        "Review report: %d / %d bookings flagged → %s",
        len(flagged),
        len(bookings),
        path,
    )


# ---------------------------------------------------------------------------
# Phase 2 output writers
# ---------------------------------------------------------------------------

def _write_group_links(edges: list, path: Path) -> None:
    """Write one AffinityEdge per line as JSON (JSONL format)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for edge in edges:
            fh.write(json.dumps(_serialisable(edge), ensure_ascii=False) + "\n")
    logger.info("Wrote %d affinity edges → %s", len(edges), path)


def _write_resolved_groups(
    clusters: list,
    large_clusters: list,
    path: Path,
) -> None:
    """Write resolved clusters and large weak clusters to JSON."""
    data = {
        "clusters": _serialisable(clusters),
        "large_clusters": _serialisable(large_clusters),
    }
    _write_json(data, path)


_AMBIGUOUS_COLUMNS = [
    "booking_no",
    "reference_raw_text",
    "normalized_candidate",
    "matched_booking_nos",
    "reason",
    "confidence",
    "source_field",
]


def _write_ambiguous_cases(cases: List[AmbiguousGroupCase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_AMBIGUOUS_COLUMNS)
        writer.writeheader()
        for c in cases:
            writer.writerow({
                "booking_no":           c.booking_no,
                "reference_raw_text":   c.reference_raw_text,
                "normalized_candidate": c.normalized_candidate,
                "matched_booking_nos":  "; ".join(c.matched_booking_nos),
                "reason":               c.reason,
                "confidence":           f"{c.confidence:.3f}",
                "source_field":         c.source_field,
            })
    logger.info(
        "Ambiguous group cases: %d → %s", len(cases), path
    )


# ---------------------------------------------------------------------------
# Org value dump
# ---------------------------------------------------------------------------

def _write_org_values(bookings: List[Booking], path: Path) -> None:
    """Dump all unique org/group values to a text file for alias config work."""
    path.parent.mkdir(parents=True, exist_ok=True)
    values: set = set()
    for b in bookings:
        gs = b.group_signals
        if gs.organization:
            values.add(f"org: {gs.organization}")
        if gs.group_field:
            values.add(f"group_field: {gs.group_field}")
    with open(path, "w", encoding="utf-8") as fh:
        for v in sorted(values):
            fh.write(v + "\n")
    logger.info("Wrote %d unique org/group values → %s", len(values), path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args(argv: list | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sarons Dal campsite allocator — Phase 1: ingest and normalise bookings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--bookings-basic",
        required=True,
        metavar="FILE",
        help="Path to bookings_basic.csv",
    )
    p.add_argument(
        "--bookings-spec",
        required=True,
        metavar="FILE",
        help="Path to bookings_specification.csv",
    )
    p.add_argument(
        "--spots",
        required=True,
        metavar="FILE",
        help="Path to spots.csv",
    )
    p.add_argument(
        "--output-dir",
        default="output",
        metavar="DIR",
        help="Directory for output files (default: output/)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return p.parse_args(argv)


def main(argv: list | None = None) -> int:
    args = _parse_args(argv)
    logging.getLogger().setLevel(args.log_level)

    out = Path(args.output_dir)

    # ------------------------------------------------------------------ ingest
    logger.info("=== Phase 1: ingestion ===")
    try:
        raw_bookings = merge_bookings(
            Path(args.bookings_basic),
            Path(args.bookings_spec),
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    try:
        spots = load_spots(Path(args.spots))
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    # ------------------------------------------------------------------ normalise
    logger.info("=== Phase 1: normalisation ===")
    bookings = normalise_all(raw_bookings)

    # ------------------------------------------------------------------ outputs
    logger.info("=== Phase 1: writing outputs ===")
    _write_json(bookings, out / "bookings_normalized.json")
    _write_json(spots, out / "spots_normalized.json")
    _write_review_report(bookings, out / "review_report.csv")
    _write_org_values(bookings, out / "org_values.txt")

    # ------------------------------------------------------------------ phase 2
    logger.info("=== Phase 2: group resolution ===")
    normalizer = GroupNormalizer()
    extractor = NameReferenceExtractor(normalizer)
    signals = aggregate_all(bookings, normalizer, extractor)
    edges, ambiguous_cases, _alias_suggestions = build_graph(signals, bookings)
    all_bnos = [b.booking_no for b in bookings]
    clusters, large_clusters = resolve_groups(edges, all_bnos)

    logger.info("=== Phase 2: writing outputs ===")
    _write_group_links(edges, out / "group_links.jsonl")
    _write_resolved_groups(clusters, large_clusters, out / "resolved_groups.json")
    _write_ambiguous_cases(ambiguous_cases, out / "ambiguous_group_cases.csv")

    # ------------------------------------------------------------------ summary
    low_conf = sum(1 for b in bookings if b.is_low_confidence)
    needs_review = sum(1 for b in bookings if b.review_flags)
    unknown_type = sum(1 for b in bookings if b.vehicle.vehicle_type == "unknown")
    no_spot_count = sum(1 for b in bookings if b.vehicle.required_spot_count is None)
    explicit_width = sum(
        1 for b in bookings
        if "total_width_from_explicit_field" in b.vehicle.review_flags
    )
    width_fallback = len(bookings) - explicit_width
    width_oob = sum(
        1 for b in bookings if "width_out_of_bounds" in b.vehicle.review_flags
    )

    print("\n=== Summary ===")
    print(f"  Bookings loaded and normalised : {len(bookings)}")
    print(f"  Spots loaded                   : {len(spots)}")
    print(f"  Low confidence (<0.5)          : {low_conf}")
    print(f"  Need review (any flag)         : {needs_review}")
    print(f"  Unknown vehicle type           : {unknown_type}")
    print(f"  Unknown spot count             : {no_spot_count}")
    print(f"  Width: explicit raw_total_width: {explicit_width}")
    print(f"  Width: fallback (inferred)     : {width_fallback}")
    print(f"  Width: out-of-bounds flag      : {width_oob}")

    total_clustered = sum(len(c.members) for c in clusters)
    total_lwc_members = sum(len(lwc.all_member_booking_nos) for lwc in large_clusters)
    print(f"\n  Groups:")
    print(f"  Affinity edges                 : {len(edges)}")
    print(f"  Resolved clusters              : {len(clusters)}  ({total_clustered} bookings)")
    print(f"  Large weak clusters            : {len(large_clusters)}  ({total_lwc_members} bookings)")
    print(f"  Ambiguous name references      : {len(ambiguous_cases)}")
    print(f"  Output directory               : {out.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
