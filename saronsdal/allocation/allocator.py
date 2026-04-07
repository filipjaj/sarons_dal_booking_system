"""Phase 3B — Greedy deterministic spot allocator (v2 — Sirvoy triplet units).

Algorithm:
  1. Sort bookings by descending constraint strength (most-constrained first).
     Within equal strength, sort by booking_no for determinism.
  2. For each booking:
     a. Build candidate list: all allocatable Triplets with all three spots free.
     b. Rank candidates via candidate_ranker.rank_candidates(), which:
        - scores preferences (proximity, terrain, section match, …)
        - performs a hard length feasibility check on the anchor spot
        - scores group proximity (1/(1+mean_distance) to assigned members)
        - discards hard violations (wrong section, vehicle restriction, too long)
        - combines: total = W_PREF * pref + W_GROUP * group
     c. Assign the top-ranked Triplet.
     d. Mark all three underlying spot IDs as occupied.
  3. Bookings with no valid candidates are left unassigned with a reason.

Constraint strength heuristic:
  +10  specific spot IDs requested
  + 4  specific row/subsection requested (Phase 2.5 subsection suggestion)
  + 2  section preference set
  + 1  group membership (org or group_field set)
  + 1  near-text social signal (source or target of a resolved near-ref)

Each booking is assigned exactly one Triplet (three consecutive spots).
The output carries the exact Sirvoy Room ID string for each assignment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from saronsdal.allocation.candidate_ranker import rank_candidates
from saronsdal.allocation.group_scorer import GroupContext
from saronsdal.allocation.preference_scorer import score_spot
from saronsdal.models.normalized import Booking, Spot
from saronsdal.models.triplet import Triplet
from saronsdal.spatial.topology_loader import Topology

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output data class
# ---------------------------------------------------------------------------


@dataclass
class AllocationResult:
    """Allocation outcome for a single booking."""
    booking_no: str
    full_name: str
    assigned_section: Optional[str]     # section name; None when unassigned
    assigned_spot_id: Optional[str]     # first (anchor) spot ID; None when unassigned
    assigned_room_id: Optional[str]     # exact Sirvoy Room ID string; None when unassigned
    assigned_spot_ids: List[str]        # all spot IDs in the room; [] when unassigned
    score: float                        # final preference score (0.0 when unassigned)
    explanation: Dict[str, Any]         # score components + notes
    unassigned_reason: Optional[str]    # set only when assigned_room_id is None

    @property
    def is_assigned(self) -> bool:
        return self.assigned_room_id is not None

    @property
    def assigned_spot(self) -> Optional[str]:
        """'Section AnchorSpotID' e.g. 'Furulunden A16'. None when unassigned."""
        if self.assigned_spot_id is None:
            return None
        return f"{self.assigned_section} {self.assigned_spot_id}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _constraint_strength(
    booking: Booking,
    sub=None,
    near_ref_map: Optional[Dict[str, List[str]]] = None,
    near_ref_target_set: Optional[Set[str]] = None,
) -> int:
    """Compute a constraint-strength integer for sorting (higher = more constrained).

    Args:
        booking:              The booking to evaluate.
        sub:                  Phase 2.5 SubsectionSuggestion for this booking.
                              Its suggested_spot_ids and suggested_rows are treated as
                              equivalent to explicit Phase 1 requests.
        near_ref_map:         Phase 2.5 near-text reference map (source → targets).
                              A booking that IS a source gets +1.
        near_ref_target_set:  Set of booking_nos that appear as TARGETS of any near-ref.
                              A booking that IS a target gets +1, so it is placed before
                              unconstrained bookings and can serve as an anchor for
                              bookings that expressed a wish to be near it.

    Tiers:
        +10  explicit spot IDs (Phase 1) OR Phase 2.5 suggested_spot_ids
        + 4  explicit section rows (Phase 1) OR Phase 2.5 suggested_rows
        + 2  section preference (Phase 1)
        + 1  group / organisation signal (Phase 1)
        + 1  near-text ref: booking is a source OR a target of a resolved near-ref
    """
    strength = 0

    has_spot_ids = bool(booking.request.preferred_spot_ids) or bool(
        sub and sub.suggested_spot_ids
    )
    if has_spot_ids:
        strength += 10

    has_rows = bool(booking.request.preferred_section_rows) or bool(
        sub and sub.suggested_rows
    )
    if has_rows:
        strength += 4

    if booking.request.preferred_sections:
        strength += 2

    gs = booking.group_signals
    if gs.organization or gs.group_field:
        strength += 1

    # Near-text social signal: source or target of a resolved near-ref
    bno = booking.booking_no
    has_near_ref_signal = (
        (near_ref_map is not None and bno in near_ref_map)
        or (near_ref_target_set is not None and bno in near_ref_target_set)
    )
    if has_near_ref_signal:
        strength += 1

    return strength


def _build_pref_map(pref_suggestions) -> Dict[str, Any]:
    """booking_no → StructuredPreferences (from Phase 2.5)."""
    if not pref_suggestions:
        return {}
    return {p.booking_no: p.preferences for p in pref_suggestions}


def _build_subsection_map(subsection_suggestions) -> Dict[str, Any]:
    """booking_no → SubsectionSuggestion (from Phase 2.5)."""
    if not subsection_suggestions:
        return {}
    return {s.booking_no: s for s in subsection_suggestions}


#: Minimum ResolvedRef.confidence to treat a near-text link as a proximity target.
#: Accepts full_name / surname / family matches; rejects weak first_name guesses.
NEAR_REF_CONFIDENCE_THRESHOLD: float = 0.65

#: match_type values that represent place / group membership rather than
#: a personal name wish.  These links are made symmetric: if A → B via
#: group_reference, B → A is injected so that after *any* member is placed
#: the rest of the place-group can cluster toward it.
SYMMETRIC_REF_TYPES: frozenset[str] = frozenset({"group_reference"})


def _build_near_ref_map(
    near_refs,
    threshold: float = NEAR_REF_CONFIDENCE_THRESHOLD,
    symmetric_types: frozenset = SYMMETRIC_REF_TYPES,
) -> Dict[str, List[str]]:
    """booking_no → [booking_nos this booking wants to be near] (Phase 2.5 near-text refs).

    Two source types are consumed from each NearTextSuggestion:

    resolved_refs (Gemini personal-name / group_reference resolutions):
        - Each ref has a single ``matched_booking_no`` (or None when unresolved).
        - Filtered by: matched_booking_no is not None, match_type != "unresolved",
          confidence >= threshold.

    place_refs (deterministic place/city-based group references):
        - Each ref is a dict produced by place_resolver.PlaceResolutionResult.
        - Has ``matched_booking_nos`` (plural list) rather than a single booking.
        - Filtered by: match_type != "unresolved", confidence >= threshold.
        - All matched_booking_nos in a qualifying place_ref are added as targets.

    Symmetry:
        For ``match_type`` values in ``symmetric_types`` (default: ``group_reference``),
        a reverse link is injected automatically.  ``place_group`` refs are
        directional by default — only the booking that expressed the wish gets the
        proximity pull.  Pass ``symmetric_types=frozenset({"place_group"})`` to
        change this.

    Args:
        near_refs:       List of NearTextSuggestion objects from reference_resolutions.jsonl.
        threshold:       Minimum confidence to include a resolved reference.
        symmetric_types: match_type values for which reverse links are injected.

    Returns:
        Dict mapping booking_no to a deduplicated list of target booking_nos.
    """
    if not near_refs:
        return {}

    # First pass: collect directed edges
    result: Dict[str, List[str]] = {}
    symmetric_pairs: List[tuple] = []   # (source, target) that need reverse injection

    for suggestion in near_refs:
        targets: List[str] = []

        # --- resolved_refs: Gemini personal-name / group_reference links ---
        for ref in suggestion.resolved_refs:
            if (
                ref.matched_booking_no is not None
                and ref.match_type != "unresolved"
                and ref.confidence >= threshold
            ):
                tgt = ref.matched_booking_no
                if tgt not in targets:
                    targets.append(tgt)
                if ref.match_type in symmetric_types:
                    symmetric_pairs.append((suggestion.booking_no, tgt))

        # --- place_refs: deterministic place/city group links ---
        # Each entry is a dict with matched_booking_nos (plural list).
        for pr in (suggestion.place_refs or []):
            if not isinstance(pr, dict):
                continue
            pr_match_type = pr.get("match_type", "unresolved")
            pr_confidence = float(pr.get("confidence", 0.0))
            if pr_match_type == "unresolved" or pr_confidence < threshold:
                continue
            for tgt in pr.get("matched_booking_nos") or []:
                if tgt is None:
                    continue
                if tgt not in targets:
                    targets.append(tgt)
                if pr_match_type in symmetric_types:
                    symmetric_pairs.append((suggestion.booking_no, tgt))

        if targets:
            result[suggestion.booking_no] = targets

    # Second pass: inject reverse links for symmetric types
    for src, tgt in symmetric_pairs:
        entry = result.setdefault(tgt, [])
        if src not in entry:
            entry.append(src)

    return result


def _build_group_map(clusters) -> Dict[str, List[str]]:
    """booking_no → list of co-member booking_nos (from Phase 2 clusters).

    Supports both the current ResolvedCluster schema (field: ``members``) and
    the older schema (field: ``member_booking_nos``) for backward compatibility
    with previously serialised resolved_groups.json files.
    """
    if not clusters:
        return {}
    result: Dict[str, List[str]] = {}
    for cluster in clusters:
        # Current schema: members.  Older schema: member_booking_nos.
        if hasattr(cluster, "members"):
            members = cluster.members
        elif hasattr(cluster, "member_booking_nos"):
            members = cluster.member_booking_nos
        else:
            logger.warning(
                "Cluster %r has neither 'members' nor 'member_booking_nos' — skipped",
                getattr(cluster, "cluster_id", "?"),
            )
            continue
        for bno in members:
            result[bno] = [m for m in members if m != bno]
    return result


# ---------------------------------------------------------------------------
# Group-context seeding helpers
# ---------------------------------------------------------------------------


def _build_group_context_map(
    bookings: List[Booking],
    group_map: Dict[str, List[str]],
    near_ref_map: Dict[str, List[str]],
    sub_map: Dict[str, Any],
) -> Dict[str, GroupContext]:
    """Build a GroupContext for every booking that has linked co-members with preferences.

    For each booking B, collects the preferences of all linked co-members (NOT B
    itself) and accumulates them into weighted counts.  One vote per member per
    preference item regardless of how many times that member repeated it.

    A booking only gets an entry in the result map if at least one linked co-member
    contributed a preference signal (section, row, or spot_id).

    Args:
        bookings:     All normalized bookings.
        group_map:    Phase 2 cluster membership (booking_no → [co-member booking_nos]).
        near_ref_map: Phase 2.5 near-text links (booking_no → [target booking_nos]).
        sub_map:      Phase 2.5 subsection suggestions (booking_no → SubsectionSuggestion).

    Returns:
        Dict mapping booking_no → GroupContext for bookings with usable co-member data.
    """
    booking_lookup: Dict[str, Booking] = {b.booking_no: b for b in bookings}
    result: Dict[str, GroupContext] = {}

    for bno, booking in booking_lookup.items():
        cluster_members: Set[str] = set(group_map.get(bno, []))
        near_members: Set[str] = set(near_ref_map.get(bno, []))
        linked: Set[str] = cluster_members | near_members
        if not linked:
            continue

        section_weights: Dict[str, int] = {}
        row_weights: Dict[str, int] = {}
        spot_id_weights: Dict[str, int] = {}
        contributing: List[str] = []
        sources: Dict[str, str] = {}

        for m_bno in linked:
            member = booking_lookup.get(m_bno)
            if member is None:
                continue  # reference to a booking not in our dataset

            # Collect this member's preferences; deduplicate within the member
            # so one member = one vote per dimension regardless of repetition.
            m_sections: Set[str] = set(member.request.preferred_sections)

            m_rows: Set[str] = {sr.row for sr in member.request.preferred_section_rows}
            m_sub = sub_map.get(m_bno)
            if m_sub:
                m_rows |= set(m_sub.suggested_rows)

            m_spots: Set[str] = set(member.request.preferred_spot_ids)
            if m_sub:
                m_spots |= set(m_sub.suggested_spot_ids)

            # Accumulate votes
            for sec in m_sections:
                section_weights[sec] = section_weights.get(sec, 0) + 1
            for row in m_rows:
                row_weights[row] = row_weights.get(row, 0) + 1
            for sid in m_spots:
                spot_id_weights[sid] = spot_id_weights.get(sid, 0) + 1

            if m_sections or m_rows or m_spots:
                contributing.append(m_bno)

            # Determine link source for provenance
            in_cluster = m_bno in cluster_members
            in_near = m_bno in near_members
            if in_cluster and in_near:
                sources[m_bno] = "both"
            elif in_cluster:
                sources[m_bno] = "cluster"
            else:
                sources[m_bno] = "near_ref"

        if section_weights or row_weights or spot_id_weights:
            result[bno] = GroupContext(
                section_weights=section_weights,
                row_weights=row_weights,
                spot_id_weights=spot_id_weights,
                contributing_booking_nos=contributing,
                contributor_sources=sources,
            )

    logger.debug(
        "Group context map: %d/%d bookings have usable co-member preference data",
        len(result), len(bookings),
    )
    return result


# ---------------------------------------------------------------------------
# Diagnostic helper (non-allocating)
# ---------------------------------------------------------------------------


def diagnose_pipeline(
    bookings: List[Booking],
    probe_booking_nos: List[str],
    clusters=None,
    subsection_suggestions=None,
    near_refs=None,
) -> Dict[str, Any]:
    """Return a diagnostic snapshot of the social-signal pipeline for probe bookings.

    Does NOT run allocation.  Reports on the intermediate maps so you can trace
    exactly where a near-ref or cluster link is present, filtered, or missing.

    Returns a dict keyed by booking_no, each value containing:
        raw_near_ref_suggestion  — what was parsed from the NearTextSuggestion
        near_ref_map_entry       — what _build_near_ref_map produced
        group_map_entry          — what _build_group_map produced
        group_context            — what _build_group_context_map produced (or None)
        constraint_strength      — computed sort key value
    """
    sub_map   = _build_subsection_map(subsection_suggestions)
    group_map = _build_group_map(clusters)
    near_ref_map = _build_near_ref_map(near_refs)
    group_context_map = _build_group_context_map(bookings, group_map, near_ref_map, sub_map)
    near_ref_target_set: Set[str] = {
        bno for targets in near_ref_map.values() for bno in targets
    }

    booking_lookup: Dict[str, Booking] = {b.booking_no: b for b in bookings}

    # Raw NearTextSuggestion data per booking
    raw_suggestions: Dict[str, Any] = {}
    if near_refs:
        for s in near_refs:
            if s.booking_no in probe_booking_nos:
                raw_suggestions[s.booking_no] = {
                    "resolved_refs": [
                        {
                            "raw_fragment": r.raw_fragment,
                            "matched_booking_no": r.matched_booking_no,
                            "match_type": r.match_type,
                            "confidence": r.confidence,
                        }
                        for r in s.resolved_refs
                    ],
                    "place_refs_count": len(s.place_refs or []),
                    "place_refs": s.place_refs or [],
                }

    result: Dict[str, Any] = {}
    for bno in probe_booking_nos:
        booking = booking_lookup.get(bno)
        ctx = group_context_map.get(bno)

        result[bno] = {
            "full_name": booking.full_name if booking else "NOT FOUND",
            "constraint_strength": _constraint_strength(
                booking, sub_map.get(bno), near_ref_map, near_ref_target_set
            ) if booking else None,

            # ── Link maps ────────────────────────────────────────────────────
            "group_map_entry": group_map.get(bno, []),
            "near_ref_map_entry": near_ref_map.get(bno, []),
            "is_near_ref_target_of": [
                src for src, tgts in near_ref_map.items() if bno in tgts
            ],

            # ── Raw NearTextSuggestion ────────────────────────────────────────
            "raw_near_ref_suggestion": raw_suggestions.get(bno, "NO ENTRY IN near_refs"),

            # ── Group context ─────────────────────────────────────────────────
            "group_context_present": ctx is not None,
            "group_context": {
                "contributing_booking_nos": ctx.contributing_booking_nos,
                "contributor_sources": ctx.contributor_sources,
                "section_weights": ctx.section_weights,
                "row_weights": ctx.row_weights,
                "spot_id_weights": ctx.spot_id_weights,
            } if ctx else None,
        }

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def allocate(
    bookings: List[Booking],
    triplets: List[Triplet],
    topo: Topology,
    clusters=None,                # List[ResolvedCluster] | None
    pref_suggestions=None,        # List[PreferenceSuggestion] | None
    subsection_suggestions=None,  # List[SubsectionSuggestion] | None
    near_refs=None,               # List[NearTextSuggestion] | None
) -> List[AllocationResult]:
    """Run the greedy allocation algorithm assigning one Triplet per booking.

    Args:
        bookings:               All normalized bookings.
        triplets:               Candidate Sirvoy room units from sirvoy_room_ids.csv.
                                Only Triplets with ``is_allocatable == True`` enter
                                the candidate pool.
        topo:                   Loaded Topology (spatial coordinates).
        clusters:               Phase 2 resolved booking clusters (for group proximity).
        pref_suggestions:       Phase 2.5 structured preference suggestions.
        subsection_suggestions: Phase 2.5 row/spot-range resolutions.
        near_refs:              Phase 2.5 near-text reference resolutions
                                (reference_resolutions.jsonl).  Directional:
                                booking A gets proximity pull toward booking B
                                when A expressed a near-text wish for B.

    Returns:
        One AllocationResult per booking, in booking_no order.
        Each assigned result carries:
            assigned_room_id  — exact Sirvoy Room ID string
            assigned_spot_ids — all three underlying spot IDs
            assigned_spot_id  — first (anchor) spot ID (proximity anchor)
    """
    pref_map      = _build_pref_map(pref_suggestions)
    sub_map       = _build_subsection_map(subsection_suggestions)
    group_map     = _build_group_map(clusters)
    near_ref_map  = _build_near_ref_map(near_refs)

    # Group-context seeding: merged co-member preferences for first-in-cluster
    # placement.  Built after all link maps so it uses the full link graph.
    group_context_map = _build_group_context_map(
        bookings, group_map, near_ref_map, sub_map
    )

    # Set of booking_nos that are *targets* of any near-ref link.
    # Used to boost constraint strength so that "anchor" bookings are placed
    # before the bookings that expressed a wish to be near them.
    near_ref_target_set: Set[str] = {
        bno for targets in near_ref_map.values() for bno in targets
    }

    # Individual spot IDs that have been occupied by an assigned triplet.
    # Keyed by (section, spot_id) to avoid cross-section collisions.
    occupied_spot_keys: Set[Tuple[str, str]] = set()

    # Assignments so far — stores anchor (section, first_spot_id) for group
    # proximity calculations (distance is measured from the anchor spot).
    assignments: Dict[str, Tuple[str, str]] = {}

    # Sort: most-constrained first; tie-break on booking_no for full determinism.
    sorted_bookings = sorted(
        bookings,
        key=lambda b: (
            -_constraint_strength(
                b,
                sub_map.get(b.booking_no),
                near_ref_map,
                near_ref_target_set,
            ),
            b.booking_no,
        ),
    )

    results: List[AllocationResult] = []

    for booking in sorted_bookings:
        bno = booking.booking_no
        prefs = pref_map.get(bno)
        sub = sub_map.get(bno)

        # Collect explicit spot/row preferences
        preferred_spot_ids = list(booking.request.preferred_spot_ids)
        preferred_rows = [sr.row for sr in booking.request.preferred_section_rows]

        # Supplement from Phase 2.5 subsection suggestions
        if sub:
            for sid in sub.suggested_spot_ids:
                if sid not in preferred_spot_ids:
                    preferred_spot_ids.append(sid)
            for row in sub.suggested_rows:
                if row not in preferred_rows:
                    preferred_rows.append(row)

        # Spots already assigned to social neighbours:
        #   - symmetric cluster members (Phase 2 resolved_groups)
        #   - directional near-text targets (Phase 2.5 reference_resolutions)
        co_members   = group_map.get(bno, [])
        near_targets = near_ref_map.get(bno, [])
        relevant_bnos: Set[str] = set(co_members) | set(near_targets)
        group_assigned: List[Tuple[str, str]] = [
            assignments[m] for m in relevant_bnos if m in assignments
        ]

        # Group-context seeding: active only when no co-members are placed yet.
        # Once any co-member is assigned, normal group proximity takes over.
        group_context = (
            group_context_map.get(bno)
            if not group_assigned
            else None
        )

        # Available candidates: allocatable triplets with all spot IDs free
        candidates = [
            t for t in triplets
            if t.is_allocatable and all(
                (t.section, sid) not in occupied_spot_keys
                for sid in t.spot_ids
            )
        ]

        if not candidates:
            results.append(AllocationResult(
                booking_no=bno,
                full_name=booking.full_name,
                assigned_section=None,
                assigned_spot_id=None,
                assigned_room_id=None,
                assigned_spot_ids=[],
                score=0.0,
                explanation={"error": "no available triplets"},
                unassigned_reason="no_available_triplets",
            ))
            continue

        # Rank candidates (filters hard violations, combines pref + group scores)
        ranked = rank_candidates(
            booking=booking,
            candidates=candidates,
            topo=topo,
            prefs=prefs,
            group_spots=group_assigned or None,
            preferred_spot_ids=preferred_spot_ids or None,
            preferred_rows=preferred_rows or None,
            group_context=group_context,
        )

        if not ranked:
            # All candidates had hard violations — report dominant violation
            violation_counts: Dict[str, int] = {}
            for t in candidates:
                if t.first_spot is None:
                    continue
                sc = score_spot(
                    spot=t.first_spot,
                    booking=booking,
                    topo=topo,
                    prefs=prefs,
                    preferred_spot_ids=preferred_spot_ids or None,
                    preferred_rows=preferred_rows or None,
                )
                for v in sc.violations:
                    violation_counts[v] = violation_counts.get(v, 0) + 1
            dominant = (
                max(violation_counts, key=violation_counts.__getitem__)
                if violation_counts
                else "unknown"
            )
            logger.warning(
                "Booking %s: no valid triplet (dominant violation=%s, checked %d triplets)",
                bno, dominant, len(candidates),
            )
            results.append(AllocationResult(
                booking_no=bno,
                full_name=booking.full_name,
                assigned_section=None,
                assigned_spot_id=None,
                assigned_room_id=None,
                assigned_spot_ids=[],
                score=0.0,
                explanation={"violation_counts": violation_counts},
                unassigned_reason=f"all_triplets_violated:{dominant}",
            ))
            continue

        # Assign the best-ranked triplet
        best = ranked[0]
        triplet = best.triplet

        # Mark all three spot IDs occupied
        for sid in triplet.spot_ids:
            occupied_spot_keys.add((triplet.section, sid))

        # Record anchor position for group proximity of subsequent bookings
        assignments[bno] = (triplet.section, triplet.first_spot_id)

        explanation: Dict[str, Any] = {
            "preference_score": best.preference_score,
            "group_score": best.group_score,
            "preference_components": best.spot_score.components,
            "preference_notes": best.spot_score.notes,
            "group_notes": best.group_detail.notes,
            "group_proximity_to": [f"{s}/{i}" for s, i in group_assigned],
            "group_context_seeded": group_context is not None,
        }
        if group_context is not None:
            explanation["group_context_contributors"] = group_context.contributor_sources
            explanation["group_context_section_weights"] = group_context.section_weights
            explanation["group_context_row_weights"] = group_context.row_weights
            explanation["group_context_spot_id_weights"] = group_context.spot_id_weights

        results.append(AllocationResult(
            booking_no=bno,
            full_name=booking.full_name,
            assigned_section=triplet.section,
            assigned_spot_id=triplet.first_spot_id,
            assigned_room_id=triplet.room_id,
            assigned_spot_ids=list(triplet.spot_ids),
            score=best.total_score,
            explanation=explanation,
            unassigned_reason=None,
        ))

    # Restore original booking_no order for output
    order = {b.booking_no: i for i, b in enumerate(bookings)}
    results.sort(key=lambda r: order.get(r.booking_no, 0))

    assigned_count = sum(1 for r in results if r.is_assigned)
    seeded_count = sum(
        1 for r in results
        if r.is_assigned and r.explanation.get("group_context_seeded")
    )
    logger.info(
        "Allocation complete: %d/%d bookings assigned, %d unassigned "
        "(group clusters: %d, near-ref links: %d bookings, "
        "group-context seeded: %d first-in-cluster placements)",
        assigned_count, len(results), len(results) - assigned_count,
        len(group_map), len(near_ref_map), seeded_count,
    )
    return results
