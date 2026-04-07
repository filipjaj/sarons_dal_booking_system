"""Phase 2.5 — Prompt builders for each Gemini task.

Each builder returns a user-message string.  The shared system instruction is
kept separate so the GeminiClient can attach it to the model once at init time.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from saronsdal.llm.candidate_builder import (
    GroupPhraseCandidate,
    NearTextCandidate,
    PreferenceCandidate,
    SubsectionCandidate,
    WeakClusterCandidate,
)

# Maximum characters of raw booking text to include in LLM prompts.
# Prevents excessively long user-supplied text from dominating the prompt.
_MAX_RAW_TEXT_LEN = 500

# ---------------------------------------------------------------------------
# Shared system instruction (attached to model, not repeated per prompt)
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = """\
You are an assistant helping manage campsite bookings at "Sarons Dal", a
Pentecostal Christian camping ground near Jæren, Norway.

Key context:
- Campsite sections: Furulunden, Elvebredden, Fjellterrassen, Vårdalen,
  Egelandsletta, Furutoppen, Internatet, Bedehuset.
- Many guests are from Pentecostal churches in south-western Norway:
  Betel, Filadelfia, Pinsemenigheten, Betania, Sion, Pinsekirken.
- "bibelskolen" = the Bible school building at the campsite.
- "i fjor" = last year.  "gjengen" = the group/gang.
- "camp X" = youth church camp name.
- Guest messages are mainly in Norwegian; occasionally in English.
- Spot IDs follow the pattern [Row][Number], e.g. "D25", "E07", "A1".

Return responses as clean JSON matching the schema provided in each prompt.
Do not add fields not in the schema.  Do not add commentary outside the JSON.
"""


# ---------------------------------------------------------------------------
# Selector 1 + 3: group phrase classification
# ---------------------------------------------------------------------------

def build_group_phrase_prompt(
    phrase_cands: List[GroupPhraseCandidate],
    cluster_cands: List[WeakClusterCandidate],
    bookings: List[dict],
    city_hints: Optional[Dict[str, dict]] = None,
) -> str:
    """Prompt classifying a batch of group-phrase + weak-cluster candidates.

    city_hints: optional dict mapping phrase (lowercase) to a CityContext-like
    dict with keys {matched_canonical, city, confidence, rationale}.  When
    present and confidence is in the advisory range (0.40–0.84), the hint is
    embedded in the phrase item so Gemini can factor it in.
    """
    bno_map = {b.get("booking_no"): b for b in bookings}
    city_hints = city_hints or {}

    items = []
    for c in phrase_cands:
        contexts = []
        for bno in c.booking_nos[:3]:
            b = bno_map.get(bno, {})
            sections = b.get("request", {}).get("preferred_sections", [])
            raw = " | ".join(filter(None, [
                b.get("raw_guest_message", ""),
                b.get("raw_comment", ""),
                b.get("raw_location_wish", ""),
            ]))[:200]
            contexts.append({
                "booking_no": bno,
                "section": sections[0] if sections else "",
                "raw_text": raw,
            })
        item: dict = {
            "id": c.phrase,
            "phrase": c.phrase,
            "raw_variants": c.raw_variants,
            "frequency": c.frequency,
            "source_fields": c.source_fields,
            "booking_contexts": contexts,
        }
        hint = city_hints.get(c.phrase)
        if hint and 0.40 <= hint.get("confidence", 0.0) < 0.85:
            item["city_hint"] = {
                "booking_city": hint.get("city", ""),
                "suggested_canonical": hint.get("matched_canonical"),
                "confidence": hint.get("confidence", 0.0),
                "rationale": hint.get("rationale", ""),
            }
        items.append(item)

    for wc in cluster_cands:
        items.append({
            "id": wc.canonical_label.lower(),
            "phrase": wc.canonical_label.lower(),
            "raw_variants": [wc.canonical_label],
            "frequency": len(wc.member_booking_nos),
            "source_fields": ["cluster_label"],
            "booking_contexts": [],
        })

    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["id", "classification", "suggested_canonical",
                         "confidence", "reasoning"],
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Must match the 'id' field of the input item.",
                },
                "classification": {
                    "type": "string",
                    "enum": ["new_group_alias", "known_group_variant", "place_name",
                             "preference_language", "person_name", "noise"],
                },
                "suggested_canonical": {
                    "type": ["string", "null"],
                    "description": (
                        "Proper-cased canonical name for new_group_alias / "
                        "known_group_variant / place_name.  null for noise or person_name."
                    ),
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reasoning": {"type": "string"},
            },
        },
    }

    return (
        "Classify each phrase below.\n\n"
        "Rules:\n"
        "- Phrases starting with preference verbs (foretrekker, ønsker, vil bo, "
        "helst, gjerne) → 'preference_language'\n"
        "- Norwegian church names (Pinsemenigheten, Betania, Betel, Filadelfia, "
        "Sion, Pinsekirken) → 'new_group_alias' or 'known_group_variant'\n"
        "- 'camp X' patterns → 'new_group_alias' (Pentecostal youth camps)\n"
        "- 'X gjengen' / 'X gruppa' → 'new_group_alias' (local social group)\n"
        "- Section names that slipped through (Furulunden, Elvebredden…) → 'place_name'\n"
        "- Two-word phrase where last word ends in surname suffix "
        "(-sen, -vik, -haug, -strand, -lund, -berg, -rud) → likely 'person_name'\n"
        "- When a 'city_hint' is present on an item, treat it as a strong signal "
        "for the suggested_canonical — raise your confidence accordingly.\n"
        "- Return one result object per input item, with 'id' matching the input.\n\n"
        f"Input:\n{json.dumps(items, ensure_ascii=False, indent=2)}\n\n"
        f"Return a JSON array:\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )


# ---------------------------------------------------------------------------
# Selector 2: near-text reference resolution
# ---------------------------------------------------------------------------

def build_near_text_prompt(
    cand: NearTextCandidate,
    roster: List[dict],
    place_context: Optional[List[dict]] = None,
) -> str:
    """Prompt resolving one booking's near-text fragments against a guest roster.

    place_context: optional list of PlaceRef-like dicts from the deterministic
    place resolver; injected into the prompt so Gemini can leverage pre-resolved
    city matches.
    """
    schema = {
        "type": "object",
        "required": ["booking_no", "resolved_refs", "unresolved_fragments", "notes"],
        "properties": {
            "booking_no": {"type": "string"},
            "resolved_refs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["raw_fragment", "matched_booking_no",
                                 "match_type", "confidence"],
                    "properties": {
                        "raw_fragment": {"type": "string"},
                        "matched_booking_no": {"type": ["string", "null"]},
                        "match_type": {
                            "type": "string",
                            "enum": ["full_name", "surname", "family",
                                     "first_name", "group_reference", "unresolved"],
                        },
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                },
            },
            "unresolved_fragments": {
                "type": "array", "items": {"type": "string"},
            },
            "notes": {"type": "string"},
        },
    }

    place_section = ""
    if place_context:
        place_section = (
            "\nPre-resolved place references (deterministic, city-based):\n"
            + json.dumps(place_context, ensure_ascii=False, indent=2)
            + "\n- For fragments already in this list with confidence >= 0.65, "
            "prefer 'group_reference' as match_type and use the matched_booking_nos "
            "as your answer unless you see a strong reason not to.\n"
        )

    return (
        f"Booking {cand.booking_no} ({cand.full_name}, "
        f"{cand.check_in}–{cand.check_out}) wrote they want to stay near:\n"
        f"{json.dumps([t[:_MAX_RAW_TEXT_LEN] for t in cand.raw_near_texts], ensure_ascii=False)}\n\n"
        "From the co-attending guest list below, identify which booking each "
        "fragment refers to.\n"
        "- Use 'group_reference' for fragments referring to a group (not one person)\n"
        "- Use 'unresolved' when you cannot match confidently\n"
        "- Do not match on first name alone if multiple guests share it\n"
        + place_section
        + f"\nCo-attending guests:\n{json.dumps(roster, ensure_ascii=False, indent=2)}\n\n"
        f"Return JSON:\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )


# ---------------------------------------------------------------------------
# Selector 4: preference extraction
# ---------------------------------------------------------------------------

def build_preference_prompt(candidates: List[PreferenceCandidate]) -> str:
    """Prompt structuring preference text for a batch of bookings."""
    items = [
        {
            "booking_no": c.booking_no,
            "full_name": c.full_name,
            "raw_text": c.raw_text[:_MAX_RAW_TEXT_LEN],
            "already_extracted_sections": c.extracted_sections,
            "detected_signals": c.missing_signals,
        }
        for c in candidates
    ]

    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["booking_no", "preferences", "confidence"],
            "properties": {
                "booking_no": {"type": "string"},
                "preferences": {
                    "type": "object",
                    "properties": {
                        "avoid_river":        {"type": "boolean"},
                        "avoid_noise":        {"type": "boolean"},
                        "same_as_last_year":  {"type": "boolean"},
                        "extra_space":        {"type": "boolean"},
                        "near_bibelskolen":   {"type": "boolean"},
                        "near_hall":          {"type": "boolean"},
                        "flat_ground":        {"type": "boolean"},
                        "terrain_pref":       {"type": "string"},
                        "near_toilet":        {"type": "boolean"},
                        "near_forest":        {"type": "boolean"},
                        "quiet_spot":         {"type": "boolean"},
                        "drainage_concern":   {"type": "boolean"},
                        "accessibility":      {"type": "boolean"},
                        "inferred_section":   {"type": "string"},
                        "notes":              {"type": "string"},
                    },
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
    }

    return (
        "Extract structured campsite placement preferences from each booking's "
        "Norwegian text.\n\n"
        "Guidelines:\n"
        "- 'i fjor' / 'samme plass' / 'de siste X år' → same_as_last_year=true\n"
        "- 'vekke fra elva' / 'lengst vekke' → avoid_river=true\n"
        "- 'flat plass' / 'flatt underlag' → flat_ground=true\n"
        "- 'bibelskolen' → near_bibelskolen=true\n"
        "- 'toalett' / 'wc' → near_toilet=true\n"
        "- 'drenering' / 'bekk igjennom' → drainage_concern=true\n"
        "- 'begrenset førlighet' / 'rullator' / 'rullestol' → accessibility=true\n"
        "- 'nedoverbakke' → terrain_pref='downhill'; 'i høyden' → terrain_pref='uphill'\n"
        "- Section name in text not already in already_extracted_sections → inferred_section\n"
        "- Return one result per booking_no.\n\n"
        f"Input:\n{json.dumps(items, ensure_ascii=False, indent=2)}\n\n"
        f"Return a JSON array:\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )


# ---------------------------------------------------------------------------
# Selector 5: subsection / spot-range resolution
# ---------------------------------------------------------------------------

def build_subsection_prompt(candidates: List[SubsectionCandidate]) -> str:
    """Prompt resolving row and spot preferences for a batch of bookings."""
    items = [
        {
            "booking_no": c.booking_no,
            "full_name": c.full_name,
            "raw_text": c.raw_text[:_MAX_RAW_TEXT_LEN],
            "extracted_section": c.extracted_section,
            "already_captured_rows": c.already_captured_rows,
            "unresolved_patterns": c.unresolved_patterns,
        }
        for c in candidates
    ]

    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["booking_no", "suggested_rows", "suggested_spot_ids",
                         "confidence", "notes"],
            "properties": {
                "booking_no": {"type": "string"},
                "suggested_rows": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Row letters, e.g. ['D', 'E'].",
                },
                "suggested_spot_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific spot IDs, e.g. ['D25', 'D26', 'D27'].",
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "notes": {"type": "string"},
            },
        },
    }

    return (
        "Resolve row and spot preferences from each booking's Norwegian text.\n\n"
        "Guidelines:\n"
        "- 'D eller E' / 'D evt E' → suggested_rows=['D', 'E']\n"
        "- 'felt A' → suggested_rows=['A']\n"
        "- 'D25-D27' or 'D25-27' → suggested_spot_ids=['D25', 'D26', 'D27']\n"
        "- 'E07-09' → suggested_spot_ids=['E07', 'E08', 'E09']\n"
        "- 'E01 til 12' → suggested_spot_ids=['E01', 'E02', ..., 'E12']\n"
        "- Rows already in already_captured_rows are resolved — do not repeat them\n"
        "- Return one result per booking_no.\n\n"
        f"Input:\n{json.dumps(items, ensure_ascii=False, indent=2)}\n\n"
        f"Return a JSON array:\n{json.dumps(schema, ensure_ascii=False, indent=2)}"
    )
