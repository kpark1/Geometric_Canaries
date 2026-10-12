from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np

"""Patterns that can be used to find boundaries in the generated text.

The dictionary keys are names for the patterns so that we can choose which
ones to use and also see which one found a boundary. ``\n`` is a regular
newline, while ``Ċ`` is how a newline may appear when looking at Byte Pair
Encoding tokens. ``find_boundary`` checks every enabled pattern and uses the
earliest match. If patterns match at the same place, the order in ``enabled``
decides which one to use. If ``enabled`` is not specified, the dictionary order
is used.
"""
BOUNDARY_PATTERNS: dict[str, re.Pattern[str]] = {
    "eoseg": re.compile(r"^[ \t]*EOSEG[ \t]*$", re.IGNORECASE | re.MULTILINE),
    "blank_line": re.compile(r"[\nĊ]\s*[\nĊ]"),
    "step": re.compile(r"[\nĊ](?=Step \d)", re.IGNORECASE),
    "numbered": re.compile(r"[\nĊ](?=\d+[.)]\s)"),
    "code_block": re.compile(r"[\nĊ](?=```)"),
    "transition": re.compile(
        r"[\nĊ](?=(?:First|Second|Next|Then|Now|Finally|Therefore|Thus)[,\s])",
        re.IGNORECASE,
    ),
    "legacy_marker": re.compile(r"[\nĊ](?=\s*<;>)"),
}

# prevents extremely short segments
MIN_SEGMENT_TOKENS = 8

# define smells: sorry, admit, ...
LEXICAL_SMELLS = re.compile(
    r"\bsorry\b|\badmit\b|native_decide|\bobviously\b|\btrivially\b"
    r"|clearly\s+true|it is easy to see",
    re.IGNORECASE,
)


# After every token, the code asks whether a segment boundary has appeared
def find_boundary(
    text: str,
    from_char: int,
    enabled: Sequence[str] = tuple(BOUNDARY_PATTERNS),
) -> tuple[int, str] | None:
    """Look for the next boundary in the generated text.

    ``text`` is all of the text generated so far, and ``from_char`` is the
    character position where the search should start. Note that ``from_char``
    is a character position, not a token number. ``enabled`` contains the names
    of the boundary patterns to check.

    Returns:
        ``(boundary_char, kind)`` when a boundary is found.
        ``boundary_char`` is the character position just after the matched boundary,
        and ``kind`` is the name of the pattern that matched. It returns ``None`` if
        none of the enabled patterns find a boundary.
    """
    matches = []
    for priority, kind in enumerate(enabled):
        pattern = BOUNDARY_PATTERNS[kind]
        match = pattern.search(text, from_char)
        if match:
            print(f'{kind=}, {match.start()=}, {match.end()=}')
            matches.append((match.start(), priority, match.end(), kind))

    if matches:
        _, _, boundary_char, kind = min(matches)
        return boundary_char, kind

    return None


def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else 1.0 - float(a @ b) / (na * nb)


def score_segment(
    seg: dict[str, Any],
    history: list[dict[str, Any]],
    warmup: int = 0,
) -> float:
    smell_score = 1.5 if seg["smell"] else 0.0
    if len(history) <= warmup:
        return smell_score + 0.1  # Allow smells to trigger even during warmup

    ents = np.array([h["mean_entropy"] for h in history])
    z_ent = (seg["mean_entropy"] - ents.mean()) / max(float(ents.std()), 0.25)
    cents = [h["centroid"] for h in history if h["centroid"] is not None]
    if seg["centroid"] is not None and cents:
        run = np.mean(cents, axis=0)
        drifts = np.array([cosine_dist(c, run) for c in cents])
        d = cosine_dist(seg["centroid"], run)
        z_drift = (
            (d - drifts.mean()) / max(float(drifts.std()), 0.05)
            if len(drifts) > 1
            else 0.0
        )
    else:
        z_drift = 0.0
    return 0.5 * z_ent + 0.5 * z_drift + smell_score
