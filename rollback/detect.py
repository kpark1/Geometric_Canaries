import re
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

# Handle both standard newlines and byte-level BPE newlines ('Ċ')
BOUNDARY_PATTERN = re.compile(
    r"[\nĊ]\s*[\nĊ]"
    r"|[\nĊ](?=Step \d)"
    r"|[\nĊ](?=\d+[\.)]\s)"
    r"|[\nĊ](?=```)"
    r"|[\nĊ](?=(?:First|Second|Next|Then|Now|Finally|Therefore|Thus)[,\s])"
    r"|[\nĊ](?=\s*<;>)",
    flags=re.IGNORECASE,
)

# prevents extremely short segments
MIN_SEGMENT_TOKENS = 8

# define smells: sorry, admit, ...
LEXICAL_SMELLS = re.compile(
    r"\bsorry\b|\badmit\b|native_decide|\bobviously\b|\btrivially\b"
    r"|clearly\s+true|it is easy to see",
    re.IGNORECASE,
)


def _load_backtrack_markers(path: Path) -> re.Pattern:
    """Compile backtrack_markers.yaml into one pattern (see that file for sourcing).

    Single-word markers get word boundaries so they don't match inside unrelated
    words (e.g. "wait" shouldn't match "await"); multi-word phrases are matched
    as plain substrings, since they're already specific enough on their own.
    """
    with path.open(encoding="utf-8") as f:
        tiers = yaml.safe_load(f)
    markers = [marker for tier in tiers.values() for marker in tier]
    parts = [re.escape(m) if " " in m else rf"\b{re.escape(m)}\b" for m in markers]
    return re.compile("|".join(parts), re.IGNORECASE)


BACKTRACK_MARKERS = _load_backtrack_markers(
    Path(__file__).with_name("backtrack_markers.yaml")
)


# After every token, the code asks whether a segment boundary has appeared
def find_boundary(
    text: str,
    from_char: int,
    boundary_pattern: re.Pattern = BOUNDARY_PATTERN,
) -> int | None:
    m = boundary_pattern.search(text, from_char)
    if m:
        return m.end()

    nl1 = text.find("\n", from_char + 100)
    nl2 = text.find("Ċ", from_char + 100)
    nls = [n for n in (nl1, nl2) if n != -1]
    return min(nls) + 1 if nls else None


def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else 1.0 - float(a @ b) / (na * nb)


class Canary(Protocol):
    """A signal that scores a closed segment for how strongly it should be distrusted.

    Higher scores mean more suspicious; `prove()` flags a segment for intervention
    when a canary's score clears its `thresh`.
    """

    def __call__(
        self,
        seg: dict[str, Any],
        history: list[dict[str, Any]],
        warmup: int = 0,
    ) -> float: ...


def smell_canary(
    seg: dict[str, Any],
    history: list[dict[str, Any]],
    warmup: int = 0,
) -> float:
    """Score a segment from lexical smells plus entropy/embedding drift vs. history."""
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


def _unit(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return v / (np.linalg.norm(v) + eps)


def average_angle_error_canary(
    seg: dict[str, Any],
    history: list[dict[str, Any]],
    warmup: int = 0,
) -> float:
    """Angle (radians) between the two most recent segment-centroid displacements.

    Ported from geocanary/geometry/metrics.py (PR #1). That version averages
    the angle over a whole finished trace; this scores online instead, using
    each closed segment's centroid as one step embedding and reporting how
    sharply the trajectory just turned relative to where it was heading.
    Confident reasoning moves fairly directly (small angle); misled reasoning
    zig-zags (large angle, up to pi).
    """
    if seg["centroid"] is None or len(history) <= warmup:
        return 0.0

    centroids = [h["centroid"] for h in history if h["centroid"] is not None]
    centroids.append(seg["centroid"])
    if len(centroids) < 3:
        return 0.0

    v_prev = _unit(centroids[-2] - centroids[-3])
    v_last = _unit(centroids[-1] - centroids[-2])
    cosine = float(np.dot(v_prev, v_last))
    return float(np.arccos(max(-1.0, min(1.0, cosine))))


def number_of_mistakes_canary(
    seg: dict[str, Any],
    history: list[dict[str, Any]],
    warmup: int = 0,
    jump_percentile: float = 75.0,
) -> float:
    """1.0 if this segment both names a caught mistake and jumps embedding-wise.

    Ported from geocanary/geometry/metrics.py (PR #1): a segment counts as a
    caught mistake only when a backtrack marker co-occurs with a centroid jump
    large relative to the run so far -- requiring both avoids counting a
    reflection word with no real pivot behind it.
    """
    if len(history) <= warmup or not BACKTRACK_MARKERS.search(seg["text"]):
        return 0.0

    prev_centroid = next(
        (h["centroid"] for h in reversed(history) if h["centroid"] is not None),
        None,
    )
    if prev_centroid is None or seg["centroid"] is None:
        return 0.0

    dists = [
        cosine_dist(a["centroid"], b["centroid"])
        for a, b in pairwise(history)
        if a["centroid"] is not None and b["centroid"] is not None
    ]
    jump = cosine_dist(seg["centroid"], prev_centroid)
    if not dists:
        return 0.0

    return 1.0 if jump >= float(np.percentile(dists, jump_percentile)) else 0.0


# Canaries selectable by the pipeline, keyed by the name passed to `prove(canary=...)`.
CANARIES: dict[str, Canary] = {
    "smell": smell_canary,
    "aae": average_angle_error_canary,
    "nom": number_of_mistakes_canary,
}


def get_canary(name: str) -> Canary:
    try:
        return CANARIES[name]
    except KeyError:
        raise ValueError(
            f"unknown canary {name!r}; available: {sorted(CANARIES)}"
        ) from None


def resolve_canary(canary: str | Canary) -> tuple[str, Canary]:
    """Resolve a registered name or a caller-supplied callable to (name, callable)."""
    if isinstance(canary, str):
        return canary, get_canary(canary)
    return getattr(canary, "__name__", repr(canary)), canary
