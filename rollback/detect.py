import re
from typing import Any

import numpy as np

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
    r"|clearly\s+true|it is easy to see", re.IGNORECASE)


# After every token, the code asks whether a segment boundary has appeared
def find_boundary(
    text: str,
    from_char: int,
    boundary_pattern: re.Pattern = BOUNDARY_PATTERN,
) -> int | None:
    m = boundary_pattern.search(text, from_char)
    if m:
        return m.end()

    nl1 = text.find('\n', from_char + 100)
    nl2 = text.find('Ċ', from_char + 100)
    nls = [n for n in (nl1, nl2) if n != -1]
    return min(nls) + 1 if nls else None


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
        return smell_score + 0.1 # Allow smells to trigger even during warmup

    ents = np.array([h["mean_entropy"] for h in history])
    z_ent = (seg["mean_entropy"] - ents.mean()) / max(float(ents.std()), 0.25)
    cents = [h["centroid"] for h in history if h["centroid"] is not None]
    if seg["centroid"] is not None and cents:
        run = np.mean(cents, axis=0)
        drifts = np.array([cosine_dist(c, run) for c in cents])
        d = cosine_dist(seg["centroid"], run)
        z_drift = ((d - drifts.mean()) / max(float(drifts.std()), 0.05)
                   if len(drifts) > 1 else 0.0)
    else:
        z_drift = 0.0
    return 0.5 * z_ent + 0.5 * z_drift + smell_score
