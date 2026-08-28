from pathlib import Path

import numpy as np
from rollback_types import StopReason

from validation.aggregate import aae_trace, nom_trace
from validation.schema import (
    CanaryScoreRecord,
    Grade,
    GradeRecord,
    Segment,
    TraceRecord,
)
from validation.storage import write_grade, write_score, write_trace

FIXTURES_ROOT = Path(__file__).resolve().parents[2] / "tests" / "fixtures"

_CHALLENGE_IDS = ("chal_add_comm", "chal_le_trans", "chal_nat_zero")
_ATTEMPTS_PER_CHALLENGE = 2
_CENTROID_DIM = 8
_SEED = 20260828


def _make_segment(rng: np.random.Generator, smell: bool) -> Segment:
    return Segment(
        text="  intro h\n  simp\n",
        centroid=rng.standard_normal(_CENTROID_DIM),
        mean_entropy=float(rng.uniform(0.2, 1.5)),
        smell=smell,
    )


def _build_records(
    challenge_id: str,
    attempt: int,
    rng: np.random.Generator,
) -> tuple[TraceRecord, CanaryScoreRecord, GradeRecord]:
    n_segments = int(rng.integers(4, 7))
    smells = rng.random(n_segments) < 0.2
    segments = [_make_segment(rng, bool(smells[i])) for i in range(n_segments)]

    truncated = attempt == _ATTEMPTS_PER_CHALLENGE - 1
    stop_reason = StopReason.MAX_NEW_TOKENS if truncated else StopReason.EOS

    trace = TraceRecord(
        challenge_id=challenge_id,
        attempt=attempt,
        text="theorem t : True := by\n  trivial\n",
        segments=segments,
        stop_reason=stop_reason,
    )

    nom_series = smells.astype(float).tolist()
    aae_series = rng.uniform(0.0, np.pi, n_segments).tolist()
    score = CanaryScoreRecord(
        challenge_id=challenge_id,
        attempt=attempt,
        nom=nom_trace(nom_series),
        aae=aae_trace(aae_series),
        nom_series=nom_series,
        aae_series=aae_series,
    )

    if truncated:
        grade, detail = Grade.INCOMPLETE, "hit max_new_tokens; no gradeable proof"
    elif any(segments[i]["smell"] for i in range(n_segments)):
        grade, detail = Grade.INVALID, "comparator rejected: sorry-cheat detected"
    else:
        grade, detail = Grade.VALID, "comparator accepted: elaborated, statement intact"
    grade_record = GradeRecord(
        challenge_id=challenge_id,
        attempt=attempt,
        grade=grade,
        detail=detail,
    )

    return trace, score, grade_record


def generate_fixtures(root: Path = FIXTURES_ROOT) -> None:
    """Write a small deterministic set of trace/score/grade fixtures under ``root``."""
    rng = np.random.default_rng(_SEED)
    for challenge_id in _CHALLENGE_IDS:
        for attempt in range(_ATTEMPTS_PER_CHALLENGE):
            trace, score, grade_record = _build_records(challenge_id, attempt, rng)
            write_trace(root, trace)
            write_score(root, score)
            write_grade(root, grade_record)


if __name__ == "__main__":
    generate_fixtures()
