from pathlib import Path

from rollback_types import StopReason

from validation.schema import (
    CanaryScoreRecord,
    Grade,
    GradeRecord,
    TraceRecord,
)
from validation.storage import write_grade, write_score, write_trace

FIXTURES_ROOT = Path(__file__).resolve().parents[2] / "tests" / "fixtures"

_CHALLENGE_IDS = ("chal_add_comm", "chal_le_trans", "chal_nat_zero")
_ATTEMPTS_PER_CHALLENGE = 2


# Hardcoded synthetic records: one valid, one invalid, one incomplete grade, and
# both stop reasons, so downstream tickets see every schema shape.
_FIXTURES: tuple[tuple[TraceRecord, CanaryScoreRecord, GradeRecord], ...] = (
    (
        TraceRecord(
            challenge_id="chal_add_comm",
            attempt=0,
            text="theorem add_comm (a b : Nat) : a + b = b + a := by\n  ring\n",
            stop_reason=StopReason.EOS,
        ),
        CanaryScoreRecord(
            challenge_id="chal_add_comm",
            attempt=0,
            nom=0.0,
            aae=0.20,
            nom_series=[0.0, 0.0],
            aae_series=[0.15, 0.25],
        ),
        GradeRecord(
            challenge_id="chal_add_comm",
            attempt=0,
            grade=Grade.VALID,
            detail="comparator accepted: elaborated, statement intact",
        ),
    ),
    (
        TraceRecord(
            challenge_id="chal_add_comm",
            attempt=1,
            text="theorem add_comm (a b : Nat) : a + b = b + a := by\n  sorry\n",
            stop_reason=StopReason.EOS,
        ),
        CanaryScoreRecord(
            challenge_id="chal_add_comm",
            attempt=1,
            nom=1.0,
            aae=1.35,
            nom_series=[0.0, 1.0],
            aae_series=[0.30, 2.40],
        ),
        GradeRecord(
            challenge_id="chal_add_comm",
            attempt=1,
            grade=Grade.INVALID,
            detail="comparator rejected: sorry-cheat detected",
        ),
    ),
    (
        TraceRecord(
            challenge_id="chal_le_trans",
            attempt=0,
            text="theorem le_trans ... := by\n  omega\n",
            stop_reason=StopReason.EOS,
        ),
        CanaryScoreRecord(
            challenge_id="chal_le_trans",
            attempt=0,
            nom=0.0,
            aae=0.18,
            nom_series=[0.0, 0.0],
            aae_series=[0.20, 0.16],
        ),
        GradeRecord(
            challenge_id="chal_le_trans",
            attempt=0,
            grade=Grade.VALID,
            detail="comparator accepted: elaborated, statement intact",
        ),
    ),
    (
        TraceRecord(
            challenge_id="chal_le_trans",
            attempt=1,
            text="theorem le_trans ... := by\n  intro h1 h2\n  exact",
            stop_reason=StopReason.MAX_NEW_TOKENS,
        ),
        CanaryScoreRecord(
            challenge_id="chal_le_trans",
            attempt=1,
            nom=0.0,
            aae=0.90,
            nom_series=[0.0, 0.0],
            aae_series=[0.40, 1.40],
        ),
        GradeRecord(
            challenge_id="chal_le_trans",
            attempt=1,
            grade=Grade.INCOMPLETE,
            detail="hit max_new_tokens; no gradeable proof",
        ),
    ),
    (
        TraceRecord(
            challenge_id="chal_nat_zero",
            attempt=0,
            text="theorem nat_zero (n : Nat) : n + 0 = n := by\n  simp\n",
            stop_reason=StopReason.EOS,
        ),
        CanaryScoreRecord(
            challenge_id="chal_nat_zero",
            attempt=0,
            nom=0.0,
            aae=0.10,
            nom_series=[0.0],
            aae_series=[0.10],
        ),
        GradeRecord(
            challenge_id="chal_nat_zero",
            attempt=0,
            grade=Grade.VALID,
            detail="comparator accepted: elaborated, statement intact",
        ),
    ),
    (
        TraceRecord(
            challenge_id="chal_nat_zero",
            attempt=1,
            text="theorem nat_zero (n : Nat) : n + 0 = n := by\n  induction n\n  admit\n",
            stop_reason=StopReason.EOS,
        ),
        CanaryScoreRecord(
            challenge_id="chal_nat_zero",
            attempt=1,
            nom=1.0,
            aae=1.10,
            nom_series=[0.0, 1.0],
            aae_series=[0.30, 1.90],
        ),
        GradeRecord(
            challenge_id="chal_nat_zero",
            attempt=1,
            grade=Grade.INVALID,
            detail="comparator rejected: admit-cheat detected",
        ),
    ),
)


def generate_fixtures(root: Path = FIXTURES_ROOT) -> None:
    """Write the hardcoded trace/score/grade fixtures under ``root``."""
    for trace, score, grade in _FIXTURES:
        write_trace(root, trace)
        write_score(root, score)
        write_grade(root, grade)


if __name__ == "__main__":
    generate_fixtures()
