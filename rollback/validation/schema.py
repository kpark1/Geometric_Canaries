from enum import StrEnum
from typing import TypedDict

import numpy as np
from rollback_types import StopReason


class Segment(TypedDict):
    """Per-segment features a post-hoc canary consumes.

    Shape matches the online canaries' input so a stored trace can be scored
    without regeneration.
    """

    text: str
    centroid: np.ndarray | None
    mean_entropy: float
    smell: bool


class TraceRecord(TypedDict):
    """Output of a vanilla (no-intervention) generation attempt.

    Carries enough to both score canaries post-hoc (``segments``) and grade the
    proof (``text``, ``stop_reason``).
    """

    challenge_id: str
    attempt: int
    text: str
    segments: list[Segment]
    stop_reason: StopReason


class CanaryScoreRecord(TypedDict):
    """Post-hoc canary scores for one trace, keyed by (challenge_id, attempt).

    ``nom`` and ``aae`` are trace-level aggregates of the per-segment series.
    """

    challenge_id: str
    attempt: int
    nom: float
    aae: float
    nom_series: list[float]
    aae_series: list[float]


class Grade(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    INCOMPLETE = "incomplete"


class GradeRecord(TypedDict):
    """Three-way grade for one trace, keyed by (challenge_id, attempt)."""

    challenge_id: str
    attempt: int
    grade: Grade
    detail: str
