import json
import os
from pathlib import Path
from typing import Any

from rollback_types import StopReason

from validation.schema import (
    CanaryScoreRecord,
    Grade,
    GradeRecord,
    TraceRecord,
)

RESULTS_ROOT_ENV = "GEOMETRIC_CANARIES_RESULTS_ROOT"
DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results"


def results_root() -> Path:
    """Return the results root, honouring the override env var (gitignored on disk)."""
    override = os.environ.get(RESULTS_ROOT_ENV)
    return Path(override) if override else DEFAULT_RESULTS_ROOT


def trace_path(root: Path, challenge_id: str, attempt: int) -> Path:
    return root / "traces" / challenge_id / f"{attempt}.json"


def score_path(root: Path, challenge_id: str, attempt: int) -> Path:
    return root / "scores" / challenge_id / f"{attempt}.json"


def grade_path(root: Path, challenge_id: str, attempt: int) -> Path:
    return root / "grades" / challenge_id / f"{attempt}.json"


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_trace(root: Path, trace: TraceRecord) -> Path:
    payload: dict[str, Any] = {
        "challenge_id": trace["challenge_id"],
        "attempt": trace["attempt"],
        "text": trace["text"],
        "stop_reason": trace["stop_reason"].value,
    }
    return _write_json(
        trace_path(root, trace["challenge_id"], trace["attempt"]), payload
    )


def read_trace(root: Path, challenge_id: str, attempt: int) -> TraceRecord:
    data = _read_json(trace_path(root, challenge_id, attempt))
    return TraceRecord(
        challenge_id=data["challenge_id"],
        attempt=data["attempt"],
        text=data["text"],
        stop_reason=StopReason(data["stop_reason"]),
    )


def write_score(root: Path, score: CanaryScoreRecord) -> Path:
    return _write_json(
        score_path(root, score["challenge_id"], score["attempt"]), dict(score)
    )


def read_score(root: Path, challenge_id: str, attempt: int) -> CanaryScoreRecord:
    data = _read_json(score_path(root, challenge_id, attempt))
    return CanaryScoreRecord(
        challenge_id=data["challenge_id"],
        attempt=data["attempt"],
        nom=data["nom"],
        aae=data["aae"],
        nom_series=data["nom_series"],
        aae_series=data["aae_series"],
    )


def write_grade(root: Path, grade: GradeRecord) -> Path:
    payload: dict[str, Any] = {
        "challenge_id": grade["challenge_id"],
        "attempt": grade["attempt"],
        "grade": grade["grade"].value,
        "detail": grade["detail"],
    }
    return _write_json(
        grade_path(root, grade["challenge_id"], grade["attempt"]), payload
    )


def read_grade(root: Path, challenge_id: str, attempt: int) -> GradeRecord:
    data = _read_json(grade_path(root, challenge_id, attempt))
    return GradeRecord(
        challenge_id=data["challenge_id"],
        attempt=data["attempt"],
        grade=Grade(data["grade"]),
        detail=data["detail"],
    )
