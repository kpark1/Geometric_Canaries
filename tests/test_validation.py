import math

import numpy as np
from rollback_types import StopReason
from validation.fixtures import (
    _ATTEMPTS_PER_CHALLENGE,
    _CHALLENGE_IDS,
    FIXTURES_ROOT,
    generate_fixtures,
)
from validation.schema import (
    CanaryScoreRecord,
    Grade,
    GradeRecord,
    Segment,
    TraceRecord,
)
from validation.storage import (
    read_grade,
    read_score,
    read_trace,
    write_grade,
    write_score,
    write_trace,
)


def _sample_trace():
    return TraceRecord(
        challenge_id="chal_x",
        attempt=3,
        text="theorem t : True := by trivial",
        segments=[
            Segment(
                text="intro h",
                centroid=np.array([0.1, -0.2, 0.3], dtype=np.float64),
                mean_entropy=0.75,
                smell=False,
            ),
            Segment(text="sorry", centroid=None, mean_entropy=1.4, smell=True),
        ],
        stop_reason=StopReason.EOS,
    )


def _segments_equal(a: Segment, b: Segment) -> bool:
    if a["text"] != b["text"] or a["smell"] != b["smell"]:
        return False
    if not math.isclose(a["mean_entropy"], b["mean_entropy"], rel_tol=1e-9):
        return False
    if a["centroid"] is None or b["centroid"] is None:
        return a["centroid"] is None and b["centroid"] is None
    return bool(np.allclose(a["centroid"], b["centroid"]))


def test_trace_round_trip(tmp_path):
    trace = _sample_trace()
    write_trace(tmp_path, trace)
    loaded = read_trace(tmp_path, trace["challenge_id"], trace["attempt"])

    assert loaded["challenge_id"] == trace["challenge_id"]
    assert loaded["attempt"] == trace["attempt"]
    assert loaded["text"] == trace["text"]
    assert loaded["stop_reason"] == trace["stop_reason"]
    assert len(loaded["segments"]) == len(trace["segments"])
    assert all(
        _segments_equal(x, y)
        for x, y in zip(loaded["segments"], trace["segments"], strict=True)
    )


def test_score_round_trip(tmp_path):
    score = CanaryScoreRecord(
        challenge_id="chal_x",
        attempt=1,
        nom=2.0,
        aae=1.25,
        nom_series=[1.0, 0.0, 1.0],
        aae_series=[0.5, 2.0, 1.25],
    )
    write_score(tmp_path, score)
    loaded = read_score(tmp_path, score["challenge_id"], score["attempt"])
    assert loaded == score


def test_grade_round_trip(tmp_path):
    grade = GradeRecord(
        challenge_id="chal_x",
        attempt=0,
        grade=Grade.INCOMPLETE,
        detail="hit max_new_tokens",
    )
    write_grade(tmp_path, grade)
    loaded = read_grade(tmp_path, grade["challenge_id"], grade["attempt"])
    assert loaded == grade


def test_committed_fixtures_load_and_validate():
    assert FIXTURES_ROOT.exists(), "run `python -m validation.fixtures` to regenerate"
    for challenge_id in _CHALLENGE_IDS:
        for attempt in range(_ATTEMPTS_PER_CHALLENGE):
            trace = read_trace(FIXTURES_ROOT, challenge_id, attempt)
            score = read_score(FIXTURES_ROOT, challenge_id, attempt)
            grade = read_grade(FIXTURES_ROOT, challenge_id, attempt)

            assert trace["challenge_id"] == challenge_id
            assert trace["attempt"] == attempt
            assert isinstance(trace["stop_reason"], StopReason)
            assert trace["segments"]
            assert isinstance(grade["grade"], Grade)
            assert len(score["nom_series"]) == len(trace["segments"])
            assert len(score["aae_series"]) == len(trace["segments"])


def test_fixtures_regenerate_matches_committed(tmp_path):
    generate_fixtures(tmp_path)
    for challenge_id in _CHALLENGE_IDS:
        for attempt in range(_ATTEMPTS_PER_CHALLENGE):
            fresh = read_trace(tmp_path, challenge_id, attempt)
            committed = read_trace(FIXTURES_ROOT, challenge_id, attempt)
            assert fresh["text"] == committed["text"]
            assert fresh["stop_reason"] == committed["stop_reason"]
            assert len(fresh["segments"]) == len(committed["segments"])
