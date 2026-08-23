import numpy as np
import pytest
from detect import (
    CANARIES,
    average_angle_error_canary,
    get_canary,
    resolve_canary,
    smell_canary,
)


def make_segment(smell: bool = False, mean_entropy: float = 1.0, centroid=None):
    return {"smell": smell, "mean_entropy": mean_entropy, "centroid": centroid}


def make_history_entry(mean_entropy: float, centroid=None):
    return {"mean_entropy": mean_entropy, "centroid": centroid}


class TestCanaryRegistry:
    def test_smell_is_registered(self):
        assert CANARIES["smell"] is smell_canary

    def test_aae_is_registered(self):
        assert CANARIES["aae"] is average_angle_error_canary

    def test_get_canary_returns_registered_callable(self):
        assert get_canary("smell") is smell_canary

    def test_get_canary_unknown_name_raises(self):
        with pytest.raises(ValueError, match=r"unknown canary 'nope'.*smell"):
            get_canary("nope")

    def test_registry_supports_additional_canaries(self, monkeypatch):
        """The pipeline must be able to select a signal other than 'smell'."""

        def mistake_count_canary(seg, history, warmup=0):
            return float(seg.get("mistakes", 0))

        monkeypatch.setitem(CANARIES, "mistake_count", mistake_count_canary)

        assert get_canary("mistake_count") is mistake_count_canary
        assert "mistake_count" in CANARIES


class TestResolveCanary:
    def test_resolves_registered_name(self):
        name, fn = resolve_canary("smell")
        assert name == "smell"
        assert fn is smell_canary

    def test_resolves_unknown_name_raises(self):
        with pytest.raises(ValueError):
            resolve_canary("nope")

    def test_resolves_custom_callable_by_name(self):
        def my_canary(seg, history, warmup=0):
            return 0.0

        name, fn = resolve_canary(my_canary)
        assert name == "my_canary"
        assert fn is my_canary

    def test_resolves_callable_object_without_dunder_name(self):
        class CallableCanary:
            def __call__(self, seg, history, warmup=0):
                return 0.0

        stub = CallableCanary()  # has no __name__ attribute, unlike a function
        name, fn = resolve_canary(stub)
        assert fn is stub
        assert isinstance(name, str) and name


class TestSmellCanary:
    def test_smell_during_warmup_scores_higher(self):
        smelly = smell_canary(make_segment(smell=True), history=[])
        clean = smell_canary(make_segment(smell=False), history=[])
        assert smelly == pytest.approx(1.6)
        assert clean == pytest.approx(0.1)
        assert smelly > clean

    def test_respects_warmup_parameter(self):
        history = [make_history_entry(mean_entropy=1.0) for _ in range(3)]
        # still within warmup -> short-circuits to the smell-only score
        score = smell_canary(make_segment(smell=True), history, warmup=3)
        assert score == pytest.approx(1.6)

    def test_entropy_outlier_scores_higher_once_past_warmup(self):
        history = [make_history_entry(mean_entropy=1.0) for _ in range(5)]
        normal = smell_canary(make_segment(mean_entropy=1.0), history)
        outlier = smell_canary(make_segment(mean_entropy=10.0), history)
        assert outlier > normal

    def test_centroid_drift_scores_higher_once_past_warmup(self):
        run_centroid = np.array([1.0, 0.0, 0.0])
        history = [
            make_history_entry(mean_entropy=1.0, centroid=run_centroid)
            for _ in range(5)
        ]
        aligned = smell_canary(
            make_segment(mean_entropy=1.0, centroid=run_centroid), history
        )
        drifted = smell_canary(
            make_segment(mean_entropy=1.0, centroid=np.array([0.0, 1.0, 0.0])),
            history,
        )
        assert drifted > aligned

    def test_missing_centroid_falls_back_to_zero_drift(self):
        history = [
            make_history_entry(mean_entropy=1.0, centroid=np.array([1.0, 0.0]))
            for _ in range(3)
        ]
        # should not raise even though this segment has no centroid
        score = smell_canary(make_segment(mean_entropy=1.0, centroid=None), history)
        assert isinstance(score, float)


class TestAverageAngleErrorCanary:
    def test_needs_at_least_three_centroids(self):
        history = [make_history_entry(mean_entropy=1.0, centroid=np.array([1.0, 0.0]))]
        score = average_angle_error_canary(
            make_segment(centroid=np.array([2.0, 0.0])), history
        )
        assert score == pytest.approx(0.0)

    def test_respects_warmup_parameter(self):
        history = [
            make_history_entry(mean_entropy=1.0, centroid=np.array([float(i), 0.0]))
            for i in range(3)
        ]
        score = average_angle_error_canary(
            make_segment(centroid=np.array([3.0, 0.0])), history, warmup=3
        )
        assert score == pytest.approx(0.0)

    def test_straight_trajectory_scores_near_zero(self):
        # centroids walking in a straight line: displacements never turn
        history = [
            make_history_entry(mean_entropy=1.0, centroid=np.array([float(i), 0.0]))
            for i in range(3)
        ]
        score = average_angle_error_canary(
            make_segment(centroid=np.array([3.0, 0.0])), history
        )
        # arccos is numerically steep near 0/pi, so use a looser tolerance here
        # than the exact right-angle case below
        assert score == pytest.approx(0.0, abs=1e-3)

    def test_reversed_trajectory_scores_near_pi(self):
        # centroid doubles back the way it came: a sharp, suspicious turn
        history = [
            make_history_entry(mean_entropy=1.0, centroid=np.array([0.0, 0.0])),
            make_history_entry(mean_entropy=1.0, centroid=np.array([1.0, 0.0])),
        ]
        score = average_angle_error_canary(
            make_segment(centroid=np.array([0.0, 0.0])), history
        )
        assert score == pytest.approx(np.pi, abs=1e-3)

    def test_right_angle_turn_scores_near_half_pi(self):
        history = [
            make_history_entry(mean_entropy=1.0, centroid=np.array([0.0, 0.0])),
            make_history_entry(mean_entropy=1.0, centroid=np.array([1.0, 0.0])),
        ]
        score = average_angle_error_canary(
            make_segment(centroid=np.array([1.0, 1.0])), history
        )
        assert score == pytest.approx(np.pi / 2, abs=1e-6)

    def test_missing_centroid_returns_zero(self):
        history = [
            make_history_entry(mean_entropy=1.0, centroid=np.array([float(i), 0.0]))
            for i in range(3)
        ]
        score = average_angle_error_canary(make_segment(centroid=None), history)
        assert score == pytest.approx(0.0)
