import numpy as np
import pytest
from detect import (
    BACKTRACK_MARKERS,
    CANARIES,
    average_angle_error_canary,
    get_canary,
    number_of_mistakes_canary,
    resolve_canary,
    smell_canary,
)


def make_segment(
    smell: bool = False,
    mean_entropy: float = 1.0,
    centroid=None,
    text: str = "",
):
    return {
        "smell": smell,
        "mean_entropy": mean_entropy,
        "centroid": centroid,
        "text": text,
    }


def make_history_entry(mean_entropy: float, centroid=None):
    return {"mean_entropy": mean_entropy, "centroid": centroid}


class TestCanaryRegistry:
    def test_smell_is_registered(self):
        assert CANARIES["smell"] is smell_canary

    def test_aae_is_registered(self):
        assert CANARIES["aae"] is average_angle_error_canary

    def test_nom_is_registered(self):
        assert CANARIES["nom"] is number_of_mistakes_canary

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


def mixed_jump_history():
    """Four segments whose consecutive centroids drift a little, not a lot --
    enough for number_of_mistakes_canary's jump-percentile threshold to sit
    above zero, so "small jump" and "big jump" test cases are unambiguous."""
    return [
        make_history_entry(mean_entropy=1.0, centroid=np.array([1.0, 0.0])),
        make_history_entry(mean_entropy=1.0, centroid=np.array([1.0, 0.1])),
        make_history_entry(mean_entropy=1.0, centroid=np.array([1.0, -0.1])),
        make_history_entry(mean_entropy=1.0, centroid=np.array([1.0, 0.1])),
    ]


class TestNumberOfMistakesCanary:
    def test_no_marker_scores_zero_even_with_a_jump(self):
        history = mixed_jump_history()
        score = number_of_mistakes_canary(
            make_segment(centroid=np.array([0.0, 1.0]), text="so we conclude x=1"),
            history,
        )
        assert score == pytest.approx(0.0)

    def test_marker_without_a_jump_scores_zero(self):
        history = mixed_jump_history()
        score = number_of_mistakes_canary(
            make_segment(
                centroid=np.array([1.0, 0.1]),  # same direction as the last centroid
                text="wait, let's double check",
            ),
            history,
        )
        assert score == pytest.approx(0.0)

    def test_marker_with_a_jump_scores_one(self):
        history = mixed_jump_history()
        score = number_of_mistakes_canary(
            make_segment(
                centroid=np.array([0.0, 1.0]),  # sharply off the run's direction
                text="wait, that's not right",
            ),
            history,
        )
        assert score == pytest.approx(1.0)

    def test_respects_warmup_parameter(self):
        history = mixed_jump_history()
        score = number_of_mistakes_canary(
            make_segment(centroid=np.array([0.0, 1.0]), text="wait, that's not right"),
            history,
            warmup=len(history),
        )
        assert score == pytest.approx(0.0)

    def test_missing_centroid_scores_zero(self):
        history = mixed_jump_history()
        score = number_of_mistakes_canary(
            make_segment(centroid=None, text="wait, that's not right"), history
        )
        assert score == pytest.approx(0.0)

    def test_no_jump_distribution_yet_scores_zero(self):
        history = [make_history_entry(mean_entropy=1.0, centroid=np.array([1.0, 0.0]))]
        score = number_of_mistakes_canary(
            make_segment(centroid=np.array([0.0, 1.0]), text="wait, that's not right"),
            history,
        )
        assert score == pytest.approx(0.0)

    def test_generic_discourse_connectives_are_not_markers(self):
        """ "However"/"instead" are excluded on purpose -- see BACKTRACK_MARKERS."""
        history = mixed_jump_history()
        score = number_of_mistakes_canary(
            make_segment(
                centroid=np.array([0.0, 1.0]),
                text="However, instead of that approach, we try another way",
            ),
            history,
        )
        assert score == pytest.approx(0.0)

    def test_explicit_error_specific_phrase_is_a_mistake(self):
        """ "sign error" isn't in the grounded research list -- it's from the
        explicit tier in backtrack_markers.yaml, specific enough to be low-noise."""
        history = mixed_jump_history()
        score = number_of_mistakes_canary(
            make_segment(
                centroid=np.array([0.0, 1.0]),
                text="that's a sign error in the last step",
            ),
            history,
        )
        assert score == pytest.approx(1.0)


class TestBacktrackMarkers:
    """Loader-level checks on the compiled pattern from backtrack_markers.yaml."""

    @pytest.mark.parametrize(
        "text",
        [
            "Wait, that's not right.",
            "hmm, let me look at this again",
            "actually, no, that's a sign error",
            "let me reconsider this step",
            "we ran into a problem with the substitution",
        ],
    )
    def test_matches_known_markers(self, text):
        assert BACKTRACK_MARKERS.search(text)

    @pytest.mark.parametrize(
        "text",
        [
            "However, instead of that approach, we proceed directly.",
            "Actually this follows immediately from the definition.",
            "The error bound is within tolerance.",
            "We are still awaiting the next step.",  # shouldn't match "wait"
        ],
    )
    def test_does_not_match_generic_prose(self, text):
        assert not BACKTRACK_MARKERS.search(text)
