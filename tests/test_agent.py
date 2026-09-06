from functools import partial

from agent import agent_prove
from evaluate import run_eval
from verina import build_prompt
from rollback_types import LeanStatus, ProveRunResult


def lean4_block(body: str) -> str:
    return f"plan...\n```lean4\n{body}\n```"


def make_fake_prove(texts: list[str]):
    """Fake prove_fn that returns texts[turn - 1] and records every call's kwargs."""
    calls: list[dict] = []

    def fake_prove(prompt, **kwargs):
        calls.append(kwargs)
        turn = len(calls)
        return {
            "mode": kwargs["mode"],
            "canary": "smell",
            "prompt": prompt,
            "text": texts[turn - 1],
            "ids": [],
            "entropy": [],
            "logprob": [],
            "hidden": None,
            "segments": [],
            "events": [{"type": "noop", "turn": turn}],
            "abandoned": [],
            "rollbacks": turn,
            "n_forward": 10 * turn,
            "n_prefills": 1,
            "gen_tokens": 5 * turn,
            "stop_reason": "eos",
            "wall_s": 0.1 * turn,
            "lean_status": None,
        }

    fake_prove.calls = calls
    return fake_prove


def lean_ok(code: str, timeout_s: float) -> tuple[LeanStatus, str]:
    return LeanStatus.OK, ""


def lean_always_fails(code: str, timeout_s: float) -> tuple[LeanStatus, str]:
    return LeanStatus.ERROR, f"still broken: {code}"


class TestAgentProve:
    def test_succeeds_first_turn(self):
        fake_prove = make_fake_prove([lean4_block("attempt 1")])
        result = agent_prove(
            "prove it",
            model=None,
            tokenizer=None,
            capture_layer=0,
            max_turns=3,
            prove_fn=fake_prove,
            lean_check_fn=lean_ok,
        )
        assert result["turns"] == 1
        assert result["lean_status"] == LeanStatus.OK
        assert result["gen_tokens"] == 5
        assert result["rollbacks"] == 1
        assert len(fake_prove.calls) == 1

    def test_fails_then_succeeds_aggregates_cost_and_grows_conversation(self):
        fake_prove = make_fake_prove(
            [lean4_block("attempt 1"), lean4_block("attempt 2")]
        )

        def fake_lean(code: str, timeout_s: float) -> tuple[LeanStatus, str]:
            if "attempt 1" in code:
                return LeanStatus.ERROR, "type mismatch on line 2"
            return LeanStatus.OK, ""

        result = agent_prove(
            "prove it",
            model=None,
            tokenizer=None,
            capture_layer=0,
            max_turns=3,
            prove_fn=fake_prove,
            lean_check_fn=fake_lean,
        )
        assert result["turns"] == 2
        assert result["lean_status"] == LeanStatus.OK
        # each turn's cost is summed, not just the last turn's
        assert result["gen_tokens"] == 5 + 10
        assert result["n_forward"] == 10 + 20
        assert result["rollbacks"] == 1 + 2
        assert len(result["events"]) == 2
        assert len(fake_prove.calls) == 2

        second_call_messages = fake_prove.calls[1]["messages"]
        assert second_call_messages[0] == {"role": "user", "content": "prove it"}
        assert second_call_messages[1]["role"] == "assistant"
        assert "attempt 1" in second_call_messages[1]["content"]
        assert second_call_messages[2]["role"] == "user"
        assert "type mismatch on line 2" in second_call_messages[2]["content"]

    def test_seed_varies_across_turns(self):
        fake_prove = make_fake_prove(
            [lean4_block("a"), lean4_block("b"), lean4_block("c")]
        )
        agent_prove(
            "prove it",
            model=None,
            tokenizer=None,
            capture_layer=0,
            max_turns=3,
            seed=5,
            prove_fn=fake_prove,
            lean_check_fn=lean_always_fails,
        )
        assert [call["seed"] for call in fake_prove.calls] == [5, 6, 7]

    def test_exhausts_max_turns_without_success(self):
        fake_prove = make_fake_prove([lean4_block("a"), lean4_block("b")])
        result = agent_prove(
            "prove it",
            model=None,
            tokenizer=None,
            capture_layer=0,
            max_turns=2,
            prove_fn=fake_prove,
            lean_check_fn=lean_always_fails,
        )
        assert result["turns"] == 2
        assert result["lean_status"] == LeanStatus.ERROR
        assert len(fake_prove.calls) == 2

    def test_no_lean4_block_reports_no_proof_found(self):
        fake_prove = make_fake_prove(["just prose, no code block"])
        result = agent_prove(
            "prove it",
            model=None,
            tokenizer=None,
            capture_layer=0,
            max_turns=1,
            prove_fn=fake_prove,
            lean_check_fn=lean_ok,
        )
        assert result["lean_status"] == "no proof found"

    def test_result_has_every_prove_run_result_key(self):
        fake_prove = make_fake_prove([lean4_block("attempt 1")])
        result = agent_prove(
            "prove it",
            model=None,
            tokenizer=None,
            capture_layer=0,
            max_turns=1,
            prove_fn=fake_prove,
            lean_check_fn=lean_ok,
        )
        assert set(ProveRunResult.__annotations__) <= set(result)


FAKE_CHALLENGE = {
    "id": "fake_bug",
    "lean_code": "def foo (n : Nat) : Nat := n\n"
    "theorem foo_id (n : Nat) : foo n = n := by sorry",
    "buggy": False,
}


class TestAgentProveThroughRunEval:
    """agent_prove, wired through the real run_eval + build_prompt, the same way
    rollback.py's CLI wires it -- confirms the two modules actually compose."""

    def test_run_eval_produces_one_row_per_challenge_mode_seed(self):
        # run_eval calls prove_fn once per (challenge, mode, seed)
        fake_prove = make_fake_prove([lean4_block("attempt 1")])
        prove_fn = partial(
            agent_prove,
            model=None,
            tokenizer=None,
            capture_layer=0,
            max_turns=1,
            prove_fn=fake_prove,
            lean_check_fn=lean_ok,
        )

        df = run_eval(
            [FAKE_CHALLENGE],
            build_prompt,
            prove_fn,
            modes=("plain",),
            seeds=(0,),
            canary="smell",
        )

        assert len(df) == 1
        assert (df["buggy"] == False).all()  # noqa: E712
        assert (df["outcome"] == LeanStatus.OK).all()
        assert (df["canary"] == "smell").all()
