"""Multi-turn agentic proof loop: generate, check via Lean, feed errors back, retry.

Unlike prove(), which is a single generate-detect-rollback pass, this wraps
prove() in an outer retry loop that gives the model a tool it can act on: real
Lean elaboration feedback on its own previous attempt. Each turn still runs
prove()'s full streaming generation with the same rollback/restart/canary
machinery intact -- only the outer retry-on-failure loop is new.
"""

import time
from collections.abc import Callable
from typing import Any

from detect import Canary
from lean import (
    DEFAULT_TIMEOUT_S,
    extract_lean_deepseek_prover_style,
    lean_check,
)
from rollback_types import (
    NO_PROOF_FOUND,
    AgentProveRunResult,
    LeanStatus,
    ProveRunResult,
    ResponseLeanStatus,
)

NO_PROOF_FEEDBACK = (
    "Your previous response did not contain a fenced ```lean4 code block. "
    "Please produce one."
)


def _retry_message(lean_code: str | None, feedback: str) -> str:
    if lean_code is None:
        return NO_PROOF_FEEDBACK
    return (
        f"Your previous attempt did not elaborate in Lean 4:\n\n{feedback}\n\n"
        "Revise your proof and produce a corrected ```lean4 block."
    )

MULTI_TURN_INSTRUCTIONS = (
    "\n\n"
    "Include the full proof file as the last fenced ```lean4 code block in your response. "
    "I will run that proof through the lean checker and report any feedback to you if it doesn't elaborate, "
    "in which case, you will have another attempt. "
    "Subsequent attempts should also include a ```lean4 code block. "
)


def agent_prove(
    prompt: str,
    model: Any,
    tokenizer: Any,
    capture_layer: int,
    max_turns: int = 3,
    mode: str = "rollback",
    canary: str | Canary = "smell",
    max_new_tokens: int = 2048,
    max_rollbacks: int = 3,
    thresh: float = 1.8,
    temperature: float = 1.0,
    top_p: float = 0.95,
    seed: int = 0,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    prove_fn: Callable[..., ProveRunResult] | None = None,
    lean_check_fn: Callable[[str, float], tuple[LeanStatus, str]] = lean_check,
) -> AgentProveRunResult:
    """Give the model up to max_turns attempts, checking each via real Lean.

    Each turn calls prove_fn() once, so it gets the same mode/canary machinery
    as a single-attempt prove() call. If the extracted proof doesn't elaborate
    cleanly, Lean's diagnostic output is appended as a new user turn and the
    model tries again, up to max_turns. Stops early on the first turn that
    elaborates with no `sorry`.

    prove_fn/lean_check_fn are injectable (defaulting to the real prove/
    lean_check) so this loop is unit-testable without a real model or a Lean
    toolchain -- same spirit as rollback.py's partial(run_proof, model=...).
    prove_fn defaults to the real prove() via a lazy import, so importing this
    module (with a fake prove_fn supplied) doesn't require torch either.
    """
    if prove_fn is None:
        from prove import prove as prove_fn

    prompt = prompt + MULTI_TURN_INSTRUCTIONS # this is somewhat hacky, but it's a fast way to get us to something approximating the results we want

    t0 = time.time()
    messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
    turn_history: list[dict[str, Any]] = []

    total_rollbacks = 0
    total_n_forward = 0
    total_n_prefills = 0
    total_gen_tokens = 0
    all_events: list[dict[str, Any]] = []
    all_abandoned: list[dict[str, Any]] = []
    result: ProveRunResult | None = None
    final_status: ResponseLeanStatus = NO_PROOF_FOUND

    for turn in range(1, max_turns + 1):
        result = prove_fn(
            prompt,
            model=model,
            tokenizer=tokenizer,
            capture_layer=capture_layer,
            mode=mode,
            canary=canary,
            max_new_tokens=max_new_tokens,
            max_rollbacks=max_rollbacks,
            thresh=thresh,
            temperature=temperature,
            top_p=top_p,
            seed=seed + turn - 1,  # vary sampling across turns
            timeout_s=timeout_s,
            messages=list(messages),
        )
        messages.append({"role": "assistant", "content": result["text"]})

        total_rollbacks += result["rollbacks"]
        total_n_forward += result["n_forward"]
        total_n_prefills += result["n_prefills"]
        total_gen_tokens += result["gen_tokens"]
        all_events.extend(result["events"])
        all_abandoned.extend(result["abandoned"])

        lean_code = extract_lean_deepseek_prover_style(result["text"])
        if lean_code is None:
            status: ResponseLeanStatus = NO_PROOF_FOUND
            feedback = "(no ```lean4 block found in the response)"
        else:
            status, feedback = lean_check_fn(lean_code, timeout_s=timeout_s)

        turn_history.append(
            {
                "turn": turn,
                "lean_status": status,
                "feedback": feedback,
                "gen_tokens": result["gen_tokens"],
                "n_forward": result["n_forward"],
                "wall_s": result["wall_s"],
                "text": result["text"],
            }
        )
        final_status = status

        if status == LeanStatus.OK:
            break
        if turn < max_turns:
            messages.append(
                {"role": "user", "content": _retry_message(lean_code, feedback)}
            )

    assert result is not None  # max_turns >= 1 guarantees at least one iteration

    return {
        **result,
        "prompt": prompt,
        "rollbacks": total_rollbacks,
        "n_forward": total_n_forward,
        "n_prefills": total_n_prefills,
        "gen_tokens": total_gen_tokens,
        "events": all_events,
        "abandoned": all_abandoned,
        "wall_s": time.time() - t0,
        "lean_status": final_status,
        "turns": len(turn_history),
        "turn_history": turn_history,
    }
