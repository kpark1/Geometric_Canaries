import re
from collections.abc import Callable, Iterable

import pandas as pd
from rollback_rust import BuildPromptFunction
from rollback_types import Challenge, ProveRunResult

CLAIM_PROOF = re.compile(r"```lean4?[\s\S]*?```|\bQED\b", re.IGNORECASE)
CLAIM_FALSE = re.compile(
    r"counterexample|statement is false|does not hold"
    r"|cannot be proved",
    re.IGNORECASE,
)


def proxy_outcome(text: str) -> str:
    """Stub outcome until Lean verification is wired in."""
    if CLAIM_FALSE.search(text):
        return "claims_false"
    if CLAIM_PROOF.search(text):
        return "claims_proof"
    return "no_conclusion"


def run_eval(
    challenges: Iterable[Challenge],
    build_prompt_fn: BuildPromptFunction,
    prove_fn: Callable[..., ProveRunResult],
    modes: tuple[str, ...] = ("plain", "rollback", "restart"),
    seeds: tuple[int, ...] = (0, 1),
    max_new_tokens: int = 384,
    thresh: float = 1.8,
) -> pd.DataFrame:
    rows = []
    for challenge in challenges:
        prompt = build_prompt_fn(challenge)
        for mode in modes:
            for seed in seeds:
                result = prove_fn(
                    prompt,
                    mode=mode,
                    max_new_tokens=max_new_tokens,
                    thresh=thresh,
                    seed=seed,
                )
                rows.append(
                    {
                        "id": challenge["id"],
                        "buggy": challenge["buggy"],
                        "mode": mode,
                        "seed": seed,
                        "outcome": result["lean_status"],
                        "stop_reason": result["stop_reason"],
                        "prompt": result.get("prompt", prompt),
                        "text": result["text"],
                        "events": result["events"],
                        "interventions": result["rollbacks"],
                        "gen_tokens": result["gen_tokens"],
                        "n_forward": result["n_forward"],
                        "prefills": result["n_prefills"],
                        "wall_s": result["wall_s"],
                    }
                )
    return pd.DataFrame(rows)
