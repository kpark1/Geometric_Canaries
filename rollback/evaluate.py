from collections.abc import Callable, Iterable

import pandas as pd
from rollback_types import Challenge, ProveRunResult
from verina import BuildPromptFunction


def run_eval(
    challenges: Iterable[Challenge],
    build_prompt_fn: BuildPromptFunction,
    prove_fn: Callable[..., ProveRunResult],
    modes: tuple[str, ...] = ("plain", "rollback", "restart"),
    seeds: tuple[int, ...] = (0, 1),
    max_new_tokens: int = 384,
    thresh: float = 1.8,
    canary: str = "smell",
) -> pd.DataFrame:
    rows = []
    for challenge in challenges:
        prompt = build_prompt_fn(challenge)
        for mode in modes:
            for seed in seeds:
                result = prove_fn(
                    prompt,
                    mode=mode,
                    canary=canary,
                    max_new_tokens=max_new_tokens,
                    thresh=thresh,
                    seed=seed,
                )
                rows.append(
                    {
                        "id": challenge["id"],
                        "buggy": challenge["buggy"],
                        "mode": mode,
                        "canary": result["canary"],
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
                        "raw_result": result,
                    }
                )
    return pd.DataFrame(rows)
