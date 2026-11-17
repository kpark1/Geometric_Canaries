from collections.abc import Callable, Iterable, Mapping
from typing import Any

import pandas as pd
from rollback_types import ProveRunResult


def run_eval(
    challenges: Iterable[Mapping[str, Any]],
    build_prompt_fn: Callable[..., str],
    build_challenge_lean_fn: Callable[..., str],
    prove_fn: Callable[..., ProveRunResult],
    variants: tuple[str, ...] = ("fixed", "buggy"),
    modes: tuple[str, ...] = ("plain", "rollback", "restart"),
    seeds: tuple[int, ...] = (0, 1),
    max_new_tokens: int = 384,
    thresh: float = 1.8,
    canary: str = "smell",
) -> pd.DataFrame:
    rows = []
    for challenge in challenges:
        paired = "fixed" in challenge and "buggy" in challenge
        challenge_variants: tuple[str | None, ...] = variants if paired else (None,)
        for variant in challenge_variants:
            if variant is not None:
                prompt = build_prompt_fn(challenge, variant)
                challenge_lean = build_challenge_lean_fn(challenge, variant)
            else:
                prompt = build_prompt_fn(challenge)
                challenge_lean = build_challenge_lean_fn(challenge)
            for mode in modes:
                for seed in seeds:
                    result = prove_fn(
                        prompt,
                        mode=mode,
                        canary=canary,
                        max_new_tokens=max_new_tokens,
                        thresh=thresh,
                        seed=seed,
                        challenge_lean=challenge_lean,
                        theorem_name=challenge.get("theorem_name", ""),
                    )
                    row = {
                        "id": challenge["id"],
                        "mode": mode,
                        "seed": seed,
                        "outcome": result["lean_status"],
                        "comparator_status": result["comparator_status"],
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
                    if variant is not None:
                        row.update(
                            variant=variant,
                            provable=challenge[variant]["provable"],
                        )
                    else:
                        row.update(
                            buggy=challenge["buggy"],
                            canary=result["canary"],
                        )
                    rows.append(row)
    return pd.DataFrame(rows)
