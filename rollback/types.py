
from typing import Any, TypedDict


class ProveRunResult(TypedDict):
    mode: Any
    prompt: Any
    text: Any
    ids: Any
    entropy: Any
    logprob: Any
    hidden: Any
    segments: Any
    events: Any
    abandoned: Any
    rollbacks: Any
    n_forward: Any
    n_prefills: Any
    gen_tokens: Any
    stop_reason: Any
    wall_s: Any