
from typing import Any, TypedDict

import numpy as np


class ProveRunResult(TypedDict):
    mode: str # TODO
    prompt: str
    text: str
    ids: list[int]
    entropy: np.ndarray # TODO
    logprob: np.ndarray # TODO
    hidden: Any # TODO
    segments: list
    events: list[dict[str, Any]] # TODO
    abandoned: list[dict[str, Any]] # TODO
    rollbacks: int
    n_forward: int
    n_prefills: int
    gen_tokens: int
    stop_reason: str # TODO
    wall_s: float