
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypedDict

import numpy as np

class StopReason(StrEnum):
    EOS = "eos"
    MAX_NEW_TOKENS = "max_new_tokens"

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
    stop_reason: StopReason
    wall_s: float


@dataclass
class ProcessResult:
    exit_code: int
    stdout: str

@dataclass
class ProcessTimeout: pass

type ProcessResultOrTimeout = ProcessResult | ProcessTimeout
