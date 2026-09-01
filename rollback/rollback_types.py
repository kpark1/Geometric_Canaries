from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, TypedDict

import numpy as np


class StopReason(StrEnum):
    EOS = "eos"
    MAX_NEW_TOKENS = "max_new_tokens"


class LeanStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    SORRY = "sorry"
    TIMEOUT = "timeout"


type NoProofFound = Literal["no proof found"]
type ResponseLeanStatus = LeanStatus | NoProofFound
NO_PROOF_FOUND: NoProofFound = "no proof found"


class ProveRunResult(TypedDict):
    mode: str  # TODO
    canary: str
    prompt: str
    text: str
    ids: list[int]
    entropy: np.ndarray  # TODO
    logprob: np.ndarray  # TODO
    hidden: Any  # TODO
    segments: list
    events: list[dict[str, Any]]  # TODO
    abandoned: list[dict[str, Any]]  # TODO
    rollbacks: int
    n_forward: int
    n_prefills: int
    gen_tokens: int
    stop_reason: StopReason
    wall_s: float
    lean_status: ResponseLeanStatus


class AgentProveRunResult(ProveRunResult):
    """prove_fn-compatible result from agent.agent_prove: a ProveRunResult with
    cost/event fields summed across all turns, plus per-turn history."""

    turns: int
    turn_history: list[dict[str, Any]]


@dataclass
class ProcessResult:
    exit_code: int
    stdout: str


@dataclass
class ProcessTimeout:
    pass


type ProcessResultOrTimeout = ProcessResult | ProcessTimeout


class Challenge(TypedDict):
    id: str
    definition: str
    spec: str
    buggy: bool
