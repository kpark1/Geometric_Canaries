from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Token:
    """Measurements associated with one generated token."""

    token_id: int
    logits: InitVar[torch.Tensor | None]
    hidden: np.ndarray

    entropy: float = field(init=False)
    logprob: float = field(init=False)

    def __post_init__(self, logits: torch.Tensor | None) -> None:
        """Calculate uncertainty measurements from the predictive logits."""
        # Force hint tokens use NaN vectors because no sampler selected them.
        if logits is None:
            self.entropy = np.nan
            self.logprob = np.nan
            return

        logps = F.log_softmax(logits, dim=-1)
        self.entropy = float(-(logps.exp() * logps).sum())
        self.logprob = float(logps[self.token_id])

    @classmethod
    def forced(cls, token_id: int, hidden: np.ndarray) -> "Token":
        """Create an injected token with undefined sampling measurements."""
        return cls(token_id=token_id, logits=None, hidden=hidden)


@dataclass(frozen=True)
class Segment:
    """A bounded view into the tokens and decoded text of a State."""

    state: "State" = field(repr=False)
    seg_start_tok: int
    seg_end_tok: int
    seg_start_char: int
    seg_end_char: int

    @property
    def tokens(self) -> list[Token]:
        """Return the tokens covered by this segment."""
        return self.state.tokens[self.seg_start_tok:self.seg_end_tok]

    @property
    def text(self) -> str:
        """Return the decoded text covered by this segment."""
        return self.state.text[self.seg_start_char:self.seg_end_char]

    @property
    def mean_entropy(self) -> float:
        """Return the mean token entropy within this segment."""
        return float(np.nanmean([token.entropy for token in self.tokens]))

    @property
    def min_logprob(self) -> float:
        """Return the minimum sampled-token log probability."""
        return float(np.nanmin([token.logprob for token in self.tokens]))

    @property
    def centroid(self) -> np.ndarray | None:
        """Return the segment's mean predictive hidden representation."""
        hidden = [token.hidden for token in self.tokens]
        return np.stack(hidden).mean(axis=0) if hidden else None

    # OLD: def snapshot(self, idx: int, lexical_smells) -> dict[str, Any]:
    def snapshot(self, lexical_smells) -> dict[str, Any]:  # NEW
        """Create an immutable-style record for scoring and history."""
        return {
            # REMOVED: "idx": idx,
            "tok_start": self.seg_start_tok,
            "tok_end": self.seg_end_tok,
            "char_start": self.seg_start_char,
            "char_end": self.seg_end_char,
            "text": self.text,
            "mean_entropy": self.mean_entropy,
            "min_logprob": self.min_logprob,
            "centroid": self.centroid,
            "smell": bool(lexical_smells.search(self.text)),
        }


@dataclass
class State:
    """Mutable state for the current surviving generation branch."""

    tokens: list[Token] = field(default_factory=list)
    text: str = ""

    seg_start_tok: int = 0
    seg_start_char: int = 0
    probe_char: int = 0

    @property
    def ids(self) -> list[int]:
        """Return the token IDs for the current surviving generation."""
        return [token.token_id for token in self.tokens]

    @property
    def step(self) -> int:
        """Return the number of tokens in the current generation."""
        return len(self.tokens)

    def add_token(self, token: Token) -> None:
        """Append one generated token to the surviving generation."""
        self.tokens.append(token)

    def decode(self, tokenizer) -> None:
        """Decode all surviving token IDs into readable text."""
        self.text = tokenizer.decode(self.ids, skip_special_tokens=True)

    def truncate(self, token_index: int, tokenizer) -> None:
        """Discard tokens from token_index onward and rebuild decoded text."""
        del self.tokens[token_index:]
        self.decode(tokenizer)


    def reset(self) -> None:
        """Clear state that should be discarded during a full restart."""
        self.tokens.clear()
        self.text = ""

        self.seg_start_tok = 0
        self.seg_start_char = 0
        self.probe_char = 0

@dataclass
class RunStats:
    """Accounting that survives rollback and full generation restarts."""

    n_forward: int = 0
    n_prefills: int = 1
    rollbacks: int = 0

    events: list[dict[str, Any]] = field(default_factory=list)
    abandoned: list[dict[str, Any]] = field(default_factory=list)

    def add_forward(self, positions: int) -> None:
        """Add processed token positions to the forward-pass accounting."""
        self.n_forward += positions
