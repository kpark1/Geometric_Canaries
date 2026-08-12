from __future__ import annotations

import re
from dataclasses import InitVar, dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class Token:
    """Measurements associated with one generated token.
    entropy: how uncertain the model was across all possible next tokens (including this token)
    logprob: the log probability to the token that was selected
    hidden: model's internal vector representation at the selected layer
    """


    token_id: int
    logits: InitVar[torch.Tensor | None] 
    hidden: np.ndarray

    entropy: float = field(init=False)
    logprob: float = field(init=False)

    def __post_init__(self, logits: torch.Tensor | None) -> None:
        """Automatically called for @dataclass after initialization:
            Calculate uncertainty measurements from the predictive logits."""
        # Force hint tokens use NaN vectors because no sampler selected them.
        if logits is None:
            self.entropy = np.nan
            self.logprob = np.nan
            return

        logps = F.log_softmax(logits, dim=-1)
        self.entropy = float(-(logps.exp() * logps).sum())
        self.logprob = float(logps[self.token_id])

    @classmethod
    def inject(cls, token_id: int, hidden: np.ndarray) -> "Token":
        """Create an injected token with undefined sampling measurements."""
        return cls(token_id=token_id, logits=None, hidden=hidden)


@dataclass
class State:
    """Mutable state for the current surviving generation branch.
    Token objects
    → token IDs such as [1847, 374, 264, ...]
    → tokenizer.decode(...)
    → readable text
    """ 

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
    def token_count(self) -> int:
        """Return the number of tokens in the current generation."""
        return len(self.tokens)

    def calc_segment(
        self,
        seg_end_tok: int,
        seg_end_char: int,
        lexical_smells: re.Pattern,
    ) -> dict[str, Any]:
        """Calculate a detached record for the current segment."""
        segment_tokens = self.tokens[self.seg_start_tok:seg_end_tok]
        segment_text = self.text[self.seg_start_char:seg_end_char]
        hidden = [token.hidden for token in segment_tokens]

        return {
            "tok_start": self.seg_start_tok,
            "tok_end": seg_end_tok,
            "char_start": self.seg_start_char,
            "char_end": seg_end_char,
            "text": segment_text,
            "mean_entropy": float(np.nanmean([
                token.entropy for token in segment_tokens
            ])),
            "min_logprob": float(np.nanmin([
                token.logprob for token in segment_tokens
            ])),
            "centroid": np.stack(hidden).mean(axis=0) if hidden else None,
            "smell": bool(lexical_smells.search(segment_text)),
        }

    def add_token(self, token: Token) -> None:
        """Append one generated token to the surviving generation."""
        self.tokens.append(token)

    def decode(self, tokenizer: Any) -> None:
        """Decode all surviving token IDs into readable text."""
        self.text = tokenizer.decode(self.ids, skip_special_tokens=True)

    def truncate(self, token_index: int, tokenizer: Any) -> None:
        """Remove unwanted tokens from token_index onward and rebuild self.text i.e. decoded text from remaining tokens."""
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
