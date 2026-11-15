import logging
from typing import Any

import numpy as np
from state import RunStats, State, Token

logger = logging.getLogger("prove")


def log_initial_state(prompt_len: int, state: State) -> None:
    logger.debug(
        "initial state: prompt_len=%d step=%d ids=%r text=%r "
        "seg_start_tok=%d seg_start_char=%d probe_char=%d",
        prompt_len,
        state.token_count,
        state.ids,
        state.text,
        state.seg_start_tok,
        state.seg_start_char,
        state.probe_char,
    )


def log_step_start(state: State) -> None:
    logger.debug(
        "step %d start: ids_len=%d text_len=%d",
        state.token_count,
        len(state.ids),
        len(state.text),
    )


def log_sampled_token(
    state: State,
    token: Token,
    tokenizer: Any,
) -> None:
    logger.debug(
        "sampled token: step=%d token_id=%d raw=%r entropy=%r "
        "logprob=%r hidden_shape=%r text_tail=%r",
        state.token_count,
        token.token_id,
        tokenizer.convert_ids_to_tokens(token.token_id),
        token.entropy,
        token.logprob,
        token.hidden.shape,
        state.text[-120:],
    )


def log_eos(state: State) -> None:
    logger.debug("EOS reached at step %d", state.token_count)


def log_boundary_search(probe_char: int, boundary_char: int | None) -> None:
    logger.debug(
        "boundary search: probe_char=%d result=%r",
        probe_char,
        boundary_char,
    )


def log_boundary_rejected(token_count: int, minimum: int) -> None:
    logger.debug(
        "boundary rejected: token_count=%d minimum=%d",
        token_count,
        minimum,
    )


def log_segment(segment: dict[str, Any], flagged: bool) -> None:
    logger.debug(
        "segment closed: tok_range=(%d, %d) text=%r "
        "mean_entropy=%r min_logprob=%r smell=%r "
        "score=%r flagged=%r",
        segment["tok_start"],
        segment["tok_end"],
        segment["text"],
        segment["mean_entropy"],
        segment["min_logprob"],
        segment["smell"],
        segment["score"],
        flagged,
    )


def log_final_state(
    state: State,
    entropy: np.ndarray,
    logprob: np.ndarray,
    hidden: np.ndarray,
    segments: list[dict[str, Any]],
    stats: RunStats,
) -> None:
    logger.debug(
        "final state: ids_len=%d text_len=%d entropy_len=%d "
        "logprob_len=%d hidden_len=%d segments=%d events=%r rollbacks=%d",
        len(state.ids),
        len(state.text),
        len(entropy),
        len(logprob),
        len(hidden),
        len(segments),
        stats.events,
        stats.rollbacks,
    )
