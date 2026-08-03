import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import DynamicCache

from prompt import encode_prompt
from segment import (
    LEXICAL_SMELLS,
    MIN_SEGMENT_TOKENS,
    find_boundary,
    score_segment,
)

# OLD: from state import Token, State, RunStats
from state import RunStats, Segment, State, Token  # NEW

HINT = ("\nWait -- let me re-check the last step carefully before continuing. "
        "I should verify each claim against the definitions.\n")

logger = logging.getLogger(__name__)


def _sample(logits, temperature, top_p, gen):
    # selects the token with the highest logit if temperature is zero or negative
    if temperature <= 0:
        return int(logits.argmax())

    # convert logits into probabilities
    probs = F.softmax(logits / temperature, dim=-1)
    sp, si = probs.sort(descending=True)
    # keep the top-p candidate set
    keep = (sp.cumsum(-1) - sp) <= top_p
    keep[0] = True
    sp = sp * keep
    idx = torch.multinomial(sp / sp.sum(), 1, generator=gen)
    return int(si[idx])


def _token_index_at_char(ids, char_index, tokenizer):
    """Map a decoded character boundary to an exclusive token index."""
    for i in range(len(ids)):
        prefix = tokenizer.decode(ids[:i + 1], skip_special_tokens=True)
        if len(prefix) >= char_index:
            return i + 1
    return len(ids)


@torch.no_grad()
def prove(prompt, model, tokenizer, capture_layer, mode="rollback",
          max_new_tokens=512, max_rollbacks=3, thresh=1.8,
          temperature=1.0, top_p=0.95, seed=0, inject_hint=True):
    """One streaming generate-detect-rollback run.

    mode:
      'plain'    -- generate once, detect nothing
      'rollback' -- on a flagged segment, crop the KV cache back to the
                    checkpoint at the segment start and resample from there
      'restart'  -- on a flagged segment, throw everything away and resample
                    from the prompt (matched-intervention baseline: same
                    detector, no checkpointing)

    A *checkpoint* is just a sequence length: the KV cache up to the start of
    the current segment.  Rollback = cache.crop(ck) -- the prompt and the good
    prefix are never recomputed.

    Returns dict with text, per-token signals, segments, events, forward-pass
    and wall-time accounting, and abandoned branches (counterfactual data).

    procedures summary:
    generate one token
    ↓
    append it and decode the accumulated text
    ↓
    is there a new boundary?
    ├── no  → generate the next token
    └── yes → is the segment at least 8 tokens?
              ├── no  → keep accumulating
              └── yes → close and score the segment
                        ↓
                        score above threshold?
                        ├── no  → accept and begin next segment
                        └── yes → roll back and regenerate
    """
    t0 = time.time()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    input_ids = encode_prompt(prompt, tokenizer).to(model.device)
    prompt_len = input_ids.shape[1]

    cache = DynamicCache()
    out = model(
        input_ids=input_ids,
        past_key_values=cache,
        use_cache=True,
        output_hidden_states=True,
    )
    logits = out.logits[0, -1].float()

    state = State()
    stats = RunStats(n_forward=prompt_len, n_prefills=1)
    history = []
    segments = []

    logger.debug(
        "initial state: prompt_len=%d step=%d ids=%r text=%r "
        "seg_start_tok=%d seg_start_char=%d probe_char=%d",
        prompt_len,
        state.step,
        state.ids,
        state.text,
        state.seg_start_tok,
        state.seg_start_char,
        state.probe_char,
    )

    def close_segment(seg_end_tok, seg_end_char):
        segment = Segment(
            state=state,
            seg_start_tok=state.seg_start_tok,
            seg_end_tok=seg_end_tok,
            seg_start_char=state.seg_start_char,
            seg_end_char=seg_end_char,
        )
        snapshot = segment.snapshot(lexical_smells=LEXICAL_SMELLS)
        snapshot["score"] = score_segment(snapshot, history)
        return snapshot

    while state.step < max_new_tokens:
        logger.debug(
            "step %d start: ids_len=%d text_len=%d",
            state.step,
            len(state.ids),
            len(state.text),
        )

        tok = _sample(logits.cpu(), temperature, top_p, gen)
        predictive_hidden = (
            out.hidden_states[capture_layer][0, -1]
            .float()
            .cpu()
            .numpy()
        )
        token = Token(
            token_id=tok,
            logits=logits,
            hidden=predictive_hidden,
        )
        state.add_token(token)
        state.decode(tokenizer)

        logger.debug(
            "sampled token: step=%d token_id=%d raw=%r entropy=%r "
            "logprob=%r hidden_shape=%r text_tail=%r",
            state.step,
            tok,
            tokenizer.convert_ids_to_tokens(tok),
            token.entropy,
            token.logprob,
            token.hidden.shape,
            state.text[-120:],
        )

        if tok == tokenizer.eos_token_id:
            logger.debug("EOS reached at step %d", state.step)
            break

        boundary_char = find_boundary(state.text, state.probe_char)
        logger.debug(
            "boundary search: probe_char=%d result=%r",
            state.probe_char,
            boundary_char,
        )

        if boundary_char is not None:
            seg_end_tok = _token_index_at_char(
                state.ids,
                boundary_char,
                tokenizer,
            )
            segment_token_count = seg_end_tok - state.seg_start_tok

            if segment_token_count < MIN_SEGMENT_TOKENS:
                state.probe_char = boundary_char
                logger.debug(
                    "boundary rejected: token_count=%d minimum=%d",
                    segment_token_count,
                    MIN_SEGMENT_TOKENS,
                )
            else:
                segment = close_segment(seg_end_tok, boundary_char)
                flagged = segment["score"] > thresh
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

                # Should the program reject this segment and perform the
                # configured intervention?
                should_intervene = (
                    flagged
                    and mode != "plain"
                    and stats.rollbacks < max_rollbacks
                )
                if should_intervene:
                    stats.rollbacks += 1
                    stats.abandoned.append({
                        "segment_start_tok": segment["tok_start"],
                        "segment_end_tok": segment["tok_end"],
                        "text": state.text,
                        "ids": list(state.ids),
                        "entropy": [t.entropy for t in state.tokens],
                        "logprob": [t.logprob for t in state.tokens],
                    })

                    if mode == "rollback":
                        checkpoint = prompt_len + segment["tok_start"]
                        cache.crop(checkpoint)
                        state.truncate(segment["tok_start"], tokenizer)
                        stats.events.append({
                            "type": "rollback",
                            "to_tok": segment["tok_start"],
                            "score": segment["score"],
                        })

                        hint_ids = []
                        if inject_hint:
                            hint_ids = tokenizer(
                                HINT,
                                add_special_tokens=False,
                            )["input_ids"]

                        if hint_ids:
                            hint_tensor = torch.tensor(
                                [hint_ids],
                                device=model.device,
                            )
                            out = model(
                                input_ids=hint_tensor,
                                past_key_values=cache,
                                use_cache=True,
                                output_hidden_states=True,
                            )
                            stats.add_forward(len(hint_ids))

                            hidden_size = out.hidden_states[capture_layer].shape[-1]
                            for hint_token_id in hint_ids:
                                hint_hidden = np.full(
                                    hidden_size,
                                    np.nan,
                                    dtype=np.float32,
                                )
                                state.add_token(Token.forced(
                                    token_id=hint_token_id,
                                    hidden=hint_hidden,
                                ))

                            state.decode(tokenizer)
                            state.seg_start_tok = state.step
                            state.seg_start_char = len(state.text)
                            state.probe_char = len(state.text)
                            logits = out.logits[0, -1].float()
                            continue

                        cache.crop(checkpoint - 1)
                        if state.ids:
                            last_token = torch.tensor(
                                [[state.ids[-1]]],
                                device=model.device,
                            )
                        else:
                            last_token = input_ids[:, -1:]

                        out = model(
                            input_ids=last_token,
                            past_key_values=cache,
                            use_cache=True,
                            output_hidden_states=True,
                        )
                        stats.add_forward(1)
                        state.seg_start_tok = state.step
                        state.seg_start_char = len(state.text)
                        state.probe_char = len(state.text)
                        logits = out.logits[0, -1].float()
                        continue

                    if mode == "restart":
                        cache = DynamicCache()
                        out = model(
                            input_ids=input_ids,
                            past_key_values=cache,
                            use_cache=True,
                            output_hidden_states=True,
                        )
                        stats.add_forward(prompt_len)
                        stats.n_prefills += 1
                        stats.events.append({
                            "type": "restart",
                            "score": segment["score"],
                        })
                        logits = out.logits[0, -1].float()
                        state.reset()
                        history.clear()
                        segments.clear()
                        continue

                history.append(segment)
                segments.append(segment)
                state.seg_start_tok = segment["tok_end"]
                state.seg_start_char = segment["char_end"]
                state.probe_char = segment["char_end"]

        next_token = torch.tensor([[tok]], device=model.device)
        out = model(
            input_ids=next_token,
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
        )
        logits = out.logits[0, -1].float()
        stats.add_forward(1)

    if state.step - state.seg_start_tok > 0:
        segments.append(close_segment(state.step, len(state.text)))

    entropy = np.array([token.entropy for token in state.tokens])
    logprob = np.array([token.logprob for token in state.tokens])
    hidden = (
        np.stack([token.hidden for token in state.tokens])
        if state.tokens
        else np.zeros((0, 1))
    )

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

    return {
        "mode": mode,
        "text": state.text,
        "ids": state.ids,
        "entropy": entropy,
        "logprob": logprob,
        "hidden": hidden,
        "segments": segments,
        "events": stats.events,
        "abandoned": stats.abandoned,
        "rollbacks": stats.rollbacks,
        "n_forward": stats.n_forward,
        "n_prefills": stats.n_prefills,
        "gen_tokens": (
            state.step
            + sum(len(branch["ids"]) for branch in stats.abandoned)
        ),
        "wall_s": time.time() - t0,
    }
