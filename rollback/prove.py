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


HINT = ("\nWait -- let me re-check the last step carefully before continuing. "
        "I should verify each claim against the definitions.\n")


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


@torch.no_grad()
def prove(prompt, model, tokenizer, capture_layer, mode="rollback", max_new_tokens=512, max_rollbacks=3,
          thresh=1.8, temperature=1.0, top_p=0.95, seed=0, inject_hint=True,
          verbose=False):
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
    Model generates tokens
    → code detects a boundary
    → segment is “closed”
    → its statistics are calculated
    → it is scored
    → accepted or rolled back
    → next segment begins
    """
    t0 = time.time()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    input_ids = encode_prompt(prompt, tokenizer).to(model.device)
    prompt_len = input_ids.shape[1]

    # stores those key/value vectors as new tokens are generated (dynamic)
    cache = DynamicCache()
    out = model(input_ids=input_ids, past_key_values=cache, use_cache=True,
                output_hidden_states=True)
    logits = out.logits[0, -1].float()
    n_forward = prompt_len          # prefill counts once; rollback never repeats it

    """
    prefill: counts how many times the whole original prompt has been processed
    generation: one new token per model call
      in rollback mode, it remains 1, because cropping preserves the cached prompt
      in restart mode, the cache is thrown away and the prompt is processed again:
    """
    n_prefills = 1

    # live state (generation-relative)
    ids, text = [], ""
    entropy, logprob, hidden = [], [], []

    # mark where the current unfinished segment begins in ids (seg_start_tok) and text (seg_start_char)
    seg_start_tok, seg_start_char = 0, 0

    # where to resume searching for a future boundary
    probe_char = 0                  # search cursor: skips rejected (too-short) boundaries
    history, segments, events, abandoned = [], [], [], []
    rollbacks = 0

    # decides that the current reasoning step has ended, so it packages and scores it
    def close_segment(end_tok, end_char):
        sl = slice(seg_start_tok, end_tok)
        h = np.stack(hidden[sl]) if hidden[sl] else None
        seg = {
            "idx": len(segments),
            "tok_start": seg_start_tok, "tok_end": end_tok,
            "text": text[seg_start_char:end_char],
            "mean_entropy": float(np.nanmean(entropy[sl])),
            "min_logprob": float(np.nanmin(logprob[sl])),
            "centroid": h.mean(axis=0) if h is not None else None,
            "smell": bool(LEXICAL_SMELLS.search(text[seg_start_char:end_char])),
        }
        seg["score"] = score_segment(seg, history)
        return seg

    step = 0
    while step < max_new_tokens:
        tok = _sample(logits.cpu(), temperature, top_p, gen)
        h = out.hidden_states[capture_layer][0, -1].float().cpu().numpy()
        logps = F.log_softmax(logits, dim=-1)
        entropy.append(float(-(logps.exp() * logps).sum()))
        logprob.append(float(logps[tok]))
        hidden.append(h)
        ids.append(tok)
        text += tokenizer.decode([tok])
        step += 1

        if tok == tokenizer.eos_token_id:
            break

        # --- online boundary check --------------------------------------
        b = find_boundary(text, probe_char)
        if b is not None:
            # map char boundary -> token index
            # char->token map: walk per-token decodes (consistent with `text`)
            pos, end_tok = 0, len(ids)
            for i, tid in enumerate(ids):
                pos += len(tokenizer.decode([tid]))
                if pos >= b:
                    end_tok = i + 1
                    break
            if end_tok - seg_start_tok < MIN_SEGMENT_TOKENS:
                probe_char = b          # too short: merge onwards, keep searching
            else:
                seg = close_segment(end_tok, b)
                segments.append(seg)
                flagged = seg["score"] > thresh
                if verbose:
                    print(f"  seg {seg['idx']:2d} tok {seg['tok_start']:4d}-"
                          f"{seg['tok_end']:4d} score {seg['score']:+.2f}"
                          f"{'  << FLAG' if flagged else ''}")
                if flagged and mode != "plain" and rollbacks < max_rollbacks:
                    rollbacks += 1
                    abandoned.append({
                        "at_segment": seg["idx"], "text": text,
                        "ids": list(ids), "entropy": list(entropy),
                        "logprob": list(logprob),
                    })
                    if mode == "rollback":
                        ck = prompt_len + seg["tok_start"]
                        # before cropping: [prompt][good prefix][flagged segment]
                        # after cropping: [prompt][good prefix]
                        cache.crop(ck)

                        # delete the flagged segment from the other stored data
                        del ids[seg["tok_start"]:]
                        del entropy[seg["tok_start"]:]
                        del logprob[seg["tok_start"]:]
                        del hidden[seg["tok_start"]:]
                        text = text[:seg_start_char]
                        segments.pop()          # the bad segment is gone
                        events.append({"type": "rollback", "to_tok": seg["tok_start"],
                                       "score": seg["score"]})
                        cont_ids = []
                        if inject_hint:
                            cont_ids = tokenizer(HINT, add_special_tokens=False)["input_ids"]
                        if cont_ids:
                            ct = torch.tensor([cont_ids], device=model.device)
                            out = model(input_ids=ct, past_key_values=cache,
                                        use_cache=True, output_hidden_states=True)
                            n_forward += len(cont_ids)
                            for htok in cont_ids:
                                ids.append(htok)
                                text += tokenizer.decode([htok])
                                entropy.append(np.nan); logprob.append(np.nan)
                                hidden.append(out.hidden_states[capture_layer][0, -1]
                                              .float().cpu().numpy())
                            seg_start_tok, seg_start_char = len(ids), len(text)
                            probe_char = len(text)
                            logits = out.logits[0, -1].float()
                            step = len(ids)
                            continue
                        else:
                            # need fresh logits at the checkpoint: cheapest is to
                            # re-run the last checkpoint token -- crop one extra
                            cache.crop(ck - 1)
                            last = torch.tensor([[input_ids[0, -1] if not ids
                                                  else ids[-1]]], device=model.device)
                            out = model(input_ids=last, past_key_values=cache,
                                        use_cache=True, output_hidden_states=True)
                            n_forward += 1
                            seg_start_tok, seg_start_char = len(ids), len(text)
                            probe_char = len(text)
                            logits = out.logits[0, -1].float()
                            step = len(ids)
                            continue
                    else:  # restart baseline: full re-prefill
                        cache = DynamicCache()
                        out = model(input_ids=input_ids, past_key_values=cache,
                                    use_cache=True, output_hidden_states=True)
                        n_forward += prompt_len
                        n_prefills += 1
                        logits = out.logits[0, -1].float()
                        ids, text = [], ""
                        entropy, logprob, hidden = [], [], []
                        seg_start_tok, seg_start_char, probe_char = 0, 0, 0
                        history, segments = [], []
                        events.append({"type": "restart", "score": seg["score"]})
                        step = 0
                        continue
                history.append(seg)
                seg_start_tok, seg_start_char = seg["tok_end"], b
                probe_char = b

        # --- ordinary decode step ---------------------------------------
        nt = torch.tensor([[tok]], device=model.device)
        out = model(input_ids=nt, past_key_values=cache, use_cache=True,
                    output_hidden_states=True)
        logits = out.logits[0, -1].float()
        n_forward += 1

    # close the trailing segment for bookkeeping
    if len(ids) - seg_start_tok > 0:
        seg = close_segment(len(ids), len(text))
        segments.append(seg)

    return {
        "mode": mode, "text": text, "ids": ids,
        "entropy": np.array(entropy), "logprob": np.array(logprob),
        "hidden": np.stack(hidden) if hidden else np.zeros((0, 1)),
        "segments": segments, "events": events, "abandoned": abandoned,
        "rollbacks": rollbacks,
        "n_forward": n_forward, "n_prefills": n_prefills,
        "gen_tokens": len(ids) + sum(len(a["ids"]) for a in abandoned),
        "wall_s": time.time() - t0,
    }
