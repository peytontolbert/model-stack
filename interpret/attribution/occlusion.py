from __future__ import annotations

from typing import Iterable, List, Optional

import torch
import torch.nn as nn

from interpret.model_adapter import coerce_model_inputs, get_model_adapter, patched_embedding_output, resolve_model_score


def _infer_target_token_id(
    model: nn.Module,
    input_ids: Optional[torch.Tensor],
    position: int,
    attn_mask: Optional[torch.Tensor],
    *,
    enc_input_ids: Optional[torch.Tensor] = None,
    dec_input_ids: Optional[torch.Tensor] = None,
    enc_padding_mask: Optional[torch.Tensor] = None,
    dec_self_mask: Optional[torch.Tensor] = None,
    target_feature_index: Optional[int] = None,
    score_fn=None,
) -> int:
    adapter = get_model_adapter(model)
    inputs = coerce_model_inputs(
        model,
        input_ids,
        attn_mask,
        enc_input_ids=enc_input_ids,
        dec_input_ids=dec_input_ids,
        enc_padding_mask=enc_padding_mask,
        dec_self_mask=dec_self_mask,
    )
    with torch.no_grad():
        outputs = adapter.forward(inputs)
        _, target_token_id, target_feature_index = resolve_model_score(
            model,
            outputs,
            position=position,
            target_feature_index=target_feature_index,
            score_fn=score_fn,
        )
        return int(target_token_id if target_token_id is not None else target_feature_index)


def _occlude_positions_transform(positions: Iterable[int]):
    pos = sorted({int(p) for p in positions})

    def transform(out: torch.Tensor):
        if not isinstance(out, torch.Tensor) or out.ndim != 3:
            return out
        B, T, D = out.shape
        mask = torch.zeros(B, T, 1, device=out.device, dtype=out.dtype)
        for p in pos:
            if -T <= p < T:
                mask[:, p if p >= 0 else (T + p)] = 1
        return out * (1 - mask)

    return transform


@torch.inference_mode()
def token_occlusion_importance(
    model: nn.Module,
    input_ids: Optional[torch.Tensor],
    *,
    positions: Optional[Iterable[int]] = None,
    position: int = -1,
    target_token_id: Optional[int] = None,
    target_feature_index: Optional[int] = None,
    attn_mask: Optional[torch.Tensor] = None,
    enc_input_ids: Optional[torch.Tensor] = None,
    dec_input_ids: Optional[torch.Tensor] = None,
    enc_padding_mask: Optional[torch.Tensor] = None,
    dec_self_mask: Optional[torch.Tensor] = None,
    stack: Optional[str] = None,
    score_fn=None,
    mode: str = "logit",  # "logit" | "prob" | "nll"
) -> torch.Tensor:
    """Score importance of tokens by zeroing their embedding outputs.

    Returns a vector [T] with delta in the selected score when each position is occluded.
    - mode="logit": target logit drop (clean - occluded)
      "prob": target probability drop
      "nll": increase in -log p(target)
    If positions is None, scores all positions individually; otherwise only those.
    """
    adapter = get_model_adapter(model)
    inputs = coerce_model_inputs(
        model,
        input_ids,
        attn_mask,
        enc_input_ids=enc_input_ids,
        dec_input_ids=dec_input_ids,
        enc_padding_mask=enc_padding_mask,
        dec_self_mask=dec_self_mask,
    )
    resolved_stack = stack or ("causal" if adapter.kind == "causal" else "encoder" if adapter.kind == "encoder" else "decoder")
    was_training = model.training
    model.eval()

    # Determine target if not provided
    if target_token_id is None and target_feature_index is None and score_fn is None:
        inferred = _infer_target_token_id(
            model,
            input_ids,
            position,
            attn_mask,
            enc_input_ids=enc_input_ids,
            dec_input_ids=dec_input_ids,
            enc_padding_mask=enc_padding_mask,
            dec_self_mask=dec_self_mask,
            target_feature_index=target_feature_index,
            score_fn=score_fn,
        )
        if adapter.output_module() is not None:
            target_token_id = inferred
        else:
            target_feature_index = inferred

    # Baseline score
    outputs = adapter.forward(inputs)
    base_score, target_token_id, target_feature_index = resolve_model_score(
        model,
        outputs,
        position=position,
        target_token_id=target_token_id,
        target_feature_index=target_feature_index,
        score_fn=score_fn,
    )
    base_logit = base_score.float()
    if mode == "prob" or mode == "nll":
        if adapter.output_module() is None:
            raise ValueError("mode='prob' and mode='nll' require a model with token logits")
        probs = torch.softmax(outputs.float(), dim=-1)
        base_prob = probs[0, position, target_token_id]
        base_nll = -torch.log(base_prob.clamp_min(1e-45))

    # Decide which positions to evaluate
    seq_source = adapter.sequence_tokens(inputs, stack=resolved_stack)
    if seq_source is None:
        raise ValueError("A token sequence is required to score token occlusion importance")
    T = int(seq_source.shape[1])
    eval_positions = list(range(T)) if positions is None else [int(p) for p in positions]
    scores = torch.zeros(T, device=seq_source.device, dtype=torch.float32)

    for p in eval_positions:
        with patched_embedding_output(
            adapter,
            inputs=inputs,
            stack=resolved_stack,
            transform=_occlude_positions_transform([p]),
        ):
            l = adapter.forward(inputs)
        if mode == "logit":
            occl, _, _ = resolve_model_score(
                model,
                l,
                position=position,
                target_token_id=target_token_id,
                target_feature_index=target_feature_index,
                score_fn=score_fn,
            )
            occl = occl.float()
            scores[p] = (base_logit - occl).detach()
        elif mode == "prob":
            pr = torch.softmax(l.float(), dim=-1)[0, position, target_token_id]
            scores[p] = (base_prob - pr).detach()
        elif mode == "nll":
            pr = torch.softmax(l.float(), dim=-1)[0, position, target_token_id]
            nll = -torch.log(pr.clamp_min(1e-45))
            scores[p] = (nll - base_nll).detach()
        else:
            raise ValueError("mode must be 'logit' | 'prob' | 'nll'")

    if was_training:
        model.train()
    return scores.cpu()

