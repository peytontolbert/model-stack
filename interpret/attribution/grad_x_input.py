from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from interpret.model_adapter import coerce_model_inputs, get_model_adapter, patched_embedding_output, resolve_model_score


def grad_x_input_tokens(
    model: nn.Module,
    input_ids: Optional[torch.Tensor],
    *,
    target_token_id: Optional[int] = None,
    target_feature_index: Optional[int] = None,
    position: int = -1,
    attn_mask: Optional[torch.Tensor] = None,
    enc_input_ids: Optional[torch.Tensor] = None,
    dec_input_ids: Optional[torch.Tensor] = None,
    enc_padding_mask: Optional[torch.Tensor] = None,
    dec_self_mask: Optional[torch.Tensor] = None,
    stack: Optional[str] = None,
    score_fn=None,
) -> torch.Tensor:
    """Gradient x input attribution on token embeddings; returns scores [T].

    Uses embedding output as "input". If target_token_id is None, uses argmax at position.
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

    outputs = adapter.forward(inputs)
    _, target_token_id, target_feature_index = resolve_model_score(
        model,
        outputs,
        position=position,
        target_token_id=target_token_id,
        target_feature_index=target_feature_index,
        score_fn=score_fn,
    )

    feats: dict[str, torch.Tensor] = {}

    def _capture(output: torch.Tensor) -> None:
        feats["emb"] = output

    with patched_embedding_output(
        adapter,
        inputs=inputs,
        stack=resolved_stack,
        capture=_capture,
        keep_grad=True,
    ):
        outputs = adapter.forward(inputs)
        score, _, _ = resolve_model_score(
            model,
            outputs,
            position=position,
            target_token_id=target_token_id,
            target_feature_index=target_feature_index,
            score_fn=score_fn,
        )
        model.zero_grad(set_to_none=True)
        score.backward()
        grad = feats["emb"].grad.detach()  # [B,T,D]
        emb = feats["emb"].detach()

    scores = (emb * grad).sum(dim=-1)[0].cpu()
    if was_training:
        model.train()
    return scores

