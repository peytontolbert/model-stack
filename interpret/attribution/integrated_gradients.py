from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from interpret.model_adapter import coerce_model_inputs, get_model_adapter, patched_embedding_output, resolve_model_score

@torch.no_grad()
def predict_argmax(
    model: nn.Module,
    input_ids: Optional[torch.Tensor],
    position: int = -1,
    attn_mask: Optional[torch.Tensor] = None,
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
    outputs = adapter.forward(inputs)
    _, target_token_id, target_feature_index = resolve_model_score(
        model,
        outputs,
        position=position,
        target_feature_index=target_feature_index,
        score_fn=score_fn,
    )
    return int(target_token_id if target_token_id is not None else target_feature_index)


def integrated_gradients_tokens(
    model: nn.Module,
    input_ids: Optional[torch.Tensor],
    *,
    target_token_id: Optional[int] = None,
    target_feature_index: Optional[int] = None,
    steps: int = 20,
    position: int = -1,
    attn_mask: Optional[torch.Tensor] = None,
    enc_input_ids: Optional[torch.Tensor] = None,
    dec_input_ids: Optional[torch.Tensor] = None,
    enc_padding_mask: Optional[torch.Tensor] = None,
    dec_self_mask: Optional[torch.Tensor] = None,
    stack: Optional[str] = None,
    score_fn=None,
) -> torch.Tensor:
    """Integrated gradients attribution over input token embeddings.

    Returns attribution scores per token: shape [T], using zero baseline and linear path scaling of embedding output.
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

    if target_token_id is None and target_feature_index is None and score_fn is None:
        inferred = predict_argmax(
            model,
            input_ids,
            position=position,
            attn_mask=attn_mask,
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

    # We will scale the embedding output by alpha in [0,1]

    def run_with_alpha(alpha: float) -> torch.Tensor:
        grads: dict[str, torch.Tensor] = {}

        def _transform(out: torch.Tensor) -> torch.Tensor:
            return out * alpha

        def _capture(out_alpha: torch.Tensor) -> None:
            def save_grad(grad: torch.Tensor) -> None:
                grads["emb_grad"] = grad

            out_alpha.register_hook(save_grad)

        with patched_embedding_output(
            adapter,
            inputs=inputs,
            stack=resolved_stack,
            transform=_transform,
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
        return grads["emb_grad"]

    total_grad = None
    for s in range(1, steps + 1):
        alpha = float(s) / float(steps)
        grad = run_with_alpha(alpha)
        grad = grad.detach()  # [B, T, D]
        total_grad = grad if total_grad is None else total_grad + grad

    # Average gradient along path and multiply by (emb - baseline) = emb
    with torch.no_grad():
        emb = adapter.embedding_output(inputs, stack=resolved_stack).detach()

    avg_grad = total_grad / float(steps)
    attrib = (emb * avg_grad).sum(dim=-1)  # [B, T]
    scores = attrib[0].detach().cpu()

    if was_training:
        model.train()
    return scores

