from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Optional

import torch
import torch.nn as nn

import runtime.causal as runtime_causal_module
import runtime.encoder as runtime_encoder_module
import runtime.seq2seq as runtime_seq2seq_module
from runtime.attention import _read_backend_from_env_or_file, scaled_dot_product_attention, select_attention_backend
from runtime.attention_modules import EagerAttention
from runtime.banded_attn_block import BandedAttentionBlock
from runtime.block_modules import CrossAttentionBlock, DecoderBlock, EncoderBlock, MoEBlock, ParallelTransformerBlock, TransformerBlock
from runtime.block_shared import CausalSelfAttentionBlockBase
from runtime.block_sparse_attn_block import BlockSparseAttentionBlock
from runtime.blocks import (
    apply_attention_biases,
    prepare_banded_attention_mask,
    prepare_block_sparse_attention_mask,
    prepare_cross_attention_mask,
    prepare_dilated_attention_mask,
    prepare_encoder_attention_mask,
    prepare_local_attention_mask,
    prepare_prefix_lm_attention_mask,
    prepare_segment_bidir_attention_mask,
    prepare_strided_attention_mask,
    prepare_window_pattern_attention_mask,
)
from runtime.causal import CausalLM
from runtime.dilated_local_attn_block import DilatedLocalAttentionBlock
from runtime.encoder import EncoderModel
from runtime.local_attn_block import LocalAttentionBlock
from runtime.ops import (
    activation as runtime_activation,
    attention as runtime_attention,
    apply_rotary as runtime_apply_rotary,
    embedding as runtime_embedding,
    gated_activation as runtime_gated_activation,
    head_output_projection as runtime_head_output_projection,
    linear as runtime_linear,
    merge_heads as runtime_merge_heads,
    pack_linear_weight as runtime_pack_linear_weight,
    pack_qkv_weights as runtime_pack_qkv_weights,
    prepare_attention_mask as runtime_prepare_attention_mask,
    qkv_heads_projection as runtime_qkv_heads_projection,
    qkv_packed_heads_projection as runtime_qkv_packed_heads_projection,
    resolve_linear_module_tensors as runtime_resolve_linear_module_tensors,
    resolve_rotary_embedding as runtime_resolve_rotary_embedding,
    split_heads as runtime_split_heads,
)
from runtime.prefix_lm_block import PrefixLMBlock
from runtime.segment_bidir_attn_block import SegmentBidirAttentionBlock
from runtime.seq2seq import EncoderDecoderLM
from runtime.strided_attn_block import StridedAttentionBlock
from runtime.window_pattern_attn_block import WindowPatternAttentionBlock
from tensor.mlp import MLP
from tensor.numerics import safe_softmax


ModelKind = Literal["causal", "encoder", "encoder_decoder"]
StackKind = Literal["causal", "encoder", "decoder"]
AttentionKind = Literal["self", "cross"]


@dataclass(frozen=True)
class ModelInputs:
    input_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    enc_input_ids: Optional[torch.Tensor] = None
    dec_input_ids: Optional[torch.Tensor] = None
    enc_padding_mask: Optional[torch.Tensor] = None
    dec_self_mask: Optional[torch.Tensor] = None

    @staticmethod
    def causal(input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> "ModelInputs":
        return ModelInputs(input_ids=input_ids, attention_mask=attention_mask)

    @staticmethod
    def encoder(input_ids: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> "ModelInputs":
        return ModelInputs(input_ids=input_ids, attention_mask=padding_mask)

    @staticmethod
    def encoder_decoder(
        enc_input_ids: torch.Tensor,
        dec_input_ids: torch.Tensor,
        enc_padding_mask: Optional[torch.Tensor] = None,
        dec_self_mask: Optional[torch.Tensor] = None,
    ) -> "ModelInputs":
        return ModelInputs(
            enc_input_ids=enc_input_ids,
            dec_input_ids=dec_input_ids,
            enc_padding_mask=enc_padding_mask,
            dec_self_mask=dec_self_mask,
        )


@dataclass(frozen=True)
class AttentionTarget:
    name: str
    module: nn.Module
    layer_index: int
    stack: StackKind
    kind: AttentionKind


@dataclass(frozen=True)
class BlockTarget:
    name: str
    module: nn.Module
    layer_index: int
    stack: StackKind
    kind: Literal["block", "self", "cross"]


@dataclass
class AttentionSnapshot:
    q: Optional[torch.Tensor] = None
    k: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None
    attn_mask: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    probs: Optional[torch.Tensor] = None
    head_out: Optional[torch.Tensor] = None
    output: Optional[torch.Tensor] = None


@dataclass
class MLPSnapshot:
    mlp_in: Optional[torch.Tensor] = None
    mlp_mid: Optional[torch.Tensor] = None
    mlp_out: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class MLPTarget:
    name: str
    module: MLP
    layer_index: int
    stack: StackKind
    kind: Literal["block", "self", "cross"]


def _clone_for_capture(x: Optional[torch.Tensor], *, detach: bool, move_to_cpu: bool) -> Optional[torch.Tensor]:
    if x is None:
        return None
    out = x
    if detach:
        out = out.detach()
    if move_to_cpu and out.device.type != "cpu":
        out = out.to("cpu")
    return out.clone()


def _stack_or_default(stack: Optional[str], kind: ModelKind) -> StackKind:
    if stack is not None:
        return stack  # type: ignore[return-value]
    if kind == "causal":
        return "causal"
    if kind == "encoder":
        return "encoder"
    return "decoder"


def _same_tensor_identity(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> bool:
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return False
    return (
        a.device == b.device
        and tuple(a.shape) == tuple(b.shape)
        and a.data_ptr() == b.data_ptr()
        and a.storage_offset() == b.storage_offset()
    )


class ModelAdapter:
    def __init__(self, model: nn.Module):
        self.model = model
        if isinstance(model, CausalLM):
            self.kind: ModelKind = "causal"
        elif isinstance(model, EncoderModel):
            self.kind = "encoder"
        elif isinstance(model, EncoderDecoderLM):
            self.kind = "encoder_decoder"
        elif hasattr(model, "enc_embed") and hasattr(model, "dec_embed") and hasattr(model, "decoder"):
            self.kind = "encoder_decoder"
        elif hasattr(model, "blocks") and hasattr(model, "embed"):
            self.kind = "causal"
        else:
            raise TypeError(f"Unsupported model type for interpret adapter: {type(model).__name__}")

    def forward(self, inputs: ModelInputs):
        if self.kind == "causal":
            if inputs.input_ids is None:
                raise ValueError("Causal models require input_ids")
            return self.model(inputs.input_ids, inputs.attention_mask)
        if self.kind == "encoder":
            if inputs.input_ids is None:
                raise ValueError("Encoder models require input_ids")
            return self.model(inputs.input_ids, inputs.attention_mask)
        if inputs.enc_input_ids is None or inputs.dec_input_ids is None:
            raise ValueError("Encoder-decoder models require enc_input_ids and dec_input_ids")
        return self.model(
            inputs.enc_input_ids,
            inputs.dec_input_ids,
            inputs.enc_padding_mask,
            inputs.dec_self_mask,
        )

    def output_module(self) -> Optional[nn.Module]:
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head
        return None

    def final_norm(self, *, stack: Optional[str] = None) -> Optional[nn.Module]:
        resolved = _stack_or_default(stack, self.kind)
        if self.kind == "causal":
            return getattr(self.model, "norm", None)
        if self.kind == "encoder":
            return getattr(self.model, "norm", None)
        if resolved == "encoder":
            return getattr(self.model, "enc_norm", None)
        return getattr(self.model, "dec_norm", None)

    def embedding_module(self, *, stack: Optional[str] = None) -> nn.Module:
        resolved = _stack_or_default(stack, self.kind)
        if self.kind in {"causal", "encoder"}:
            return self.model.embed
        if resolved == "encoder":
            return self.model.enc_embed
        return self.model.dec_embed

    def embedding_tokens(self, inputs: ModelInputs, *, stack: Optional[str] = None) -> torch.Tensor:
        tokens = self.sequence_tokens(inputs, stack=stack)
        if tokens is None:
            raise ValueError("A token sequence is required for embedding capture")
        return tokens

    def embedding_output(self, inputs: ModelInputs, *, stack: Optional[str] = None) -> torch.Tensor:
        embed = self.embedding_module(stack=stack)
        tokens = self.embedding_tokens(inputs, stack=stack)
        return runtime_embedding(embed.weight, tokens, getattr(embed, "padding_idx", None))

    def embedding_runtime_module(self):
        if self.kind == "causal":
            return runtime_causal_module
        if self.kind == "encoder":
            return runtime_encoder_module
        return runtime_seq2seq_module

    def named_modules(self) -> dict[str, nn.Module]:
        return dict(self.model.named_modules())

    def sequence_tokens(self, inputs: ModelInputs, *, stack: Optional[str] = None) -> Optional[torch.Tensor]:
        resolved = _stack_or_default(stack, self.kind)
        if self.kind in {"causal", "encoder"}:
            return inputs.input_ids
        if resolved == "encoder":
            return inputs.enc_input_ids
        return inputs.dec_input_ids

    def block_targets(self, *, stack: Optional[str] = None) -> list[BlockTarget]:
        resolved = _stack_or_default(stack, self.kind)
        if self.kind == "causal":
            return [
                BlockTarget(name=f"blocks.{i}", module=blk, layer_index=i, stack="causal", kind="block")
                for i, blk in enumerate(self.model.blocks)
            ]
        if self.kind == "encoder":
            return [
                BlockTarget(name=f"blocks.{i}", module=blk, layer_index=i, stack="encoder", kind="block")
                for i, blk in enumerate(self.model.blocks)
            ]
        if resolved == "encoder":
            return [
                BlockTarget(name=f"encoder.{i}", module=blk, layer_index=i, stack="encoder", kind="block")
                for i, blk in enumerate(self.model.encoder)
            ]
        out: list[BlockTarget] = []
        for i, blk in enumerate(self.model.decoder):
            out.append(BlockTarget(name=f"decoder.{i}.self_attn_block", module=blk.self_attn_block, layer_index=i, stack="decoder", kind="self"))
            out.append(BlockTarget(name=f"decoder.{i}.cross_block", module=blk.cross_block, layer_index=i, stack="decoder", kind="cross"))
        return out

    def attention_targets(self, *, stack: Optional[str] = None, kind: AttentionKind = "self") -> list[AttentionTarget]:
        resolved = _stack_or_default(stack, self.kind)
        if self.kind == "causal":
            return [
                AttentionTarget(name=f"blocks.{i}.attn", module=blk.attn, layer_index=i, stack="causal", kind="self")
                for i, blk in enumerate(self.model.blocks)
                if hasattr(blk, "attn")
            ]
        if self.kind == "encoder":
            return [
                AttentionTarget(name=f"blocks.{i}.attn", module=blk.attn, layer_index=i, stack="encoder", kind="self")
                for i, blk in enumerate(self.model.blocks)
                if hasattr(blk, "attn")
            ]
        if resolved == "encoder":
            return [
                AttentionTarget(name=f"encoder.{i}.attn", module=blk.attn, layer_index=i, stack="encoder", kind="self")
                for i, blk in enumerate(self.model.encoder)
                if hasattr(blk, "attn")
            ]
        if kind == "cross":
            return [
                AttentionTarget(name=f"decoder.{i}.cross_block.cross", module=blk.cross_block.cross, layer_index=i, stack="decoder", kind="cross")
                for i, blk in enumerate(self.model.decoder)
            ]
        out: list[AttentionTarget] = []
        for i, blk in enumerate(self.model.decoder):
            if hasattr(blk, "_ensure_self_attn"):
                blk._ensure_self_attn()
            module = getattr(blk.self_attn_block, "attn", None)
            if module is None:
                continue
            out.append(AttentionTarget(name=f"decoder.{i}.self_attn_block.attn", module=module, layer_index=i, stack="decoder", kind="self"))
        return out

    def attention_target(self, layer_index: int, *, stack: Optional[str] = None, kind: AttentionKind = "self") -> AttentionTarget:
        targets = self.attention_targets(stack=stack, kind=kind)
        for target in targets:
            if int(target.layer_index) == int(layer_index):
                return target
        raise KeyError(f"No attention target found for layer={layer_index}, stack={stack}, kind={kind}")

    def block_target(self, layer_index: int, *, stack: Optional[str] = None, kind: Optional[str] = None) -> BlockTarget:
        targets = self.block_targets(stack=stack)
        for target in targets:
            if int(target.layer_index) != int(layer_index):
                continue
            if kind is None or target.kind == kind:
                return target
        raise KeyError(f"No block target found for layer={layer_index}, stack={stack}, kind={kind}")

    def mlp_module(self, layer_index: int, *, stack: Optional[str] = None, kind: Optional[str] = None) -> MLP:
        block = self.block_target(layer_index, stack=stack, kind=kind)
        mlp = getattr(block.module, "mlp", None)
        if not isinstance(mlp, MLP):
            raise TypeError(f"Layer {layer_index} at stack={stack} has no MLP module")
        return mlp

    def mlp_targets(self, *, stack: Optional[str] = None, kind: Optional[str] = None) -> list[MLPTarget]:
        targets: list[MLPTarget] = []
        if self.kind != "encoder_decoder":
            for block in self.block_targets(stack=stack):
                mlp = getattr(block.module, "mlp", None)
                if isinstance(mlp, MLP):
                    targets.append(
                        MLPTarget(
                            name=f"{block.name}.mlp",
                            module=mlp,
                            layer_index=block.layer_index,
                            stack=block.stack,
                            kind=block.kind,
                        )
                    )
            return targets

        resolved = _stack_or_default(stack, self.kind)
        if resolved == "encoder":
            for block in self.block_targets(stack="encoder"):
                mlp = getattr(block.module, "mlp", None)
                if isinstance(mlp, MLP):
                    targets.append(
                        MLPTarget(
                            name=f"{block.name}.mlp",
                            module=mlp,
                            layer_index=block.layer_index,
                            stack=block.stack,
                            kind=block.kind,
                        )
                    )
            return targets

        kinds = [kind] if kind is not None else ["self", "cross"]
        for layer_index in range(len(self.model.decoder)):
            for block_kind in kinds:
                try:
                    mlp = self.mlp_module(layer_index, stack="decoder", kind=block_kind)
                except Exception:
                    continue
                targets.append(
                    MLPTarget(
                        name=f"decoder.{layer_index}.{block_kind}.mlp",
                        module=mlp,
                        layer_index=layer_index,
                        stack="decoder",
                        kind=block_kind,  # type: ignore[arg-type]
                    )
                )
        return targets


def get_model_adapter(model: nn.Module) -> ModelAdapter:
    return ModelAdapter(model)


def coerce_model_inputs(
    model: nn.Module,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    *,
    enc_input_ids: Optional[torch.Tensor] = None,
    dec_input_ids: Optional[torch.Tensor] = None,
    enc_padding_mask: Optional[torch.Tensor] = None,
    dec_self_mask: Optional[torch.Tensor] = None,
) -> ModelInputs:
    adapter = get_model_adapter(model)
    if adapter.kind == "causal":
        if input_ids is None:
            raise ValueError("input_ids is required for causal models")
        return ModelInputs.causal(input_ids, attention_mask)
    if adapter.kind == "encoder":
        if input_ids is None:
            raise ValueError("input_ids is required for encoder models")
        return ModelInputs.encoder(input_ids, attention_mask)
    if enc_input_ids is None or dec_input_ids is None:
        raise ValueError("enc_input_ids and dec_input_ids are required for encoder-decoder models")
    return ModelInputs.encoder_decoder(enc_input_ids, dec_input_ids, enc_padding_mask, dec_self_mask)


@contextmanager
def patched_embedding_output(
    adapter: ModelAdapter,
    *,
    inputs: Optional[ModelInputs] = None,
    stack: Optional[str] = None,
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    capture: Optional[Callable[[torch.Tensor], None]] = None,
    keep_grad: bool = False,
):
    resolved_stack = _stack_or_default(stack, adapter.kind)
    runtime_module = adapter.embedding_runtime_module()
    target_embed = adapter.embedding_module(stack=resolved_stack)
    target_weight = target_embed.weight
    target_tokens = adapter.embedding_tokens(inputs, stack=resolved_stack) if inputs is not None else None
    target_call_index = 1
    if adapter.kind == "encoder_decoder" and resolved_stack == "decoder":
        enc_embed = getattr(adapter.model, "enc_embed", None)
        if getattr(enc_embed, "weight", None) is target_weight:
            target_call_index = 2
    call_count = 0
    original_runtime_embedding = runtime_module.runtime_embedding

    def _patched_runtime_embedding(weight: torch.Tensor, input_ids: torch.Tensor, padding_idx: Optional[int]):
        nonlocal call_count
        out = original_runtime_embedding(weight, input_ids, padding_idx)
        same_tokens = _same_tensor_identity(input_ids, target_tokens)
        same_weight = _same_tensor_identity(weight, target_weight)
        matches_target = False
        if target_tokens is not None:
            matches_target = same_tokens
        elif same_weight:
            call_count += 1
            matches_target = call_count == target_call_index
        if not matches_target:
            return out
        if transform is not None:
            out = transform(out)
        if keep_grad and isinstance(out, torch.Tensor) and out.requires_grad:
            out.retain_grad()
        if capture is not None:
            capture(out)
        return out

    runtime_module.runtime_embedding = _patched_runtime_embedding
    try:
        yield
    finally:
        runtime_module.runtime_embedding = original_runtime_embedding


def resolve_model_score(
    model: nn.Module,
    outputs: torch.Tensor,
    *,
    position: int = -1,
    target_token_id: Optional[int] = None,
    target_feature_index: Optional[int] = None,
    score_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> tuple[torch.Tensor, Optional[int], Optional[int]]:
    """Resolve a scalar objective from model outputs.

    - For decoder models with an `lm_head`, defaults to a token logit objective.
    - For encoder-only models, defaults to a hidden-feature objective at `position`.
    - `score_fn` can override the objective for any model type.
    """
    if score_fn is not None:
        score = score_fn(outputs)
        if not isinstance(score, torch.Tensor):
            score = torch.as_tensor(score, device=outputs.device, dtype=outputs.dtype)
        if score.ndim != 0:
            raise ValueError("score_fn must return a scalar tensor")
        return score, target_token_id, target_feature_index

    if outputs.ndim != 3:
        raise ValueError(f"Expected model outputs with shape [B,T,D]; got {tuple(outputs.shape)}")

    adapter = get_model_adapter(model)
    if adapter.output_module() is not None:
        if target_token_id is None:
            target_token_id = int(outputs[0, position].argmax().item())
        return outputs[0, position, target_token_id], int(target_token_id), target_feature_index

    if target_feature_index is None:
        target_feature_index = int(outputs[0, position].argmax().item())
    return outputs[0, position, target_feature_index], target_token_id, int(target_feature_index)


def _expanded_kv_heads(attn: EagerAttention, kh_all: torch.Tensor, vh_all: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if int(attn.n_kv_heads) == int(attn.n_heads):
        return kh_all, vh_all
    repeat = int(attn.n_heads) // int(attn.n_kv_heads)
    return kh_all.repeat_interleave(repeat, dim=1), vh_all.repeat_interleave(repeat, dim=1)


def _prepare_attention_scores(
    attn: EagerAttention,
    qh: torch.Tensor,
    kh_all: torch.Tensor,
    add: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    kh_attn, _ = _expanded_kv_heads(attn, kh_all, kh_all)
    logits = torch.matmul(qh.float(), kh_attn.float().transpose(2, 3)) * float(attn.scaling)
    if add is not None:
        logits = logits + add.to(dtype=logits.dtype)
    elif bool(attn.is_causal) and qh.shape[2] > 1:
        tgt_len = int(qh.shape[2])
        src_len = int(kh_attn.shape[2])
        q_idx = torch.arange(tgt_len, device=qh.device).view(tgt_len, 1)
        k_idx = torch.arange(src_len, device=qh.device).view(1, src_len)
        future = k_idx > q_idx
        logits = logits.masked_fill(future.view(1, 1, tgt_len, src_len), torch.finfo(logits.dtype).min)
    probs = safe_softmax(logits.to(dtype=qh.dtype), dim=-1)
    return logits, probs


def eager_attention_forward(
    attn: EagerAttention,
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    mask: Optional[torch.Tensor],
    cache=None,
    *,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    position_ids: Optional[torch.Tensor] = None,
    patch_heads: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    capture: Optional[Callable[[AttentionSnapshot], None]] = None,
    keep_grad: bool = False,
    capture_logits: bool = False,
    capture_probs: bool = False,
) -> torch.Tensor:
    x = q
    batch_size, tgt_len, _ = x.shape
    device, dtype = x.device, x.dtype
    packed_backend = attn._packed_backend(x)

    if k is None and v is None:
        if packed_backend is not None:
            packed_weight, packed_bias, packed_sizes = attn._ensure_packed_qkv(packed_backend, x)
            q_size, k_size, v_size = packed_sizes
            qh, kh_new, vh_new = runtime_qkv_packed_heads_projection(
                x,
                packed_weight,
                packed_bias,
                q_size=q_size,
                k_size=k_size,
                v_size=v_size,
                q_heads=int(attn.n_heads),
                kv_heads=int(attn.n_kv_heads),
                backend=packed_backend,
            )
        elif attn._uses_module_runtime_linear():
            qh = runtime_split_heads(attn.w_q.runtime_linear(x), int(attn.n_heads))
            kh_new = runtime_split_heads(attn.w_k.runtime_linear(x), int(attn.n_kv_heads))
            vh_new = runtime_split_heads(attn.w_v.runtime_linear(x), int(attn.n_kv_heads))
        else:
            q_weight, q_bias = runtime_resolve_linear_module_tensors(attn.w_q, reference=x)
            k_weight, k_bias = runtime_resolve_linear_module_tensors(attn.w_k, reference=x)
            v_weight, v_bias = runtime_resolve_linear_module_tensors(attn.w_v, reference=x)
            qh, kh_new, vh_new = runtime_qkv_heads_projection(
                x,
                q_weight,
                q_bias,
                k_weight,
                k_bias,
                v_weight,
                v_bias,
                q_heads=int(attn.n_heads),
                kv_heads=int(attn.n_kv_heads),
            )
    else:
        qh = runtime_split_heads(attn.w_q.runtime_linear(x) if callable(getattr(attn.w_q, "runtime_linear", None)) else runtime_linear(x, *runtime_resolve_linear_module_tensors(attn.w_q, reference=x)), int(attn.n_heads))
        k_input = x if k is None else k
        v_input = x if v is None else v
        if callable(getattr(attn.w_k, "runtime_linear", None)):
            kh_new = runtime_split_heads(attn.w_k.runtime_linear(k_input), int(attn.n_kv_heads))
        else:
            kh_new = runtime_split_heads(runtime_linear(k_input, *runtime_resolve_linear_module_tensors(attn.w_k, reference=k_input)), int(attn.n_kv_heads))
        if callable(getattr(attn.w_v, "runtime_linear", None)):
            vh_new = runtime_split_heads(attn.w_v.runtime_linear(v_input), int(attn.n_kv_heads))
        else:
            vh_new = runtime_split_heads(runtime_linear(v_input, *runtime_resolve_linear_module_tensors(attn.w_v, reference=v_input)), int(attn.n_kv_heads))

    if bool(attn.use_rope):
        if position_embeddings is not None:
            cos, sin = position_embeddings
            qh, kh_new = runtime_apply_rotary(qh, kh_new, cos[:tgt_len], sin[:tgt_len])
        else:
            attn._ensure_rope_cache(tgt_len, x)
            qh, kh_new = runtime_apply_rotary(qh, kh_new, attn._rope_cos[:tgt_len], attn._rope_sin[:tgt_len])

    appended_to_cache = False
    if cache is not None:
        if tgt_len > 0 and hasattr(cache, "append_and_read"):
            kh_all, vh_all = cache.append_and_read(kh_new, vh_new, 0)
            appended_to_cache = True
        else:
            k_old, v_old = cache.read(0, cache.length())
            if k_old is not None and k_old.shape[2] > 0:
                kh_all = torch.cat([k_old, kh_new], dim=2)
                vh_all = torch.cat([v_old, vh_new], dim=2)
            else:
                kh_all, vh_all = kh_new, vh_new
    else:
        kh_all, vh_all = kh_new, vh_new

    src_len = int(kh_all.shape[2])
    add = runtime_prepare_attention_mask(
        mask,
        batch_size=int(batch_size),
        num_heads=int(attn.n_heads),
        tgt_len=int(tgt_len),
        src_len=int(src_len),
        position_ids=position_ids,
    )
    is_causal_flag = add is None
    forced_backend = attn.backend_override or _read_backend_from_env_or_file()
    use_runtime_native = forced_backend is None and float(attn.attn_dropout_p if attn.training else 0.0) == 0.0
    if use_runtime_native:
        head_out = runtime_attention(
            qh,
            kh_all,
            vh_all,
            attn_mask=add,
            is_causal=is_causal_flag,
            scale=float(attn.scaling),
        )
    else:
        backend = forced_backend or select_attention_backend(
            is_causal=is_causal_flag,
            dtype=dtype,
            seq=tgt_len,
            heads=int(attn.n_heads),
            device=device,
        )
        use_native_gqa = backend == "torch" and float(attn.attn_dropout_p if attn.training else 0.0) == 0.0
        if use_native_gqa:
            kh_backend, vh_backend = kh_all, vh_all
        else:
            kh_backend, vh_backend = _expanded_kv_heads(attn, kh_all, vh_all)
        mask_for_backend = add
        if mask_for_backend is not None and backend == "torch" and mask_for_backend.dtype != qh.dtype:
            mask_for_backend = mask_for_backend.to(dtype=qh.dtype)
        head_out = scaled_dot_product_attention(
            qh,
            kh_backend,
            vh_backend,
            attn_mask=mask_for_backend,
            dropout_p=(float(attn.attn_dropout_p) if attn.training else 0.0),
            backend=backend,
            is_causal=is_causal_flag,
            scale=float(attn.scaling),
        )

    if keep_grad:
        head_out.retain_grad()
    if patch_heads is not None:
        head_out = patch_heads(head_out)

    logits = probs = None
    if capture is not None or capture_logits or capture_probs:
        logits, probs = _prepare_attention_scores(attn, qh, kh_all, add)

    if cache is not None and tgt_len > 0 and not appended_to_cache:
        cache.append(kh_new, vh_new)

    if packed_backend is not None:
        packed_o_weight, packed_o_bias = attn._ensure_packed_output(packed_backend, head_out)
        output = runtime_head_output_projection(head_out, packed_o_weight, packed_o_bias, backend=packed_backend)
    elif callable(getattr(attn.w_o, "runtime_linear", None)):
        output = attn.w_o.runtime_linear(runtime_merge_heads(head_out))
    else:
        o_weight, o_bias = runtime_resolve_linear_module_tensors(attn.w_o, reference=head_out)
        output = runtime_head_output_projection(head_out, o_weight, o_bias)

    if capture is not None:
        capture(
            AttentionSnapshot(
                q=qh,
                k=kh_all,
                v=vh_all,
                attn_mask=add,
                logits=logits,
                probs=probs,
                head_out=head_out,
                output=output,
            )
        )
    return output


def _wrap_attention_with_capture(
    attn: EagerAttention,
    *,
    capture: Optional[Callable[[AttentionSnapshot], None]] = None,
    patch_heads: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    keep_grad: bool = False,
    capture_logits: bool = False,
    capture_probs: bool = False,
):
    orig_forward = attn.forward

    def forward(q, k, v, mask, cache=None, *, position_embeddings=None, position_ids=None):
        return eager_attention_forward(
            attn,
            q,
            k,
            v,
            mask,
            cache,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            patch_heads=patch_heads,
            capture=capture,
            keep_grad=keep_grad,
            capture_logits=capture_logits,
            capture_probs=capture_probs,
        )

    return orig_forward, forward


@contextmanager
def patched_attention(
    attn: EagerAttention,
    *,
    capture: Optional[Callable[[AttentionSnapshot], None]] = None,
    patch_heads: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    keep_grad: bool = False,
    capture_logits: bool = False,
    capture_probs: bool = False,
):
    orig_forward, new_forward = _wrap_attention_with_capture(
        attn,
        capture=capture,
        patch_heads=patch_heads,
        keep_grad=keep_grad,
        capture_logits=capture_logits,
        capture_probs=capture_probs,
    )
    try:
        attn.forward = new_forward  # type: ignore[assignment]
        yield
    finally:
        attn.forward = orig_forward  # type: ignore[assignment]


def mlp_forward(
    mlp: MLP,
    x: torch.Tensor,
    *,
    capture: Optional[Callable[[MLPSnapshot], None]] = None,
    patch_mid: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    keep_grad: bool = False,
) -> torch.Tensor:
    if mlp.gated:
        proj = runtime_linear(x, mlp.w_in.weight, mlp.w_in.bias)
        a, b = proj.chunk(2, dim=-1)
        name = mlp.activation_name.lower()
        leaky_relu_0p5_squared_aliases = {
            "leaky_relu_0p5_squared",
            "leaky-relu-0p5-squared",
            "leaky_relu_0.5_squared",
            "leaky-relu-0.5-squared",
        }
        if name in ("swiglu", "gated-silu"):
            mid = runtime_gated_activation(a, b, "silu")
        elif name == "geglu":
            mid = runtime_gated_activation(a, b, "gelu")
        elif name == "reglu":
            mid = runtime_gated_activation(a, b, "relu")
        elif name in leaky_relu_0p5_squared_aliases:
            mid = runtime_gated_activation(a, b, "leaky_relu_0p5_squared")
        else:
            mid = runtime_gated_activation(a, b, "silu")
    else:
        mid = runtime_activation(runtime_linear(x, mlp.w_in.weight, mlp.w_in.bias), mlp.activation_name)
    if keep_grad:
        mid.retain_grad()
    if patch_mid is not None:
        mid = patch_mid(mid)
    dropped = mlp.dropout(mid)
    out = runtime_linear(dropped, mlp.w_out.weight, mlp.w_out.bias)
    if capture is not None:
        capture(MLPSnapshot(mlp_in=x, mlp_mid=mid, mlp_out=out))
    return out


def _wrap_mlp_with_capture(
    mlp: MLP,
    *,
    capture: Optional[Callable[[MLPSnapshot], None]] = None,
    patch_mid: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    keep_grad: bool = False,
):
    orig_forward = mlp.forward

    def forward(x: torch.Tensor) -> torch.Tensor:
        return mlp_forward(mlp, x, capture=capture, patch_mid=patch_mid, keep_grad=keep_grad)

    return orig_forward, forward


@contextmanager
def patched_mlp(
    mlp: MLP,
    *,
    capture: Optional[Callable[[MLPSnapshot], None]] = None,
    patch_mid: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    keep_grad: bool = False,
):
    orig_forward, new_forward = _wrap_mlp_with_capture(
        mlp,
        capture=capture,
        patch_mid=patch_mid,
        keep_grad=keep_grad,
    )
    try:
        mlp.forward = new_forward  # type: ignore[assignment]
        yield
    finally:
        mlp.forward = orig_forward  # type: ignore[assignment]


__all__ = [
    "AttentionSnapshot",
    "AttentionTarget",
    "BlockTarget",
    "MLPSnapshot",
    "MLPTarget",
    "ModelAdapter",
    "ModelInputs",
    "coerce_model_inputs",
    "eager_attention_forward",
    "get_model_adapter",
    "patched_attention",
    "patched_mlp",
    "resolve_model_score",
]
