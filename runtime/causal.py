import torch
import torch.nn as nn

from compress import apply_compression
from runtime.block_factory import build_block_stack
from runtime.blocks import (
    apply_native_norm,
    execute_block_stack,
    prepare_attention_mask_for_heads as runtime_prepare_attention_mask_for_heads,
    stack_native_execution_info,
)
from runtime.generation import build_generation_config as runtime_build_generation_config
from runtime.generation import resolve_generation_sampling_mode as runtime_resolve_generation_sampling_mode
from runtime.ops import create_causal_mask as runtime_create_causal_mask
from runtime.ops import embedding as runtime_embedding
from runtime.ops import linear_module as runtime_linear_module
from runtime.ops import resolve_position_ids as runtime_resolve_position_ids
from runtime.positional import resolve_rope_embedding
from serve.engine import generate as engine_generate
from specs.config import ModelConfig
from tensor.norms import RMSNorm


class CausalLM(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        *,
        block_variant: str = "llama",
        drop_path_max: float = 0.0,
        init_recipe: str | None = None,
        tie_weights: bool = True,
        compress: dict | None = None,
        **overrides,
    ):
        super().__init__()
        self.cfg = cfg
        pad_idx = getattr(cfg, "pad_token_id", None)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=pad_idx)
        self.blocks = build_block_stack(
            cfg,
            variant=block_variant,
            drop_path_max=drop_path_max,
            init_recipe=init_recipe,
            **overrides,
        )
        use_rms = (str(block_variant).lower() == "llama") or (
            str(getattr(cfg, "norm", "")).lower() in ("rms", "rmsnorm")
        )
        if use_rms:
            self.norm = RMSNorm(cfg.d_model, eps=getattr(cfg, "rms_norm_eps", 1e-6))
        else:
            self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.embed.weight
        if compress is not None:
            lora_cfg = compress.get("lora") if isinstance(compress, dict) else None
            quant_cfg = compress.get("quant") if isinstance(compress, dict) else None
            self._compression_summary = apply_compression(self, lora=lora_cfg, quant=quant_cfg)
        else:
            self._compression_summary = None

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cache=None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        return_dict: bool = False,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if attention_mask is None:
            attention_mask = attn_mask
        x = (
            inputs_embeds
            if inputs_embeds is not None
            else runtime_embedding(self.embed.weight, input_ids, self.embed.padding_idx)
        )
        B, T, _ = x.shape
        token_attention_mask = None
        if attention_mask is not None and attention_mask.ndim == 2 and tuple(attention_mask.shape) == (B, T):
            token_attention_mask = attention_mask
        past_len = 0
        if position_ids is None:
            if cache_position is None and cache is not None and hasattr(cache, "layer"):
                try:
                    past_len = int(cache.layer(0).length())
                except Exception:
                    past_len = 0
            position_ids = runtime_resolve_position_ids(
                batch_size=B,
                seq_len=T,
                reference=x,
                attention_mask=token_attention_mask,
                cache_position=cache_position,
                past_length=past_len,
            )
        # Token-presence masks use the runtime causal-mask builder; explicit attention
        # masks (prefix/full additive/bool masks) are already authoritative.
        if attention_mask is not None and token_attention_mask is None:
            src_len = int(attention_mask.shape[-1])
            mask = runtime_prepare_attention_mask_for_heads(
                attention_mask,
                batch_size=B,
                num_heads=self.cfg.n_heads,
                tgt_len=T,
                src_len=src_len,
            )
        else:
            mask = runtime_create_causal_mask(
                reference=x,
                attention_mask=token_attention_mask,
                cache_position=cache_position,
                position_ids=position_ids,
            )
        head_dim = getattr(self.cfg, "head_dim", None) or int(self.cfg.d_model // self.cfg.n_heads)
        cos, sin = resolve_rope_embedding(
            reference=x,
            head_dim=head_dim,
            base_theta=float(getattr(self.cfg, "rope_theta", 1e6)),
            attention_scaling=float(getattr(self.cfg, "rope_attention_scaling", 1.0) or 1.0),
            scaling_type=getattr(self.cfg, "rope_scaling_type", None),
            scaling_factor=getattr(self.cfg, "rope_scaling_factor", None),
            original_max_position_embeddings=getattr(self.cfg, "rope_scaling_original_max_position_embeddings", None)
            or getattr(self.cfg, "max_position_embeddings", None),
            low_freq_factor=getattr(self.cfg, "rope_scaling_low_freq_factor", None),
            high_freq_factor=getattr(self.cfg, "rope_scaling_high_freq_factor", None),
            position_ids=position_ids,
        )
        x = execute_block_stack(
            self.blocks,
            x,
            mask,
            cache,
            position_embeddings=(cos, sin),
            position_ids=position_ids,
        )
        x = apply_native_norm(x, self.norm)
        logits = runtime_linear_module(x, self.lm_head)
        if return_dict:
            return {"logits": logits, "last_hidden_state": x}
        return logits

    def get_output_embeddings(self):
        return self.lm_head

    def native_execution_info(self) -> dict[str, object]:
        blocks = stack_native_execution_info(self.blocks)
        return {
            "model_type": type(self).__name__,
            "n_layers": len(blocks),
            "blocks": blocks,
            "all_blocks_native_inference_path": bool(all(block["fully_native_inference_path"] for block in blocks)),
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 128,
        do_sample: bool | None = None,
        temperature: float = 1.0,
        top_p: float | None = 1.0,
        top_k: int | None = None,
        eos_token_id: int | list[int] | None = None,
        no_repeat_ngram_size: int = 0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        attention_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        sliding_window: int | None = None,
        beam_size: int = 1,
        length_penalty: float = 1.0,
        cache_backend: str | None = None,
        return_dict: bool = True,
    ):
        resolved_attention_mask = attention_mask if attention_mask is not None else attn_mask
        resolved_do_sample = runtime_resolve_generation_sampling_mode(
            do_sample=do_sample,
            temperature=float(temperature),
            top_k=top_k,
            top_p=top_p,
        )
        eos_id = (
            int(eos_token_id)
            if isinstance(eos_token_id, int)
            else (int(eos_token_id[0]) if isinstance(eos_token_id, (list, tuple)) and eos_token_id else None)
        )
        cfg = runtime_build_generation_config(
            max_new_tokens=int(max_new_tokens),
            do_sample=resolved_do_sample,
            temperature=float(temperature),
            top_k=(int(top_k) if top_k is not None else None),
            top_p=(float(top_p) if top_p is not None else None),
            eos_id=eos_id,
            no_repeat_ngram=int(no_repeat_ngram_size),
            repetition_penalty=float(repetition_penalty),
            presence_penalty=float(presence_penalty),
            frequency_penalty=float(frequency_penalty),
            sliding_window=(int(sliding_window) if sliding_window is not None else None),
            beam_size=int(beam_size),
            length_penalty=float(length_penalty),
        )
        seq = engine_generate(
            self,
            input_ids.to(next(self.parameters()).device),
            attention_mask=(
                resolved_attention_mask.to(next(self.parameters()).device)
                if resolved_attention_mask is not None
                else None
            ),
            config=cfg,
            cache_backend=cache_backend,
        )
        if return_dict:
            return {"sequences": seq}
        return seq


TransformerLM = CausalLM

__all__ = ["CausalLM", "TransformerLM"]
