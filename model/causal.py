import torch
import torch.nn as nn

from runtime.ops import create_causal_mask as runtime_create_causal_mask
from runtime.ops import embedding as runtime_embedding
from runtime.ops import linear as runtime_linear
from runtime.ops import resolve_position_ids as runtime_resolve_position_ids
from runtime.ops import resolve_rotary_embedding as runtime_resolve_rotary_embedding
from specs.config import ModelConfig
from compress import apply_compression
from blocks.factory import build_block_stack
from blocks.native_fusion import apply_native_norm
from tensor.norms import RMSNorm
from serve.engine import generate as engine_generate
from serve.engine import GenerationConfig


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
        # Final norm: use RMSNorm for LLaMA variants or when cfg.norm indicates rmsnorm
        use_rms = (str(block_variant).lower() == "llama") or (str(getattr(cfg, "norm", "")).lower() in ("rms", "rmsnorm"))
        if use_rms:
            self.norm = RMSNorm(cfg.d_model, eps=getattr(cfg, "rms_norm_eps", 1e-6))
        else:
            self.norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.embed.weight
        # Optional compression hooks
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
        x = (
            inputs_embeds
            if inputs_embeds is not None
            else runtime_embedding(self.embed.weight, input_ids, self.embed.padding_idx)
        )
        B, T, _ = x.shape
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
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_length=past_len,
            )
        # HF-like additive mask (B,1,T,S)
        mask = runtime_create_causal_mask(
            reference=x,
            attention_mask=attention_mask,
            cache_position=cache_position,
            position_ids=position_ids,
        )
        # Rotary embeddings
        head_dim = getattr(self.cfg, "head_dim", None) or int(self.cfg.d_model // self.cfg.n_heads)
        # Respect HF rope_scaling (e.g., LLaMA 3 uses linear scaling): stretch base_theta when type=="linear"
        rope_theta = float(getattr(self.cfg, "rope_theta", 1e6))
        try:
            st = (getattr(self.cfg, "rope_scaling_type", None) or "").lower()
            fac = getattr(self.cfg, "rope_scaling_factor", None)
            if st == "linear" and fac is not None:
                rope_theta = rope_theta * float(fac)
        except Exception:
            pass
        attn_scale = float(getattr(self.cfg, "rope_attention_scaling", 1.0) or 1.0)
        cos, sin = runtime_resolve_rotary_embedding(
            reference=x,
            head_dim=head_dim,
            base_theta=rope_theta,
            attention_scaling=attn_scale,
            position_ids=position_ids,
        )

        if cache is None:
            for blk in self.blocks:
                x = blk(x, mask, None, (cos, sin), position_ids)
        else:
            for i, blk in enumerate(self.blocks):
                layer_cache = cache.layer(i)
                x = blk(x, mask, layer_cache, (cos, sin), position_ids)
        x = apply_native_norm(x, self.norm)
        logits = runtime_linear(x, self.lm_head.weight, self.lm_head.bias)
        if return_dict:
            return {"logits": logits, "last_hidden_state": x}
        return logits

    def get_output_embeddings(self):
        return self.lm_head

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int = 128,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | list[int] | None = None,
        no_repeat_ngram_size: int = 0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        attention_mask: torch.Tensor | None = None,
        return_dict: bool = True,
    ):
        eos_id = (
            int(eos_token_id)
            if isinstance(eos_token_id, int)
            else (int(eos_token_id[0]) if isinstance(eos_token_id, (list, tuple)) and eos_token_id else None)
        )
        cfg = GenerationConfig(
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
            temperature=float(temperature),
            top_k=(int(top_k) if top_k is not None else None),
            top_p=(float(top_p) if top_p is not None else None),
            eos_id=eos_id,
            no_repeat_ngram=0,
            repetition_penalty=float(repetition_penalty),
            presence_penalty=float(presence_penalty),
            frequency_penalty=float(frequency_penalty),
        )
        seq = engine_generate(
            self,
            input_ids.to(next(self.parameters()).device),
            attention_mask=(attention_mask.to(next(self.parameters()).device) if attention_mask is not None else None),
            config=cfg,
        )
        if return_dict:
            return {"sequences": seq}
        return seq
