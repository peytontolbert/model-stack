import torch
import torch.nn as nn

from specs.config import ModelConfig
from compress import apply_compression
from blocks.factory import build_block_stack
from tensor.norms import RMSNorm
from tensor.masking import build_padding_mask, broadcast_mask, combine_masks, to_additive_mask, create_causal_mask
from tensor.positional import RotaryEmbeddingHF
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
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
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
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cache=None,
        position_ids: torch.Tensor | None = None,
        return_dict: bool = False,
    ):
        x = self.embed(input_ids)
        B, T, _ = x.shape
        # Align attention_mask length with current sequence length (e.g., during decoding T may be 1)
        if attention_mask is not None:
            try:
                if attention_mask.shape[-1] != T:
                    attention_mask = attention_mask[:, -T:]
            except Exception:
                pass
        # position ids: honor provided absolute ids for KV-cache decoding; else default 0..T-1
        if position_ids is None:
            position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        # HF-like additive mask (B,1,T,S)
        mask = create_causal_mask(input_embeds=x, attention_mask=attention_mask)
        # Rotary embeddings
        head_dim = getattr(self.cfg, "head_dim", None) or int(self.cfg.d_model // self.cfg.n_heads)
        rope_theta = float(getattr(self.cfg, "rope_theta", 1e6))
        attn_scale = float(getattr(self.cfg, "rope_attention_scaling", 1.0) or 1.0)
        rope = RotaryEmbeddingHF(head_dim=head_dim, base_theta=rope_theta, attention_scaling=attn_scale, device=x.device)
        cos, sin = rope.forward(x, position_ids=position_ids)

        if cache is None:
            for blk in self.blocks:
                x = blk(x, mask, None, (cos, sin), position_ids)
        else:
            for i, blk in enumerate(self.blocks):
                layer_cache = cache.layer(i)
                x = blk(x, mask, layer_cache, (cos, sin), position_ids)
        x = self.norm(x)
        logits = self.lm_head(x)
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
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        attention_mask: torch.Tensor | None = None,
        return_dict: bool = True,
    ):
        cfg = GenerationConfig(
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=int(top_k) if top_k is not None else None,
            top_p=(float(top_p) if (top_p is not None and top_p < 1.0) else None),
            eos_id=(int(eos_token_id) if isinstance(eos_token_id, int) else (int(eos_token_id[0]) if isinstance(eos_token_id, (list, tuple)) and eos_token_id else None)),
            no_repeat_ngram=int(no_repeat_ngram_size) if no_repeat_ngram_size else 0,
            presence_penalty=float(presence_penalty),
            frequency_penalty=float(frequency_penalty),
        )
        sampler = None
        if not do_sample:
            def _greedy(logits: torch.Tensor) -> torch.Tensor:
                return torch.argmax(logits, dim=-1, keepdim=True)
            sampler = _greedy
        seq = engine_generate(self, input_ids, attention_mask=attention_mask, config=cfg, sampler=sampler)
        if return_dict:
            return {"sequences": seq}
        return seq


