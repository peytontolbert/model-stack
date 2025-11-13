import torch
import torch.nn as nn

from specs.config import ModelConfig
from compress import apply_compression
from blocks.factory import build_block_stack
from tensor.norms import RMSNorm
from tensor.masking import build_padding_mask, broadcast_mask, combine_masks, to_additive_mask, create_causal_mask, lengths_from_attention_mask
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
        x = inputs_embeds if inputs_embeds is not None else self.embed(input_ids)
        B, T, _ = x.shape
        # position ids: prefer provided, else derive from cache_position if available, else 0..T-1
        if position_ids is None:
            if cache_position is not None:
                # cache_position should represent absolute positions for current tokens
                position_ids = cache_position.view(1, -1).expand(B, -1)
            elif cache is not None and hasattr(cache, "layer"):
                try:
                    past_len = int(cache.layer(0).length())
                except Exception:
                    past_len = 0
                start = past_len
                position_ids = torch.arange(start, start + T, device=x.device).view(1, -1).expand(B, -1)
            else:
                # Derive absolute positions from attention_mask if provided, else 0..T-1
                if attention_mask is not None:
                    try:
                        lengths = lengths_from_attention_mask(attention_mask.to(torch.long))  # (B,)
                        starts = (lengths - T).clamp_min(0).view(B, 1)
                        base = torch.arange(T, device=x.device).view(1, T)
                        position_ids = (base + starts).to(dtype=torch.long)
                    except Exception:
                        position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
                else:
                    position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        # HF-like additive mask (B,1,T,S)
        mask = create_causal_mask(input_embeds=x, attention_mask=attention_mask, cache_position=cache_position, position_ids=position_ids)
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
        rope = RotaryEmbeddingHF(
            head_dim=head_dim,
            base_theta=rope_theta,
            attention_scaling=attn_scale,
            device=x.device,
            scaling_type=getattr(self.cfg, "rope_scaling_type", None),
            scaling_factor=getattr(self.cfg, "rope_scaling_factor", None),
            original_max_position_embeddings=getattr(self.cfg, "rope_scaling_original_max_position_embeddings", None),
            low_freq_factor=getattr(self.cfg, "rope_scaling_low_freq_factor", None),
            high_freq_factor=getattr(self.cfg, "rope_scaling_high_freq_factor", None),
        )
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
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        attention_mask: torch.Tensor | None = None,
        return_dict: bool = True,
    ):
        # Full recompute each step for HF parity; no KV cache
        import os, traceback
        trace = (os.getenv("GEN_TRACE", "0") == "1")
        device = next(self.parameters()).device
        seq = input_ids.to(device)
        attn = attention_mask.to(device) if attention_mask is not None else None
        eos_id = (int(eos_token_id) if isinstance(eos_token_id, int) else (int(eos_token_id[0]) if isinstance(eos_token_id, (list, tuple)) and eos_token_id else None))
       # if trace:
       #     try:
       #         print(f"[gen] start: device={device} dtype={next(self.parameters()).dtype} init_len={seq.shape[1]}")
        #    except Exception:
        #        pass
        for step in range(int(max_new_tokens)):
            try:
                out = self(input_ids=seq, attention_mask=attn, return_dict=True)
            except Exception:
            #    if trace:
            #        print(f"[gen] exception in forward at step={step} seq_len={seq.shape[1]}")
            #        traceback.print_exc()
                raise
            logits = out["logits"][:, -1, :]
            # Apply repetition / presence / frequency penalties (non-Transformers implementation)
            try:
                # seq shape: (B, T)
                B = int(seq.shape[0])
                V = int(logits.shape[-1])
                for b in range(B):
                    prev = seq[b]  # (T,)
                    if prev.numel() > 0:
                        # Repetition penalty (>1.0 discourages repeated tokens)
                        if repetition_penalty is not None and float(repetition_penalty) != 1.0:
                            rp = float(repetition_penalty)
                            # For tokens present in prev, scale logits: positive -> /= rp, negative -> *= rp
                            uniq = torch.unique(prev)
                            idx = uniq.to(logits.device)
                            vals = logits[b, idx]
                            pos = vals > 0
                            vals = torch.where(pos, vals / rp, vals * rp)
                            logits[b, idx] = vals
                        # Presence/frequency penalties (OpenAI-style)
                        pp = float(presence_penalty or 0.0)
                        fp = float(frequency_penalty or 0.0)
                        if (pp != 0.0) or (fp != 0.0):
                            # token counts in previous sequence
                            counts = torch.bincount(prev.to(torch.long), minlength=V).to(logits.device, logits.dtype)
                            # presence: subtract pp if token count > 0
                            if pp != 0.0:
                                pres_mask = (counts > 0).to(logits.dtype)
                                logits[b, :] = logits[b, :] - (pp * pres_mask)
                            # frequency: subtract fp * count
                            if fp != 0.0:
                                logits[b, :] = logits[b, :] - (fp * counts)
            except Exception:
                # Penalties are best-effort; ignore on failure to keep generation robust
                pass
            # Apply simple sampling policies
            if do_sample:
                x = logits.float()
                if temperature is not None and float(temperature) != 1.0:
                    x = x / float(max(1e-8, temperature))
                if top_k is not None and int(top_k) > 0:
                    kth = torch.topk(x, k=int(top_k), dim=-1).values[:, -1].unsqueeze(-1)
                    x = torch.where(x < kth, torch.full_like(x, float('-inf')), x)
                if top_p is not None and float(top_p) < 1.0:
                    probs = torch.softmax(x, dim=-1)
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum > float(top_p)
                    # keep at least one
                    mask[:, 0] = False
                    sorted_probs = torch.where(mask, torch.zeros_like(sorted_probs), sorted_probs)
                    probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
                    nxt = torch.multinomial(probs, num_samples=1)
                else:
                    probs = torch.softmax(x, dim=-1)
                    nxt = torch.multinomial(probs, num_samples=1)
            else:
                nxt = torch.argmax(logits, dim=-1, keepdim=True)
            #if trace:
            #    try:
            #        print(f"[gen] step={step} seq_len={seq.shape[1]} logits_dtype={logits.dtype} next_id={int(nxt[0,0].item())}")
            #    except Exception:
            #        pass
            seq = torch.cat([seq, nxt], dim=1)
            if attn is not None:
                ones = torch.ones_like(nxt, device=attn.device, dtype=attn.dtype)
                attn = torch.cat([attn, ones], dim=1)
            if eos_id is not None and int(nxt[0, 0].item()) == int(eos_id):
                break
        if return_dict:
            return {"sequences": seq}
        return seq


