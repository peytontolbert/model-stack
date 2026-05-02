import torch
import torch.nn as nn

from compress import apply_compression
from runtime.block_init import init_transformer_stack
from runtime.block_modules import DecoderBlock, EncoderBlock
from runtime.block_schedules import drop_path_linear
from runtime.blocks import apply_native_norm, execute_decoder_stack, execute_encoder_stack
from runtime.ops import embedding as runtime_embedding
from runtime.ops import linear_module as runtime_linear_module
from specs.config import ModelConfig


class EncoderDecoderLM(nn.Module):
    def __init__(
        self,
        cfg: ModelConfig,
        *,
        drop_path_max_enc: float = 0.0,
        drop_path_max_dec: float = 0.0,
        init_recipe_enc: str | None = None,
        init_recipe_dec: str | None = None,
        tie_embeddings: bool = False,
        vocab_size: int | None = None,
        compress: dict | None = None,
    ):
        super().__init__()
        self.cfg = cfg
        vocab = vocab_size if vocab_size is not None else cfg.vocab_size
        self.enc_embed = nn.Embedding(vocab, cfg.d_model)
        self.dec_embed = nn.Embedding(vocab, cfg.d_model)
        if bool(getattr(cfg, "encoder_position_embeddings", False)):
            max_positions = int(getattr(cfg, "max_position_embeddings", None) or 4096)
            self.enc_pos_embed = nn.Embedding(max_positions, cfg.d_model)
        else:
            self.enc_pos_embed = None
        if tie_embeddings:
            self.dec_embed.weight = self.enc_embed.weight
        enc_schedule = drop_path_linear(cfg.n_layers, drop_path_max_enc)
        self.encoder = nn.ModuleList([EncoderBlock(cfg, drop_path=enc_schedule[i]) for i in range(cfg.n_layers)])
        dec_schedule = drop_path_linear(cfg.n_layers, drop_path_max_dec)
        self.decoder = nn.ModuleList([DecoderBlock(cfg, drop_path=dec_schedule[i]) for i in range(cfg.n_layers)])
        self.enc_norm = nn.LayerNorm(cfg.d_model)
        self.dec_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab, bias=False)
        retrieval_head_dim = int(getattr(cfg, "retrieval_head_dim", None) or 0)
        if retrieval_head_dim > 0:
            self.retrieval_query_head = nn.Linear(cfg.d_model, retrieval_head_dim, bias=False)
            self.retrieval_doc_head = nn.Linear(cfg.d_model, retrieval_head_dim, bias=False)
        else:
            self.retrieval_query_head = None
            self.retrieval_doc_head = None
        if bool(getattr(cfg, "agent_policy_heads", False)):
            self.agent_policy_heads = nn.ModuleDict(
                {
                    "query_confidence": nn.Linear(cfg.d_model, 1),
                    "retrieval_coverage": nn.Linear(cfg.d_model, 1),
                    "ood_query": nn.Linear(cfg.d_model, 1),
                    "ood_evidence": nn.Linear(cfg.d_model, 1),
                    "answer_confidence": nn.Linear(cfg.d_model, 1),
                    "needs_verification": nn.Linear(cfg.d_model, 1),
                    "paper_action_validity": nn.Linear(cfg.d_model, 1),
                }
            )
        else:
            self.agent_policy_heads = None
        if init_recipe_enc is not None:
            init_transformer_stack(self.encoder, recipe=init_recipe_enc)
        if init_recipe_dec is not None:
            init_transformer_stack(self.decoder, recipe=init_recipe_dec)
        if compress is not None:
            lora_cfg = compress.get("lora") if isinstance(compress, dict) else None
            quant_cfg = compress.get("quant") if isinstance(compress, dict) else None
            self._compression_summary = apply_compression(self, lora=lora_cfg, quant=quant_cfg)
        else:
            self._compression_summary = None

    def encode(self, input_ids: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = runtime_embedding(self.enc_embed.weight, input_ids, self.enc_embed.padding_idx)
        if self.enc_pos_embed is not None:
            if input_ids.shape[1] > self.enc_pos_embed.num_embeddings:
                raise ValueError(
                    f"encoder input length {input_ids.shape[1]} exceeds "
                    f"encoder position capacity {self.enc_pos_embed.num_embeddings}"
                )
            positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            positions = positions.expand(input_ids.shape[0], input_ids.shape[1])
            x = x + runtime_embedding(self.enc_pos_embed.weight, positions, self.enc_pos_embed.padding_idx)
        x = execute_encoder_stack(self.encoder, x, padding_mask)
        return apply_native_norm(x, self.enc_norm)

    def encode_pooled(self, input_ids: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.encode(input_ids, padding_mask)
        if padding_mask is None:
            return hidden.mean(dim=1)
        mask = padding_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def retrieval_query_embedding(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = self.encode_pooled(input_ids, padding_mask)
        if self.retrieval_query_head is not None:
            pooled = self.retrieval_query_head(pooled)
        return torch.nn.functional.normalize(pooled.float(), dim=-1)

    def retrieval_doc_embedding(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = self.encode_pooled(input_ids, padding_mask)
        if self.retrieval_doc_head is not None:
            pooled = self.retrieval_doc_head(pooled)
        return torch.nn.functional.normalize(pooled.float(), dim=-1)

    def agent_policy_logits(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.agent_policy_heads is None:
            return {}
        pooled = self.encode_pooled(input_ids, padding_mask)
        return {name: head(pooled).squeeze(-1).float() for name, head in self.agent_policy_heads.items()}

    def decode(
        self,
        input_ids: torch.Tensor,
        memory: torch.Tensor,
        self_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        cache=None,
    ) -> torch.Tensor:
        x = runtime_embedding(self.dec_embed.weight, input_ids, self.dec_embed.padding_idx)
        x = execute_decoder_stack(
            self.decoder,
            x,
            memory,
            self_mask,
            memory_mask,
            cache,
        )
        return apply_native_norm(x, self.dec_norm)

    def forward(
        self,
        enc_input_ids: torch.Tensor,
        dec_input_ids: torch.Tensor,
        enc_padding_mask: torch.Tensor | None = None,
        dec_self_mask: torch.Tensor | None = None,
        cache=None,
    ) -> torch.Tensor:
        memory = self.encode(enc_input_ids, enc_padding_mask)
        x = self.decode(dec_input_ids, memory, dec_self_mask, enc_padding_mask, cache)
        return runtime_linear_module(x, self.lm_head)


__all__ = ["EncoderDecoderLM"]
