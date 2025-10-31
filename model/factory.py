from specs.config import ModelConfig
from .causal import CausalLM
from .prefix_lm import PrefixCausalLM
from .encoder import EncoderModel
from .seq2seq import EncoderDecoderLM


def build_causal_lm(cfg: ModelConfig, *, block: str = "llama", compress: dict | None = None, **kwargs) -> CausalLM:
    return CausalLM(cfg, block_variant=block, compress=compress, **kwargs)


def build_prefix_lm(cfg: ModelConfig, *, block: str = "prefix-lm", compress: dict | None = None, **kwargs) -> PrefixCausalLM:
    return PrefixCausalLM(cfg, block_variant=block, compress=compress, **kwargs)


def build_encoder(cfg: ModelConfig, *, compress: dict | None = None, **kwargs) -> EncoderModel:
    return EncoderModel(cfg, compress=compress, **kwargs)


def build_seq2seq(cfg: ModelConfig, *, compress: dict | None = None, **kwargs) -> EncoderDecoderLM:
    return EncoderDecoderLM(cfg, compress=compress, **kwargs)


def build_model(cfg: ModelConfig, *, task: str = "causal-lm", block: str = "llama", compress: dict | None = None, **kwargs):
    name = task.lower()
    if name in ("causal", "causal-lm", "gpt"):
        return build_causal_lm(cfg, block=block, compress=compress, **kwargs)
    if name in ("prefix", "prefix-lm"):
        return build_prefix_lm(cfg, block=block, compress=compress, **kwargs)
    if name in ("encoder", "bert"):
        return build_encoder(cfg, compress=compress, **kwargs)
    if name in ("seq2seq", "encdec", "t5"):
        return build_seq2seq(cfg, compress=compress, **kwargs)
    # Default
    return build_causal_lm(cfg, block=block, compress=compress, **kwargs)


