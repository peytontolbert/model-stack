from __future__ import annotations

from typing import Callable

from specs.config import ModelConfig


def build_causal_lm(cfg: ModelConfig, *, block: str = "llama", compress: dict | None = None, **kwargs):
    from model.causal import CausalLM

    return CausalLM(cfg, block_variant=block, compress=compress, **kwargs)


def build_prefix_lm(cfg: ModelConfig, *, block: str = "llama", compress: dict | None = None, **kwargs):
    from model.prefix_lm import PrefixCausalLM

    return PrefixCausalLM(cfg, block_variant=block, compress=compress, **kwargs)


def build_encoder(cfg: ModelConfig, *, compress: dict | None = None, **kwargs):
    from model.encoder import EncoderModel

    return EncoderModel(cfg, compress=compress, **kwargs)


def build_seq2seq(cfg: ModelConfig, *, compress: dict | None = None, **kwargs):
    from model.seq2seq import EncoderDecoderLM

    return EncoderDecoderLM(cfg, compress=compress, **kwargs)


def build_model(
    cfg: ModelConfig,
    *,
    task: str = "causal-lm",
    block: str = "llama",
    compress: dict | None = None,
    **kwargs,
):
    name = task.lower()
    if name in ("causal", "causal-lm", "gpt"):
        return build_causal_lm(cfg, block=block, compress=compress, **kwargs)
    if name in ("prefix", "prefix-lm"):
        return build_prefix_lm(cfg, block=block, compress=compress, **kwargs)
    if name in ("encoder", "bert"):
        return build_encoder(cfg, compress=compress, **kwargs)
    if name in ("seq2seq", "encdec", "t5"):
        return build_seq2seq(cfg, compress=compress, **kwargs)
    return build_causal_lm(cfg, block=block, compress=compress, **kwargs)


_MODEL_REGISTRY: dict[str, Callable[..., object]] = {}


def register_model(name: str, builder: Callable[..., object]) -> None:
    _MODEL_REGISTRY[str(name).lower()] = builder


def get_model_builder(name: str) -> Callable[..., object]:
    key = str(name).lower()
    if key not in _MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' is not registered")
    return _MODEL_REGISTRY[key]


def build_registered_model(name: str, cfg: ModelConfig, **kwargs):
    return get_model_builder(name)(cfg, **kwargs)


def _register_default_model_builders() -> None:
    _MODEL_REGISTRY.setdefault("causal", lambda cfg, **kw: build_causal_lm(cfg, **kw))
    _MODEL_REGISTRY.setdefault("causal-lm", lambda cfg, **kw: build_causal_lm(cfg, **kw))
    _MODEL_REGISTRY.setdefault("llama", lambda cfg, **kw: build_causal_lm(cfg, block="llama", **kw))
    _MODEL_REGISTRY.setdefault("gpt", lambda cfg, **kw: build_causal_lm(cfg, block="gpt", **kw))
    _MODEL_REGISTRY.setdefault("prefix", lambda cfg, **kw: build_prefix_lm(cfg, **kw))
    _MODEL_REGISTRY.setdefault("encoder", lambda cfg, **kw: build_encoder(cfg, **kw))
    _MODEL_REGISTRY.setdefault("seq2seq", lambda cfg, **kw: build_seq2seq(cfg, **kw))


_register_default_model_builders()
