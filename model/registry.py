from typing import Callable

from specs.config import ModelConfig
from .factory import build_causal_lm, build_encoder, build_prefix_lm, build_seq2seq


_MODEL_REGISTRY: dict[str, Callable[..., object]] = {}


def register_model(name: str, builder: Callable[..., object]) -> None:
    key = str(name).lower()
    _MODEL_REGISTRY[key] = builder


def get_model_builder(name: str) -> Callable[..., object]:
    key = str(name).lower()
    if key not in _MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' is not registered")
    return _MODEL_REGISTRY[key]


def build(name: str, cfg: ModelConfig, **kwargs):
    return get_model_builder(name)(cfg, **kwargs)


# Pre-register common models
try:
    register_model("causal", lambda cfg, **kw: build_causal_lm(cfg, **kw))
    register_model("causal-lm", lambda cfg, **kw: build_causal_lm(cfg, **kw))
    register_model("llama", lambda cfg, **kw: build_causal_lm(cfg, block="llama", **kw))
    register_model("gpt", lambda cfg, **kw: build_causal_lm(cfg, block="gpt", **kw))
    register_model("prefix", lambda cfg, **kw: build_prefix_lm(cfg, **kw))
    register_model("encoder", lambda cfg, **kw: build_encoder(cfg, **kw))
    register_model("seq2seq", lambda cfg, **kw: build_seq2seq(cfg, **kw))
except Exception:
    # Safe to ignore at import time if dependencies aren't ready
    pass


