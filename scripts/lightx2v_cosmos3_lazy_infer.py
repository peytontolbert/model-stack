#!/usr/bin/env python
from __future__ import annotations

"""Run LightX2V Cosmos3 with model-stack's narrow lazy-load compatibility patch."""

from lightx2v.models.networks.base_model import BaseTransformerModel
from lightx2v.models.networks.cosmos3 import model as cosmos3_model


def _cosmos3_lazy_compatible_init(self, model_path, config, device):
    if config.get("cpu_offload", False) and config.get("offload_granularity", "block") != "block":
        raise NotImplementedError("Cosmos3 LightX2V native transformer supports only block-level cpu_offload.")
    BaseTransformerModel.__init__(self, model_path, config, device)
    self._init_infer_class()
    self._init_weights()
    self._init_infer()


cosmos3_model.Cosmos3TransformerModel.__init__ = _cosmos3_lazy_compatible_init

from lightx2v.infer import main  # noqa: E402


if __name__ == "__main__":
    main()
