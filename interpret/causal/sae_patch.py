from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Optional

import torch
import torch.nn as nn

from interpret.features.sae import SparseAutoencoder
from interpret.features.sae_ops import sae_mask_features, sae_boost_features


@contextmanager
def sae_feature_mask(
    model: nn.Module,
    layer_index: int,
    sae: SparseAutoencoder,
    *,
    drop_codes: Optional[Iterable[int]] = None,
    keep_codes: Optional[Iterable[int]] = None,
    boost_codes: Optional[Iterable[int]] = None,
    boost_factor: float = 1.0,
    time_slice: Optional[slice] = None,
):
    """Hook block output to apply SAE code masking/boosting.

    Exactly one of drop_codes/keep_codes/boost_codes should be provided.
    time_slice can restrict to certain positions (e.g., slice(-1, None)).
    """
    blk = model.blocks[int(layer_index)]

    def hook(_m: nn.Module, _inp, out: torch.Tensor):
        x = out
        if time_slice is None:
            xs = x
        else:
            xs = x[:, time_slice]
        if drop_codes is not None:
            xs_hat = sae_mask_features(sae, xs, drop_codes, invert=False)
        elif keep_codes is not None:
            xs_hat = sae_mask_features(sae, xs, keep_codes, invert=True)
        elif boost_codes is not None:
            xs_hat = sae_boost_features(sae, xs, boost_codes, factor=boost_factor)
        else:
            return out
        if time_slice is None:
            return xs_hat
        x_clone = x.clone()
        x_clone[:, time_slice] = xs_hat
        return x_clone

    h = blk.register_forward_hook(hook)
    try:
        yield
    finally:
        try:
            h.remove()
        except Exception:
            pass


