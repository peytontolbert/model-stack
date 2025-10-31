import torch


def remat(fn, policy: str = "auto"):
    """Return a wrapped function that applies gradient checkpointing to `fn`.

    policy is a placeholder ("auto"|"segment"); behavior is identical for now.
    """

    def wrapped(*args, **kwargs):
        return torch.utils.checkpoint.checkpoint(fn, *args, **kwargs)

    return wrapped


def checkpoint_sequential(layers, chunk_size: int):
    """Thin wrapper around torch.utils.checkpoint.checkpoint_sequential."""
    return torch.utils.checkpoint.checkpoint_sequential(layers, chunk_size)


