import warnings
import hashlib
import torch


def safetensor_dump(tensors: dict[str, torch.Tensor], path: str):
    try:
        from safetensors.torch import save_file  # type: ignore
        save_file(tensors, path)
    except Exception:
        warnings.warn("safetensors not available; falling back to torch.save")
        torch.save(tensors, path)


def safetensor_load(path: str) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file  # type: ignore
        return load_file(path)
    except Exception:
        warnings.warn("safetensors not available; falling back to torch.load")
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            return obj
        raise TypeError("Loaded object is not a dict[str, Tensor]")


def stable_tensor_hash(x: torch.Tensor, *, mode: str = "content") -> str:
    h = hashlib.blake2b(digest_size=20)
    meta = f"{str(x.dtype)}|{tuple(x.shape)}|{tuple(x.stride())}".encode()
    if mode == "meta":
        h.update(meta)
        return h.hexdigest()
    xb = x.detach().cpu().contiguous()
    h.update(meta)
    h.update(xb.numpy().tobytes())
    return h.hexdigest()


