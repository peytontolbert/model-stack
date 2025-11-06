from typing import Dict, Any, List, Optional
import os
import re
from model.inspect import detect_target_names_from_model_full

def infer_target_names(model_id: str) -> Dict[str, str]:
    names = detect_target_names_from_model_full(model_id, target_regex=None) or {}
    # names maps short -> path within the first layer subtree
    # Example: {"q_proj": "self_attn.q_proj", "o_proj": "self_attn.o_proj", ...}
    return names

def save_peft_like(out_dir: str, adapters: Dict[str, Any], *, r: int, alpha: int, target_modules: List[str], bias: str = "none", int8: bool = False, target_paths: Optional[Dict[str, str]] = None) -> None:
    """Write a minimal PEFT LoRA config + tensors for quick benchmarking.

    Note: This is a best-effort exporter; users may still need to map names depending on the model arch.
    """
    try:
        import json as _json
        cfg = {
            "peft_type": "LORA",
            "r": int(r),
            "lora_alpha": int(alpha),
            "target_modules": target_modules,
            "lora_dropout": 0.0,
            "bias": str(bias),
            "task_type": "CAUSAL_LM",
        }
        open(os.path.join(out_dir, "adapter_config.json"), "w", encoding="utf-8").write(_json.dumps(cfg, indent=2))
        # Save tensors in a stable torch format if available
        try:
            import torch as _torch  # type: ignore

            state: Dict[str, Any] = {}
            def _map_path(i: int, name: str) -> str:
                # Map target name to likely module path; best-effort for LLaMA-like arch
                if target_paths and name in target_paths:
                    return f"base_model.model.{target_paths[name]}"
                if name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    return f"base_model.model.model.layers.{i}.self_attn.{name}"
                elif name in ("up_proj", "down_proj", "gate_proj"):
                    return f"base_model.model.model.layers.{i}.mlp.{name}"
                else:
                    return f"base_model.model.model.layers.{i}.{name}"
            for i, layer in enumerate(adapters["layers"]):
                for name, tensors in layer.items():
                    base = _map_path(i, name)
                    A = _torch.from_numpy(tensors["A"]).contiguous()
                    B = _torch.from_numpy(tensors["B"]).contiguous()
                    if int8:
                        try:
                            # Per-tensor affine quantization
                            scale_A = float(A.abs().max().item() / 127.0) if A.numel() > 0 else 1.0
                            A = _torch.quantize_per_tensor(A, scale=max(scale_A, 1e-8), zero_point=0, dtype=_torch.qint8)
                            scale_B = float(B.abs().max().item() / 127.0) if B.numel() > 0 else 1.0
                            B = _torch.quantize_per_tensor(B, scale=max(scale_B, 1e-8), zero_point=0, dtype=_torch.qint8)
                        except Exception:
                            pass
                    state[f"{base}.lora_A.weight"] = A
                    state[f"{base}.lora_B.weight"] = B
            _torch.save(state, os.path.join(out_dir, "adapter_model.bin"))
        except Exception:
            pass
    except Exception:
        pass

