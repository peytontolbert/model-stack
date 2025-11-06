from typing import Dict

def targets_map(backend: str = "local") -> Dict[str, str]:
    if backend == "local":
        return {
            "q_proj": "attn.w_q",
            "k_proj": "attn.w_k",
            "v_proj": "attn.w_v",
            "o_proj": "attn.w_o",
            "up_proj": "mlp.w_in",
            "gate_proj": "mlp.w_in",
            "down_proj": "mlp.w_out",
        }
    return {
        "q_proj": "self_attn.q_proj",
        "k_proj": "self_attn.k_proj",
        "v_proj": "self_attn.v_proj",
        "o_proj": "self_attn.o_proj",
        "up_proj": "mlp.up_proj",
        "down_proj": "mlp.down_proj",
        "gate_proj": "mlp.gate_proj",
    }