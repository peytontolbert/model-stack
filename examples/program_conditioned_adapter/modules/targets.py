from typing import Optional, Dict, Tuple

def parse_target_shapes(arg: Optional[str]) -> Optional[Dict[str, Tuple[int, int]]]:
    if not arg:
        return None
    result: Dict[str, Tuple[int, int]] = {}
    try:
        parts = [p.strip() for p in str(arg).split(",") if p.strip()]
        for p in parts:
            if "=" not in p or ":" not in p:
                continue
            name, dims = p.split("=", 1)
            a, b = dims.split(":", 1)
            result[name.strip()] = (int(a), int(b))
        return result or None
    except Exception:
        return None

