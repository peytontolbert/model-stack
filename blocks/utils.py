from typing import Any
def getattr_nested(obj: Any, path: str) -> Any:
    cur: Any = obj
    for tok in str(path).split("."):
        if not tok:
            continue
        cur = getattr(cur, tok)
    return cur

