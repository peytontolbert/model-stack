from __future__ import annotations

from typing import Any, Dict, List, Optional


class BaseTokenizer:
    def encode(self, text: str) -> List[int]:  # pragma: no cover - interface
        raise NotImplementedError

    def decode(self, ids: List[int]) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def info(self) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


class HFTokenizer(BaseTokenizer):
    def __init__(self, name_or_path: str) -> None:
        try:
            from transformers import AutoTokenizer  # type: ignore
        except Exception as e:  # pragma: no cover - optional dep
            raise RuntimeError("Install 'transformers' to use HFTokenizer") from e
        self._name = str(name_or_path)
        self._tok = AutoTokenizer.from_pretrained(self._name)
        if getattr(self._tok, "pad_token", None) is None:
            try:
                self._tok.pad_token = self._tok.eos_token
            except Exception:
                pass

    def encode(self, text: str) -> List[int]:
        return list(self._tok.encode(text, add_special_tokens=False))

    def decode(self, ids: List[int]) -> str:
        return str(self._tok.decode(ids))

    def info(self) -> Dict[str, Any]:
        return {"type": "hf", "name_or_path": self._name}


class WhitespaceTokenizer(BaseTokenizer):
    def __init__(self) -> None:
        # Extremely simple and not suitable for production; stable mapping per run
        self._vocab: Dict[str, int] = {}
        self._unk = 0
        self._vocab["<unk>"] = self._unk

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for tok in text.strip().split():
            if tok not in self._vocab:
                self._vocab[tok] = len(self._vocab)
            ids.append(self._vocab.get(tok, self._unk))
        return ids

    def decode(self, ids: List[int]) -> str:
        inv = {v: k for k, v in self._vocab.items()}
        return " ".join(inv.get(i, "<unk>") for i in ids)

    def info(self) -> Dict[str, Any]:
        return {"type": "whitespace"}


def get_tokenizer(name_or_path: Optional[str] = None) -> BaseTokenizer:
    if name_or_path is None or str(name_or_path).strip() == "":
        return WhitespaceTokenizer()
    return HFTokenizer(str(name_or_path))


__all__ = ["BaseTokenizer", "HFTokenizer", "WhitespaceTokenizer", "get_tokenizer"]


