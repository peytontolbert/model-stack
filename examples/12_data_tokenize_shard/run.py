from __future__ import annotations

from pathlib import Path

from data.tokenizer import get_tokenizer


def main() -> None:
    tok = get_tokenizer(None)
    text = "hello tiny transformer repo examples"
    ids = tok.encode(text)
    dec = tok.decode(ids)
    print({"text": text, "ids": ids, "decoded": dec})


if __name__ == "__main__":
    main()


