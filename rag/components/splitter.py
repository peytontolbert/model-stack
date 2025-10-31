from typing import Iterable, List


class BasicSplitter:
    def __init__(self, max_tokens: int = 256):
        self.max_tokens = int(max_tokens)

    def split(self, texts: Iterable[str]) -> List[str]:
        out: List[str] = []
        for t in texts:
            if len(t) <= self.max_tokens:
                out.append(t)
            else:
                start = 0
                while start < len(t):
                    out.append(t[start : start + self.max_tokens])
                    start += self.max_tokens
        return out


