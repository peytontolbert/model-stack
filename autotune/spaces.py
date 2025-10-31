from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union


class SearchSpace:
    def __init__(self, parameters: Mapping[str, "_ParamSpace"]) -> None:
        self.parameters: Dict[str, _ParamSpace] = dict(parameters)

    def sample(self, rng: random.Random) -> Dict[str, Any]:
        return {name: space.sample(rng) for name, space in self.parameters.items()}

    def is_finite(self) -> bool:
        return all(space.is_finite() for space in self.parameters.values())

    def grid(self) -> Iterator[Dict[str, Any]]:
        keys = list(self.parameters.keys())
        grids: List[List[Any]] = [list(self.parameters[k].grid()) for k in keys]
        if not grids:
            yield {}
            return

        def _product(prefix: List[Any], idx: int) -> Iterator[List[Any]]:
            if idx == len(grids):
                yield prefix
                return
            for v in grids[idx]:
                yield from _product(prefix + [v], idx + 1)

        for vals in _product([], 0):
            yield {k: v for k, v in zip(keys, vals)}


class _ParamSpace:
    def sample(self, rng: random.Random) -> Any:  # pragma: no cover
        raise NotImplementedError

    def is_finite(self) -> bool:
        return False

    def grid(self) -> Iterable[Any]:  # pragma: no cover
        raise NotImplementedError


@dataclass
class Choice(_ParamSpace):
    options: Sequence[Any]

    def sample(self, rng: random.Random) -> Any:
        return rng.choice(list(self.options))

    def is_finite(self) -> bool:
        return True

    def grid(self) -> Iterable[Any]:
        return list(self.options)


@dataclass
class IntRange(_ParamSpace):
    start: int
    stop: int
    step: int = 1

    def sample(self, rng: random.Random) -> int:
        n = max(1, (self.stop - self.start + (self.step - 1)) // self.step)
        idx = rng.randrange(n)
        return self.start + idx * self.step

    def is_finite(self) -> bool:
        return True

    def grid(self) -> Iterable[int]:
        return list(range(self.start, self.stop, self.step))


@dataclass
class Uniform(_ParamSpace):
    low: float
    high: float

    def sample(self, rng: random.Random) -> float:
        return rng.uniform(self.low, self.high)

    def grid(self) -> Iterable[float]:
        # Infinite; expose a tiny representative grid for debugging
        mid = 0.5 * (self.low + self.high)
        return [self.low, mid, self.high]


@dataclass
class LogUniform(_ParamSpace):
    low: float
    high: float
    base: float = 10.0

    def sample(self, rng: random.Random) -> float:
        a = math.log(self.low, self.base)
        b = math.log(self.high, self.base)
        return self.base ** rng.uniform(a, b)

    def grid(self) -> Iterable[float]:
        a = math.log(self.low, self.base)
        b = math.log(self.high, self.base)
        k = max(1, int(round(b - a)))
        return [self.base ** (a + i * (b - a) / k) for i in range(k + 1)]


