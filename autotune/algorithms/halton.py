from __future__ import annotations

from typing import Any, Dict, Iterable, List

from ..spaces import Choice, IntRange, LogUniform, SearchSpace, Uniform


def _radical_inverse(base: int, index: int) -> float:
    inv_base = 1.0 / float(base)
    f = inv_base
    result = 0.0
    i = index
    while i > 0:
        result += (i % base) * f
        i //= base
        f *= inv_base
    return min(max(result, 0.0), 1.0 - 1e-12)


def _first_primes(n: int) -> List[int]:
    primes: List[int] = []
    x = 2
    while len(primes) < n:
        is_prime = True
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(x)
        x += 1
    return primes


def _decode_from_unit(space: SearchSpace, unit_vector: List[float]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    keys = list(space.parameters.keys())
    for i, name in enumerate(keys):
        u = float(unit_vector[i])
        spec = space.parameters[name]
        if isinstance(spec, Choice):
            n = max(1, len(spec.options))
            idx = min(int(u * n), n - 1)
            params[name] = list(spec.options)[idx]
        elif isinstance(spec, IntRange):
            n = max(1, (spec.stop - spec.start + (spec.step - 1)) // spec.step)
            idx = min(int(u * n), n - 1)
            params[name] = spec.start + idx * spec.step
        elif isinstance(spec, Uniform):
            params[name] = spec.low + u * (spec.high - spec.low)
        elif isinstance(spec, LogUniform):
            import math
            a = math.log(spec.low, spec.base)
            b = math.log(spec.high, spec.base)
            params[name] = spec.base ** (a + u * (b - a))
        else:
            try:
                params[name] = next(iter(spec.grid()))  # type: ignore[attr-defined]
            except Exception:
                params[name] = None
    return params


class HaltonSearch:
    def __init__(self, seed: int | None = None, leap: int | None = None, scramble: bool = False) -> None:
        self.seed = seed
        self.leap = leap
        self.scramble = scramble  # placeholder; simple implementation does not scramble
        self._bases: List[int] = []
        self._start_index: int = 1

    def reset(self, space: SearchSpace, seed: int) -> None:  # noqa: ARG002
        dim = max(1, len(space.parameters))
        self._bases = _first_primes(dim)
        s = self.seed if self.seed is not None else seed
        # Use seed to select a starting index and leap to decorrelate suggestions across runs
        self._start_index = max(1, (s % 10000) + 1)
        if self.leap is None:
            self._leap = max(1, (s % 13) + 1)
        else:
            self._leap = max(1, int(self.leap))

    def suggestions(self, space: SearchSpace, max_trials: int) -> Iterable[Dict[str, Any]]:
        if not self._bases:
            self.reset(space, seed=0)
        dim = max(1, len(space.parameters))
        for t in range(int(max_trials)):
            idx = self._start_index + t * self._leap
            vec = [_radical_inverse(self._bases[d], idx) for d in range(dim)]
            yield _decode_from_unit(space, vec)


