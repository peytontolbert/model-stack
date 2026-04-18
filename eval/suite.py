from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

import torch

from .loop import evaluate_lm_next_token
from .bench import benchmark_forward, benchmark_generate
from .calibration import evaluate_ece


@dataclass
class SuiteResult:
    ppl: Dict[str, Any]
    bench_forward: Dict[str, Any]
    bench_generate: Dict[str, Any]
    ece: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ppl": self.ppl,
            "bench_forward": self.bench_forward,
            "bench_generate": self.bench_generate,
            "ece": self.ece,
        }


def run_basic_suite(
    model: torch.nn.Module,
    loader,
    *,
    device: Optional[str | torch.device] = None,
    outdir: Optional[str] = None,
    kv_cache_factory=None,
    do_sample: bool | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_id: int | None = None,
    no_repeat_ngram: int = 0,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    sliding_window: int | None = None,
    beam_size: int = 1,
    length_penalty: float = 1.0,
    cache_backend: str | None = None,
) -> SuiteResult:
    del outdir
    dev = torch.device(device) if device is not None else next(model.parameters()).device
    model = model.to(dev)
    r_ppl = evaluate_lm_next_token(model, loader, device=dev)
    r_fwd = benchmark_forward(model, device=dev)
    r_gen = benchmark_generate(
        model,
        device=dev,
        kv_cache_factory=kv_cache_factory,
        do_sample=do_sample,
        temperature=float(temperature),
        top_k=top_k,
        top_p=top_p,
        eos_id=eos_id,
        no_repeat_ngram=int(no_repeat_ngram),
        repetition_penalty=float(repetition_penalty),
        presence_penalty=float(presence_penalty),
        frequency_penalty=float(frequency_penalty),
        sliding_window=sliding_window,
        beam_size=int(beam_size),
        length_penalty=float(length_penalty),
        cache_backend=cache_backend,
    )
    r_ece = evaluate_ece(model, loader, device=dev)
    return SuiteResult(
        ppl={"nll": r_ppl.nll, "ppl": r_ppl.ppl, "acc": r_ppl.acc, "tokens": r_ppl.num_tokens},
        bench_forward={"tokens_per_sec": r_fwd.tokens_per_sec, "latency_ms": r_fwd.latency_ms},
        bench_generate={"tokens_per_sec": r_gen.tokens_per_sec, "latency_ms": r_gen.latency_ms},
        ece={"ece": r_ece.ece, "tokens": r_ece.num_tokens},
    )

