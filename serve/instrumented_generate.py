import time
from viz.session import VizSession

from .generate import generate


def generate_instrumented(model, input_ids, viz: VizSession, **kw):
    t0 = time.time()
    out = generate(model, input_ids, **kw)
    viz.log_scalar(kw.get("step", 0), "serve.latency_ms", (time.time() - t0) * 1e3)
    return out


