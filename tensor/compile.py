import contextlib
from typing import Iterable
import torch


try:
    allow_in_graph = torch._dynamo.allow_in_graph  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    def allow_in_graph(fn):
        return fn


def masked_fill_where(x: torch.Tensor, mask: torch.Tensor, fill_value: float) -> torch.Tensor:
    return torch.where(mask, torch.full_like(x, fill_value), x)


def infer_attn_shapes(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None):
    B, H, T, D = q.shape
    _B, _H, S, _D = k.shape
    assert (B == _B) and (H == _H) and (D == _D) and v.shape == k.shape
    if mask is not None:
        # broadcastable to (B,H,T,S)
        _ = (q.new_zeros(B, H, T, S) + mask.to(q.dtype)).shape
    return (B, H, S, D)


def graph_safe_seed(seed: int):
    torch.manual_seed(seed)
    # Avoid manual_seed_all during graph capture; users can use cuda_graph_seed_scope
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    except Exception:
        pass


@contextlib.contextmanager
def record_stream_guard(tensors: Iterable[torch.Tensor], stream):  # type: ignore[type-arg]
    try:
        import torch.cuda as cuda
        for t in tensors:
            if isinstance(t, torch.Tensor) and t.is_cuda:
                t.record_stream(stream)
        yield
    except Exception:
        # CPU no-op
        yield


@contextlib.contextmanager
def cuda_graph_seed_scope(seed: int):
    # Simple seed scope that avoids touching global manual_seed_all
    state = torch.random.get_rng_state()
    try:
        graph_safe_seed(seed)
        yield
    finally:
        torch.random.set_rng_state(state)


@contextlib.contextmanager
def nccl_stream_guard(stream):
    try:
        import torch.cuda as cuda
        prev = cuda.current_stream()
        if stream is not None:
            with cuda.stream(stream):
                yield
        else:
            yield
    except Exception:
        yield


def overlap_copy_compute(copy_fn, compute_fn, copy_stream=None, compute_stream=None):
    """Run copy_fn on copy_stream and compute_fn on compute_stream with best-effort overlap."""
    try:
        import torch.cuda as cuda
        cs = copy_stream or cuda.Stream()
        gs = compute_stream or cuda.Stream()
        with cuda.stream(cs):
            copy_fn()
        gs.wait_stream(cs)
        with cuda.stream(gs):
            return compute_fn()
    except Exception:
        # CPU fallback: sequential
        copy_fn()
        return compute_fn()


def cuda_graph_warmup(step_fn, *args, **kwargs):
    """Run step_fn once to initialize kernels/caches before capture."""
    try:
        return step_fn(*args, **kwargs)
    except Exception:
        return step_fn(*args, **kwargs)


def graph_replay_step(step_fn, static_args: tuple, static_kwargs: dict | None = None):
    """Capture step_fn into a CUDA graph and return a replay function.

    Note: Inputs must be persistent/static tensors on CUDA.
    """
    static_kwargs = static_kwargs or {}
    try:
        import torch.cuda as cuda
        g = cuda.CUDAGraph()
        stream = cuda.Stream()
        with cuda.stream(stream):
            cuda.synchronize()
            g.capture_begin()
            out = step_fn(*static_args, **static_kwargs)
            g.capture_end()
        def replay():
            g.replay()
            return out
        return replay
    except Exception:
        def replay_fallback():
            return step_fn(*static_args, **(static_kwargs or {}))
        return replay_fallback


def stop_grad(x: torch.Tensor) -> torch.Tensor:
    """Return a detached view (no grad)."""
    return x.detach()


def custom_grad(fn, grad_fn):
    """Wrap a stateless function with a custom gradient.

    grad_fn(x, out, grad_out) -> grad_x
    """
    import torch

    class _Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            y = fn(x)
            ctx.save_for_backward(x.detach(), y.detach())
            return y

        @staticmethod
        def backward(ctx, grad_out):
            x, y = ctx.saved_tensors
            gx = grad_fn(x, y, grad_out)
            return gx

    return _Fn.apply


