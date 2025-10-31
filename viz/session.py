import time, csv, os, json, shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Optional

import torch


class VizSession:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.dir = Path(cfg.log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "images").mkdir(exist_ok=True)
        (self.dir / "figures").mkdir(exist_ok=True)
        (self.dir / "artifacts").mkdir(exist_ok=True)
        (self.dir / "hists").mkdir(exist_ok=True)
        self._current_step: int = 0

    # ------------------------
    # Step management
    # ------------------------
    @property
    def current_step(self) -> int:
        return self._current_step

    def set_step(self, step: int) -> None:
        self._current_step = int(step)

    def step(self) -> None:
        self._current_step += 1

    # ------------------------
    # Logging primitives
    # ------------------------
    def log_scalar(self, step: Optional[int], key: str, value: float) -> None:
        s = self._current_step if step is None else int(step)
        with open(self.dir / "scalars.csv", "a", newline="") as f:
            csv.writer(f).writerow([s, key, float(value)])

    def log_text(self, step: Optional[int], key: str, text: str) -> None:
        s = self._current_step if step is None else int(step)
        rec = {"step": s, "key": key, "text": text}
        with open(self.dir / "text.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    def log_histogram(self, step: Optional[int], key: str, tensor: torch.Tensor, bins: int = 64, range: Optional[tuple[float, float]] = None) -> None:  # type: ignore[name-defined]
        import numpy as np
        s = self._current_step if step is None else int(step)
        t = tensor.detach().float().flatten().cpu().numpy()
        counts, bin_edges = np.histogram(t, bins=bins, range=range)
        rec = {"step": s, "key": key, "counts": counts.tolist(), "bin_edges": bin_edges.tolist()}
        with open(self.dir / "histograms.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    def log_image(self, step: Optional[int], key: str, image: Any, *, normalize: bool = True) -> Path:
        s = self._current_step if step is None else int(step)
        try:
            from PIL import Image
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Pillow (PIL) is required to log images") from e

        import numpy as np

        def _to_numpy(x: Any) -> "np.ndarray":  # type: ignore[name-defined]
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu()
                if x.ndim == 3 and x.shape[0] in (1, 3):
                    x = x.permute(1, 2, 0)
                x = x.numpy()
            x = x.astype("float32") if x.dtype.kind in ("f",) else x
            if normalize and x.dtype.kind == "f":
                mn, mx = float(x.min()), float(x.max())
                if mx > mn:
                    x = (x - mn) / (mx - mn)
                x = (x * 255.0).clip(0, 255).astype("uint8")
            if x.ndim == 2:
                x = np.stack([x, x, x], axis=-1)
            return x

        arr = _to_numpy(image)
        out = self.dir / "images" / f"{key}_{s}.png"
        Image.fromarray(arr).save(out)
        return out

    def log_figure(self, step: Optional[int], key: str, fig: Any) -> Path:
        s = self._current_step if step is None else int(step)
        out_html = self.dir / "figures" / f"{key}_{s}.html"
        out_png = self.dir / "figures" / f"{key}_{s}.png"
        try:
            import plotly.graph_objects as go  # type: ignore
            import plotly.io as pio  # type: ignore
            if hasattr(fig, "to_dict") or isinstance(fig, go.Figure):
                html = pio.to_html(fig, include_plotlyjs="cdn", full_html=True)
                out_html.write_text(html)
                return out_html
        except Exception:
            pass

        try:
            import matplotlib.pyplot as plt  # type: ignore
            if hasattr(fig, "savefig"):
                fig.savefig(out_png, bbox_inches="tight")
                return out_png
            elif fig is None:
                plt.savefig(out_png, bbox_inches="tight")
                return out_png
        except Exception:
            pass
        raise RuntimeError("Unsupported figure type for log_figure")

    def log_artifact(self, path: os.PathLike[str] | str, *, name: Optional[str] = None) -> Path:
        src = Path(path)
        dst_name = name or src.name
        dst = self.dir / "artifacts" / dst_name
        shutil.copy2(src, dst)
        return dst

    # ------------------------
    # Model integration helpers
    # ------------------------
    def attach_activation_stats(self, model):
        handles = []

        def _hook(name):
            def fn(_m, _inp, out):
                if isinstance(out, torch.Tensor):
                    s = self._current_step
                    self.log_scalar(s, f"{name}.mean", float(out.mean().detach().cpu()))
                    self.log_scalar(s, f"{name}.std", float(out.std().detach().cpu()))
            return fn

        for name, mod in model.named_modules():
            if isinstance(mod, (torch.nn.Linear, torch.nn.LayerNorm)):
                handles.append(mod.register_forward_hook(_hook(name)))
        return handles

    @contextmanager
    def profile(self, enabled: bool = True):
        if not enabled:
            yield
            return
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]) as prof:
            yield
        prof.export_chrome_trace(str(self.dir / f"profile_{int(time.time())}.json"))
