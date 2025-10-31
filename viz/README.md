# viz

Minimal visualization logging and static dashboard.

## Usage

```python
from viz import VizSession

viz = VizSession(type("Cfg", (), {"log_dir": ".viz"}))
viz.set_step(0)
viz.log_scalar(None, "train.loss", 1.23)

# increment step then log
viz.step()
viz.log_scalar(None, "train.loss", 1.11)
```

Render dashboard:

```bash
python -m viz.cli render --log-dir .viz --title "Training"
```

This writes `.viz/index.html` with per-key scalar charts.

Optional helpers:
- log_histogram(step, key, tensor)
- log_image(step, key, image) (requires Pillow)
- log_figure(step, key, fig) (Plotly/Matplotlib)
- log_artifact(path, name=None)
- attach_activation_stats(model)
- profile(enabled=True)
