## targets.py — Target Shapes Utility

Parses explicit target shape strings passed via CLI into `(d_out, d_in)` tuples.

### Key API
- `parse_target_shapes("q_proj=4096:4096,k_proj=4096:4096,up_proj=11008:4096")` -> `Dict[str, Tuple[int,int]]`

### Usage
Use when bypassing auto shape detection or for non‑standard targets.


