## capacity.py — Entropy and Capacity Scheduling

Scores repo/selection complexity and schedules capacity knobs (`rank`, `gsub`).

### Key APIs
- `entropy_score(g, mods, files_rel, weights="repo=0.4,subgraph=0.4,question=0.2")` -> `(score∈[0,1], diag)`
- `scale_capacity(es, rank_min, rank_max, gsub_min, gsub_max)` -> `(rank, gsub)`

### Usage
Use with runner `--entropy-aware` to adapt capacity per question.

### Notes
- Repo component blends module count (log) and import density.
- Subgraph component uses degree/breadth of the selection.
- Question component increases with files matched by the prompt.


