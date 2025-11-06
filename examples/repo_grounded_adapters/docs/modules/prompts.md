## prompts.py — Prompt Builders for Distillation

Creates simple, verifiable prompts per module using symbol tables.

### Key API
- `build_prompts_for_module(g, module, max_q=3)` -> `List[str]`
  - Includes: overview of module symbols and up to two symbol‑specific questions.

### Usage
Used by `tune.py` to assemble a small curriculum of Q/A tasks per module.


