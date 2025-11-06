## runtime.py — Programmatic Runner Helpers

Thin wrappers to compose per‑layer deltas and invoke the orchestrator in‑process.

### Key APIs
- `build_per_layer_deltas(adapters, target_names, g_sub=1.0, base_adapters=None)` -> `List[Dict[name->Tensor]]`
- `run_repo_adapter(model, adapters_npz, repo, prompt, cache_dir=None, device="cpu", gpu_ids=None, context_tokens=5000, ignore=None, timeout_sec=None)` -> `(rc, stdout, stderr)`

### Usage
Use when embedding the runner in another process or service and you need a clean (rc, stdout, stderr) contract.


