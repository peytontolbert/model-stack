## selection.py — Candidate Discovery and Reranking

Provides program region selection over a `ProgramGraph` via a policy that mixes similarity and structural signals. Example backends can add file/module heuristics and rerankers.

### Key APIs
- `RetrievalPolicy.from_spec("sim:0.6,struct:0.4", temp=0.7)` -> policy
- `policy.score_entities(prompt, program_graph)` -> ranked entity IDs
- Example backends may expose helpers like `question_aware_modules_and_files(...)` for repo‑specific flows.

### Tips
- Prefer `--of-sources question` for broad prompts; use `--zoom-symbol` for targeted flows.
- When using a backend, ensure prompts include stable tokens to match entity names/signatures for better recall.


