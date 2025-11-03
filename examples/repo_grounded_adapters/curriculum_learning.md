repo-as-curriculum scaffold that turns any codebase into a compact set of adapters that (1) inject at prompt time and (2) modulate the model at run time so answers are verifiably grounded in that repository. I’ll keep a tight focus on question/answer over a codebase, while making the lanes extensible to broader software tasks later.

1. Objective


Input: arbitrary repository RRR.


Output: a RepoQ&A Adapter Pack ARA_RAR​ containing structured “knowledge factors” + light-weight parameters that bias the model toward retrieving, citing, and reasoning strictly within RRR.


Runtime: user prompt + ARA_RAR​ ⇒ grounded answer with explicit citations (file:line), minimal hallucination, and optional verification hooks (import/compile/test).



2. Knowledge factors (what to extract from any repo)
We convert RRR into a factorized knowledge plane plus a curriculum of didactic pairs (question → answer+citation). These are the only things required to ground Q/A.
2.1 Symbolic/structural factors


Symbols: functions, classes, methods, constants, public API.


Features: name, path, signature, return/param types, decorators.




Graph edges: imports, calls, attribute accesses, inheritance, “defines/uses” links.


Ownership & modules: package → file → symbol hierarchy, visibility (public/private).


Contracts: docstrings, type hints, pre/post conditions, raises, complexity notes.


Build/run context: entrypoints, CLI, service configs, env vars.


2.2 Behavioral/empirical factors


Usage exemplars: call sites, tests, tutorial snippets, notebooks.


Assertions/invariants: from tests and defensive checks.


I/O schemas: dataclasses, Pydantic, protobufs, JSON samples.


Failure modes: exceptions, test failures, log messages.


2.3 Temporal/quality factors (optional but powerful)


Change history: commit messages touching symbol sss.


Test coverage map: lines/branches exercised per symbol.


Dependency integrity: version pins, known incompatibilities.


2.4 Vectorization (for retrieval & adapters)
For each artifact above, compute dense embeddings and sparse keys:


Dense vectors e(k)∈Rde^{(k)} \in \mathbb{R}^de(k)∈Rd:


edefe_\text{def}edef​: definition text (name + signature + docstring).


eusagee_\text{usage}eusage​: aggregated contexts of typical calls.


eteste_\text{test}etest​: assertions & counter-examples.


egraphe_\text{graph}egraph​: graph neighborhood (random-walk or GNN pooled).




Sparse keys:


trie over tokens/symbol names; BM25 over comments/docs; regex index for signatures.




We store dense embeddings factorized: E≈UV⊤E \approx U V^\topE≈UV⊤ (low-rank r≪dr \ll dr≪d) to form the memory plane that adapters can read. (This keeps packs small and swappable.)

3. Repo-as-Curriculum (how we teach the model the repo)
We synthesize a curriculum from easy to hard, targeted for Q/A:
flowchart TD
  A[Harvest] --> B[Canonicalize]
  B --> C[Curriculum Synthesis]
  C --> D[Adapter Packing]
  D --> E[Runtime Grounding]

  subgraph Harvest
    A1[Parse AST/Type Hints]
    A2[Build Code Graph]
    A3[Scan Tests/Notebooks]
    A4[Collect Config/CLI]
  end

  subgraph Canonicalize
    B1[Normalize Paths/IDs]
    B2[Deduplicate Symbols]
    B3[Unitize Chunks (≤2KB)]
  end

  subgraph Curriculum Synthesis
    C1[Level 1: Locate & Navigate]
    C2[Level 2: API Semantics]
    C3[Level 3: Usage & Edge Cases]
    C4[Level 4: Cross-Module Reasoning]
    C5[Level 5: Debug & Refactor]
  end

  subgraph Adapter Packing
    D1[Dense Factors U,V]
    D2[Sparse Keys & Trie]
    D3[Didactic Pairs (Q→A,cites)]
    D4[Routing Tables]
  end

  subgraph Runtime
    E1[Prompt Injection (hints,cites)]
    E2[Retriever over A_R]
    E3[Adapter Read (fast plane)]
    E4[Verifier (imports/tests)]
  end

3.1 Curriculum levels (for Q/A)


L1 — Locate & Navigate
“Where is foo() defined?” “List public functions in bar.module.”
Targets: pathing, module boundaries, symbol names.


L2 — API Semantics
“What does Dataset.split return?” “Which exceptions can connect() raise?”
Targets: signatures, docstrings, contracts.


L3 — Usage & Edge Cases
“How to paginate results?” “What happens if None is passed?”
Targets: tests, examples, guarded branches.


L4 — Cross-Module Reasoning
“How does ServiceA authenticate through ClientB?”
Targets: call chains, data schemas.


L5 — Debug & Refactor
“Why does test test_retry_backoff fail?”
Targets: failure traces, recent commits.


Each synthetic item is a didactic pair:
{
  "qid": "...",
  "level": 3,
  "question": "How to paginate results?",
  "gold_answer": "Use Client.list(page_size=..., cursor=...) ...",
  "citations": [{"file": "client.py", "span": [120, 188]}, ...],
  "support": ["tests/test_client.py:54-88", "docs/pagination.md:1-50"]
}

We can generate these pairs automatically from symbols/tests/docs + small templates; optional human QA improves fidelity later, but not required to ship.

4. Adapter pack format (.repoa)
A single compressed artifact you can mount at inference:
RepoQnAAdapterPack
 ├─ meta.json                 # repo hash, version, language, d, r
 ├─ factors/
 │   ├─ U.npy                 # [d, r]
 │   └─ V.npy                 # [d, r]
 ├─ index/
 │   ├─ dense.faiss           # fallback dense index (optional)
 │   ├─ sparse.bm25           # sparse term index
 │   └─ trie.sym              # symbol name trie
 ├─ graph/
 │   ├─ nodes.parquet         # id, kind, path, sig, doc digest, etc.
 │   └─ edges.parquet         # (src, rel, dst)
 ├─ curriculum/
 │   ├─ l1.jsonl ... l5.jsonl # didactic pairs
 │   └─ routes.json           # symbol→module block mapping
 └─ schemas/
     └─ io.json               # known I/O types & examples

Why factors? The U,VU,VU,V plane offers a constant-time associative read and tiny Hebbian-style “session writes” without touching the base model. The sparse index ensures exact symbol hits.

5. Injection path (prompt & runtime)
sequenceDiagram
  participant User
  participant Router
  participant Retriever
  participant Adapter
  participant LLM
  User->>Router: Q (natural language)
  Router->>Retriever: expand(Q)→queries; top-k chunks
  Retriever-->>Adapter: {chunks, cites, symbol hits}
  Adapter->>LLM: prompt injection (brief hints + citations)
  Adapter->>LLM: fast-plane read h_t → h'_t
  LLM-->>Adapter: draft answer
  Adapter->>LLM: verify & nudge (missing cites? compile fail?)
  LLM-->>User: final grounded answer (+ file:line)

5.1 Prompt-time injection (minimalistic, high yield)


Context header (≤ 10 lines): repo name, version, root module, strict grounding instruction: “Only cite symbols from this repo. If unknown, answer: ‘Not present in repository.’”


Top-k snippets (chunked, deduped, with path/line numbers).


Citations schema: instruct model to emit [[file.py:123–160]] inline.


5.2 Run-time modulation


Adapter read from M=UV⊤M = U V^\topM=UV⊤: project hidden state into repo-semantic subspace; boosts logits near repo tokens/symbols.


Optional Hebbian write (session-local): bind user query ↔ selected snippet to stabilize multi-turn Q/A.



6. Training & learning (curriculum → adapters)
Even if you want unsupervised operation from day one, we can still train a tiny router/gater for better behavior, using the synthesized curriculum as supervision.
6.1 Signals


Answer Loss: cross-entropy on gold answer text (or span extraction/cite pointers).


Attribution Loss: encourage tokens inside cited spans; penalize uncited claims.


Grounding KL: push the model to prefer repo tokens when evidence exists.


Verifier Reward (optional RL): import/compile/test success raises score.


6.2 What gets trained


Small controller head (LoRA/adapter) that outputs gates α,β\alpha,\betaα,β and routing into the factor plane blocks.


Retriever tie-ins: a shallow reranker that prioritizes chunks covering cited spans in curriculum items.


The base LLM stays frozen; packs swap instantly per repo.

7. Minimal, production-oriented scaffolding (code)
7.1 Pack builder (offline)
# pack_builder.py
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import json

def build_symbol_table(repo_root: Path) -> Dict[str, Any]:
    # Parse AST, collect defs, signatures, docstrings, paths, tests, etc.
    ...

def embed_artifacts(symbols: Dict[str, Any], d: int = 1024) -> np.ndarray:
    # Produce E in R^{N x d}: definition/usage/test+graph pooled vectors.
    ...

def factorize(E: np.ndarray, r: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    # Low-rank: E ≈ U V^T  with U,V in R^{d x r}
    # Use randomized SVD or NNLS factor for non-neg constraints
    ...

def build_sparse_indexes(symbols: Dict[str, Any]):
    # BM25 over docs/comments; trie over symbol names
    ...

def synthesize_curriculum(symbols: Dict[str, Any]) -> Dict[int, List[Dict]]:
    # Generate L1..L5 Q→A with exact citations using templates and graph
    ...

def write_pack(out_dir: Path, U: np.ndarray, V: np.ndarray, symbols, sparse, curriculum, meta):
    (out_dir / "factors").mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "factors" / "U.npy", U)
    np.save(out_dir / "factors" / "V.npy", V)
    # write symbols, sparse, curriculum, meta.json
    ...

7.2 Runtime adapter (inference)
# runtime_adapter.py
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple
import numpy as np

class RepoQAAdapter(nn.Module):
    def __init__(self, d_model: int, rank: int = 64):
        super().__init__()
        self.d = d_model; self.r = rank
        self.register_buffer("U", torch.zeros(d_model, rank))
        self.register_buffer("V", torch.zeros(d_model, rank))
        self.controller = nn.Sequential(
            nn.Linear(d_model, d_model//2), nn.SiLU(),
            nn.Linear(d_model//2, 2),  # alpha, beta
        )

    @torch.no_grad()
    def load_pack(self, pack_dir: str):
        U = torch.from_numpy(np.load(f"{pack_dir}/factors/U.npy"))
        V = torch.from_numpy(np.load(f"{pack_dir}/factors/V.npy"))
        self.U.copy_(U); self.V.copy_(V)
        # load sparse/dense indexes into a Retriever helper (not shown)

    def forward(self, h: torch.Tensor, kv_summary: torch.Tensor) -> torch.Tensor:
        # associative read: y = (V U^T) h
        Ut_h = torch.einsum("dr,btd->btr", self.U, h)
        y = torch.einsum("dr,btr->btd", self.V, Ut_h)
        alpha, beta = self.controller(h).sigmoid().unbind(-1)
        return h + alpha[..., None] * y + beta[..., None] * kv_summary

def inject_prompt(prompt: str, topk: List[Dict[str, Any]]) -> str:
    header = (
      "SYSTEM: You are answering strictly from the repository below.\n"
      "If the answer is not present, reply: 'Not present in repository.'\n"
      "Cite as [[path.py:lineStart-lineEnd]].\n\n"
      "CONTEXT:\n"
    )
    ctx = "\n".join(
      f"[[{c['path']}:{c['span'][0]}-{c['span'][1]}]]\n{c['text']}"
      for c in topk
    )
    return f"{header}{ctx}\n\nUSER: {prompt}\nASSISTANT:"

7.3 Inference loop (glue)
# inference.py
def answer(repo_pack, llm, adapter: RepoQAAdapter, prompt: str) -> str:
    topk = retriever(repo_pack, prompt)            # sparse+dense hybrid
    kv_summary = pool(topk)                        # [B,T,d] pooled vectors
    aug_prompt = inject_prompt(prompt, topk)
    # run LLM with adapter hook at chosen layers:
    with adapter.loaded(repo_pack):
        return llm.generate(aug_prompt, adapter_kv=kv_summary)

Implementation note: wire adapter.forward into your model’s forward at every 2–4 blocks for low overhead.

8. Evaluation (tight and pragmatic)


Grounded EM: exact-match answer and at least one correct citation span.


Attribution P/R: overlap between emitted citations and oracle spans.


Repo Containment: % answers whose cited paths exist in repo.


Verifier Pass Rate: if the answer includes code, does it import & run unit tests?


Latency budget: retrieval + generation ≤ X ms/token at target size.



9. How this narrows cleanly to Q/A
We deliberately exclude synthesis tasks (writing new modules, refactors, etc.) from the initial scope. The curriculum and factors above are precisely the minimal set for question answering grounded in code:


Symbols + docs → what it is


Graph + usage/tests → how it behaves


Factors U,VU,VU,V + sparse keys → how to find it instantly


Didactic pairs → how to teach the router/gater to stay inside the repo


Prompt injection + adapter read → how to convert that into verifiable answers



10. Quick “first ship” checklist


 Implement pack_builder: AST→symbols, graph, embeddings, factorization U,VU,VU,V.


 Synthesize L1–L3 curriculum (L4–L5 later).


 Build hybrid retriever (sparse+factor-guided rerank).


 Wire RepoQAAdapter at every 3rd block; test with frozen LLM.


 Add citation-style and Not-present guard in prompt template.


 Track Grounded EM + Repo Containment as your go/no-go metrics.



TL;DR
Treat the repository as a curriculum + factor plane: extract just the vectors that let your model find, cite, and reason about symbols; pack them into a swappable .repoa. At inference, inject a tiny prompt scaffold and read from the factor plane to nudge hidden states toward repo tokens—yielding concise, correctly cited answers. This is the most direct path to grounded Q/A over codebases, and the scaffold above is production-ready to implement now.


11. Scaffold implemented in this repo (grounded Q/A narrow focus)

- Harvest/Graph: `examples/repo_grounded_adapters/code_graph.py` builds symbols, imports, calls, pytest nodes; used for selection and subgraphing.
- Adapter generation (static prior): `repo_conditioned_adapter.py` creates base adapters from a repo embedding and supports subgraph embeddings; broadened target coverage (q/k/v/o + mlp up/down/gate).
- Modular runtime: `run_repo_adapter.py` injects context (heads/windows), mixes base + subgraph adapters, enforces citations; accepts `--ignore`.
- Self-tuning loop: `self_tune.py` synthesizes per-module prompts, generates answers, verifies (citations + tests), distills to JSONL, exports tuned prior and optional per-module subgraph adapters; supports chunk budget (`--context-tokens`), timeout, resume, and ignore propagation.
- On-the-fly adapters: `run_llama_with_repo_adapter_on_the_fly.py` builds subgraph adapters at prompt time and mixes them.

Gaps (next iterations):
- Curriculum synthesis (L1–L3) with oracle spans; attribution-aware training (LoRA) on verified pairs; hybrid retriever (BM25+trie) and reranker; metrics (grounded EM, containment, attribution P/R, verifier pass-rate); optional factor plane (U,V) read path.