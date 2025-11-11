Program-Conditioned Adapters: Conditioning Pretrained Language Models on Executable Systems
Abstract

We present Program-Conditioned Adapters (PCA), a method for dynamically specializing a pretrained language model to the structure and behavior of an arbitrary program—defined broadly to include APIs, CLIs, data pipelines, SQL schemas, configuration stacks, notebooks, microservices, and GUI scripts. PCA constructs a polymorphic ProgramGraph of symbols, contracts, and operational traces; derives a multi-channel embedding that encodes both static specification and empirical behavior; and synthesizes a lightweight parameter delta that modulates the base model at inference time. The result is a system that produces program conditioned output against concrete program artifacts, while preserving safety and reversibility. We formalize the objective, describe the architecture, analyze stability and capacity constraints, and propose a reproducible evaluation protocol spanning diverse program modalities.

1. Motivation

Pretrained language models possess broad priors but lack precise alignment to the evolving semantics of the programs they must operate over. Program-Conditioned Adapters address this gap by converting the state and shape of an executable system into a compact control signal that steers generation, retrieval, and reasoning toward the program’s realities: endpoints and schemas, CLI contracts, DAG dependencies, invariants, failure modes, and observed I/O. PCA thereby improves factuality, reduces hallucinations, and enables verifiable outputs that reference concrete program regions.

2. Problem Formulation

Let 
𝑀
M be a pretrained LM with parameters 
Θ
Θ. Let 
𝑃
P be a program with observable structure 
𝑆
(
𝑃
)
S(P) and behaviors 
𝐵
(
𝑃
)
B(P). Given a query 
𝑞
q, we seek a conditioning delta 
Δ
𝑃
(
𝑞
)
Δ
P
	​

(q) of small norm such that the modulated model

𝑀
Θ
⊕
Δ
𝑃
(
𝑞
)
(
𝑞
)
→
𝑎
M
Θ⊕Δ
P
	​

(q)
	​

(q)→a

produces an answer 
𝑎
a that is (i) grounded in 
𝑃
P, (ii) auditable via references into 
𝑆
(
𝑃
)
∪
𝐵
(
𝑃
)
S(P)∪B(P), and (iii) stable, with bounded deviation from 
𝑀
M’s safety and calibration profile.

3. System Overview
flowchart TD
  Q[Query q] --> S[Region Selector]
  PG[ProgramGraph Builder\n(symbols • contracts • topology • traces)] --> E
  S --> E[Multi-Channel Embedder\n z = ⨁(z_sym, z_contract, z_topo, z_text, z_trace)]
  E --> A[Adapter Synthesizer\nΔ = f(z; shape, rank, gates)]
  A --> M[Runtime Mixer\n(targeted projections, norm caps)]
  M --> G[LM Generation]
  G --> V[Verifier/Probes\n(schema checks, CLI runs, DAG dry-runs)]
  V --> L[(Logs: citations • probes • coverage)]


Key principles

Polymorphism: The ProgramGraph abstracts over program modalities.

Channelization: Separate semantic channels encode symbols, contracts, topology, text, and traces.

Delta minimality: Conditioning uses small, targeted parameter deltas with explicit safety caps.

Auditable grounding: Every output carries references to identified regions of the ProgramGraph and optional probe results.

4. The ProgramGraph Abstraction
4.1 Node & Edge Taxonomy

Entities (nodes): endpoints, CLI commands and flags, SQL tables/columns/indexes, DAG tasks, message types, config keys, notebook cells, UI actions.

Relations (edges): invocation (A→B), data-flow (produces/consumes), control-flow, foreign-key/constraint links, service calls, dependency arcs, inclusion/imports.

Artifacts (attachments): specs (OpenAPI/Protobuf/Avro), --help outputs, EXPLAIN plans, logs, tests, coverage, example payloads.

classDiagram
  class Entity {
    id: string
    kind: enum
    name: string
    path: string|URI
    contracts: Map
  }
  class Edge {
    src: Entity
    dst: Entity
    relation: enum
    meta: Map
  }
  class ProgramGraph {
    +entities(): Iterable~Entity~
    +edges(): Iterable~Edge~
    +search_refs(token): Iterable~(Entity,Span)~
    +subgraph(seeds, radius): ProgramGraph
    +artifacts(kind): Iterable~Blob~
  }
  ProgramGraph --> Entity
  ProgramGraph --> Edge

4.2 Region Selection

Given 
𝑞
q, a selector maps tokens to candidate entities using hybrid lexical/semantic matching and graph walk expansions. The result is a program region 
𝑅
⊆
𝑆
(
𝑃
)
R⊆S(P) (typically a small subgraph) used for conditioning and for post-hoc references.

5. Multi-Channel Embedding

We construct a representation

𝑧
=
N
o
r
m
 ⁣
(
𝑤
sym
𝑧
sym
⊕
𝑤
contract
𝑧
contract
⊕
𝑤
topo
𝑧
topo
⊕
𝑤
text
𝑧
text
⊕
𝑤
trace
𝑧
trace
)
,
z=Norm(w
sym
	​

z
sym
	​

⊕w
contract
	​

z
contract
	​

⊕w
topo
	​

z
topo
	​

⊕w
text
	​

z
text
	​

⊕w
trace
	​

z
trace
	​

),

where each 
𝑧
⋅
z
⋅
	​

 is derived from:

Symbols: typed tokens for entity kinds and names.

Contracts: signatures, schemas, constraints, types, pre/post conditions.

Topology: graph motifs (fans-in/out, cuts, path features), execution ordering in DAGs, service call hierarchies.

Text: specs, READMEs, runbooks, notebooks, commit notes.

Traces: runtime exemplars (request/response pairs, CLI sessions), coverage hits, failure signatures.

The pipeline is budgeted (caps per channel), hashed (for large vocab), and normalized to stabilize downstream adapter synthesis.

6. Adapter Synthesis and Runtime Mixing
6.1 Objective

Produce a projection-targeted delta 
Δ
Δ guided by 
𝑧
z, constrained by layer shapes and a gating schedule. Concretely:

Target subsets of projections (e.g., attention 
𝑞
/
𝑘
/
𝑣
/
𝑜
q/k/v/o and MLP up/down/gates) with per-projection weights.

Enforce norm caps per layer to preserve stability and safety.

Support reversible application to allow rapid switching across program regions or distinct programs.

6.2 Inference Path

Shape discovery: infer hidden sizes and projection partitions from the host model.

Delta generation: map 
𝑧
z to low-rank updates with per-channel gates.

Mixing: add the delta at runtime with hot-swap hooks; log active targets, norms, and gates.

Answering: generate text and plans with inline region references (entity IDs, artifact spans).

sequenceDiagram
  participant U as User
  participant S as Selector
  participant E as Embedder
  participant Y as Synthesizer
  participant X as Mixer
  participant L as LM
  U->>S: q
  S->>E: Region R
  E->>Y: z (channels)
  Y->>X: Δ (targets, gates)
  X->>L: Θ ⊕ Δ
  L-->>U: grounded answer + refs

7. Grounding & Verification

PCA encourages defensible answers:

Citations: entity-span references into the ProgramGraph artifacts used for conditioning.

Probes (optional but recommended):

API/Schema: validate against OpenAPI/JSON-Schema/Protobuf; replay golden requests.

CLI: dry-run commands with --help or sandbox inputs; parse exit codes.

DAG: simulate node execution or dependency resolution; check produced artifact schemas.

SQL: lint, EXPLAIN, or run against a sandbox; assert constraints.

Probe signals are logged to a telemetry stream for audit and iterative improvement.

8. Safety & Stability

Delta Norm Control: per-layer 
ℓ
2
ℓ
2
	​

 bounds relative to base weights maintain calibration.

Projection Weighting: attenuate K/V updates to reduce attention drift; emphasize output and MLP projections for lexical adaptation.

Reversibility: hot-unplug hooks guarantee restoration of the base model.

Capacity Budgeting: per-channel caps in the embedder and rank limits in the synthesizer prevent over-conditioning and maintain latency targets.

9. Evaluation Protocol
9.1 Modalities

We recommend a cross-modal benchmark suite:

API: endpoint QA, request synthesis, error remediation.

CLI: command construction, flag interpretation, pipeline recipes.

Data: SQL authoring, schema repair, migration planning.

Pipelines: DAG debugging, dependency explanations, artifact lineage.

Configs/Orchestration: Terraform/K8s/YAML reasoning, invariant checks.

9.2 Tasks & Metrics

Exactness: pass@k for probe-validated answers (e.g., API responses match schema).

Citation Faithfulness: fraction of spans that point to entities actually used in conditioning.

Edit Distance to Executable Plan: for CLI/DAG tasks, distance from output to a working command/graph.

Runtime Stability: perplexity drift and refusal rate deltas under safety policies.

Latency/Throughput: overhead of selection, embedding, synthesis, and mixing.

9.3 Ablations

Channel removal (−traces, −topology, −contracts).

Projection targeting (attention-only vs MLP-only vs mixed).

Rank and norm caps.

Selector radius and region size.

10. Related Directions (High-Level)

PCA connects to: (i) test-time adaptation with structured signals, (ii) tool-augmented LMs via program-aware plans, (iii) retrieval-conditioned modulation that integrates structured graphs and behavior traces, and (iv) parameter-efficient specialization that preserves the base model while enabling rapid domain shifts.

11. Limitations

Graph Fidelity: incomplete or stale artifacts reduce conditioning quality.

Probe Coverage: not all programs permit safe or deterministic dry-runs.

Distribution Shift: heavy reliance on traces may overfit transient behaviors.

Attribution Granularity: extremely fine-grained references (e.g., dynamic configs) can be costly to track.

12. Future Work

Unified Planning Interface: generate executable plans that couple PCA outputs with verified tool calls.

Continual Self-Tuning: harvest probe outcomes to refine selector heuristics and channel weights over time.

Multi-Program Composition: safe mixture of deltas for interacting systems (e.g., API + DB + batch DAG).

Counterfactual Probing: compare answers under masked entities to quantify causal grounding.

13. Conclusion

Program-Conditioned Adapters turn the living state of a program into a precise, reversible control signal for pretrained language models. By unifying heterogeneous artifacts and behaviors in a ProgramGraph, channelizing them into a stable embedding, and synthesizing targeted, norm-bounded deltas at inference time, PCA yields grounded, auditable, and practically useful answers across modalities—from APIs and CLIs to data pipelines and orchestration stacks. This framework extends the effective reach of pretrained models to “everything executable,” without sacrificing safety or maintainability.

Appendix A: Minimal Interfaces
erDiagram
  PROGRAMGRAPH ||--o{ ENTITY : contains
  PROGRAMGRAPH ||--o{ EDGE : contains
  ENTITY ||--o{ ARTIFACT : has
  ENTITY {
    string id
    string kind
    string name
    string path
  }
  EDGE {
    string src
    string dst
    string relation
  }
  ARTIFACT {
    string type
    string uri
    string span
  }

flowchart LR
  subgraph Backends
    A[OpenAPI/GRPC]:::b
    B[CLI/Manpages]:::b
    C[SQL/DB Introspect]:::b
    D[DAG/Orchestrators]:::b
    E[Configs/Infra]:::b
    F[Runtime Logs/Traces]:::b
  end
  A-->PG[(ProgramGraph)]
  B-->PG
  C-->PG
  D-->PG
  E-->PG
  F-->PG
  PG-->Embedder
  classDef b fill:#eef,stroke:#88a,stroke-width:1px

Appendix B: Reproducible PCA Runbook (Concise)

Build ProgramGraph for target system.

Select Region with hybrid lexical/semantic expansion.

Embed Channels (symbols, contracts, topology, text, traces) with budgets and normalization.

Synthesize Delta with projection targeting, rank 
𝑟
r, and norm caps.

Mix at Runtime; generate answer with citations.

Verify with Probes; log outcomes and adjust channel weights and selector radius.