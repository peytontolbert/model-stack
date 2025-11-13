Program-Conditioned Adapters White Paper
Executive Summary
Program-Conditioned Adapters (PCA) dynamically specialize pretrained language models to the concrete structure and behavior of arbitrary executable systems—spanning APIs, CLIs, data pipelines, schemas, configuration stacks, notebooks, and microservices—by constructing a ProgramGraph, deriving multi-channel embeddings, and synthesizing lightweight parameter deltas that preserve safety and reversibility while grounding outputs in real artifacts. The approach provides reversible conditioning, grounded decoding, and a backend-agnostic core that operates across repositories, APIs, SQL, DAGs, and other modalities without retraining the base model.

Motivation and Problem Statement
General-purpose LMs possess broad priors but lack precise alignment with the evolving semantics of the programs they serve. PCA addresses this mismatch by converting live program state into a compact control signal that steers generation toward verified endpoints, schemas, contracts, dependencies, and failure modes, resulting in more factual, auditable answers with bounded deviation from the base model’s safety profile.

System Architecture
The PCA workflow comprises a region selector, ProgramGraph builder, multi-channel embedder, adapter synthesizer, runtime mixer, and verifier/probe suite, forming a closed loop that logs citations, probes, and coverage for accountability. Operationally, deployments follow a six-stage pipeline: build the ProgramGraph, embed channelized vectors, generate low-rank adapters, mix them on demand, answer with enforced citations, and optionally self-tune through verifiable prompts and telemetry capture.

ProgramGraph Abstraction
The ProgramGraph unifies entities (endpoints, commands, tables, DAG tasks, config keys, notebook cells, UI actions), typed relations (invocation, data-flow, control-flow, constraints, service calls, dependencies), and attached artifacts (specs, help output, plans, logs, tests, payloads) behind a minimal interface supporting enumeration, search, subgraph extraction, and artifact retrieval for any program modality. Each program instance is formalized as 
P
=
⟨
U
,
E
,
A
,
C
,
S
,
O
,
T
,
Θ
⟩
P=⟨U,E,A,C,S,O,T,Θ⟩, with regions 
R
⊆
(
U
∪
E
)
R⊆(U∪E) serving as the conditioning and citation units exposed via ProgramGraph operations and associated retrieval policies.

Multi-Channel Embedding
PCA constructs a normalized representation 
z
=
Norm
(
w
sym
z
sym
⊕
w
contract
z
contract
⊕
w
topo
z
topo
⊕
w
text
z
text
⊕
w
trace
z
trace
)
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
 ) that blends symbols, contracts, topology, text, and traces under budgeted, hashed pipelines for stability and scalability. Channel composition and normalization mirror the embedding stage of the end-to-end pipeline, where graph propagation, text inclusion, and weighting controls tailor the feature mix to program characteristics.

Adapter Synthesis and Runtime Mixing
Given 
z
z, the synthesizer generates projection-targeted, low-rank deltas over attention (q/k/v/o) and MLP projections with per-target weights, norm caps, and reversible application hooks so that distinct program regions can be swapped rapidly at inference. This capability aligns with the “Adapt” and “Mix + Apply” phases that infer model shapes, expose preset weightings, support question-driven zooming, and maintain modular runners for context packing and policy enforcement.

Grounding and Safety Controls
Grounding is enforced via citations linking answers to ProgramGraph spans and optional probes that validate APIs, CLIs, DAGs, and SQL flows, with telemetry logging for audit and iteration. Safety measures include per-layer 
ℓ
2
ℓ 
2
​
  norm caps, projection weighting to minimize attention drift, reversible hot-unplug hooks, and capacity budgeting—complemented by operational controls for rank, alpha, layer schedules, and deterministic execution guards in production runners.

Evaluation Strategy
A recommended benchmark suite spans APIs, CLIs, data systems, pipelines, and infrastructure configuration, with metrics such as probe-validated exactness, citation faithfulness, executable-plan edit distance, stability under safety policies, and pipeline latency/throughput, accompanied by ablations on channels, projection targeting, rank/norm caps, and region size.

Implementation Runbook
Implementers can rely on build-time caches (adapters, embeddings, symbol and window indices, factual summaries, rerank features, source manifests) to accelerate retrieval and anchoring. The CLI-driven runbook covers base adapter creation, on-the-fly answering with optional enhancements, and verification-led self-tuning—mirroring the concise operational checklist in the appendix (build ProgramGraph, select region, embed channels, synthesize delta with rank caps, mix at runtime, verify with probes).

Research Opportunities
Documented limitations include incomplete graphs, probe coverage gaps, trace overfitting, and the cost of fine-grained attribution, while future work targets unified planning interfaces, continual self-tuning, safe multi-program composition, and counterfactual probing to quantify causal grounding. Additional options inspired by knowledge-based neural networks—such as domain priors, localized rank allocation, and interpretable rounding—offer further avenues for controllable adaptation.

Conclusion
PCA transforms executable program state into precise, reversible control signals for pretrained language models, unifying heterogeneous artifacts via ProgramGraphs, channelized embeddings, and norm-bounded deltas to deliver grounded, auditable answers across diverse modalities without sacrificing safety or maintainability.

Testing
⚠️ Not run (not requested).