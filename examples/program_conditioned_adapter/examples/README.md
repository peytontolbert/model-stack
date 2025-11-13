## Program Examples (Backends and Use Cases)

This directory hosts example programs (each provides a ProgramGraph backend, retrieval policy, and optional probes) that run on top of the Program‑Conditioned Adapter (PCA) core.

APIs (OpenAPI/gRPC)
- Program: api_grounded_qa
- Entities: endpoints, operations, schemas, examples
- Artifacts: OpenAPI files, protobufs, example requests, changelogs
- Probes: JSON‑Schema validation, contract diffs, replay of golden requests
- Tasks:
  - Change‑impact: diff spec versions → list breaking changes and affected clients
  - Grounded endpoint QA: synthesize request/response with schema validation and citations

CLIs / Manpages
- Program: cli_pipeline_planner
- Entities: commands, subcommands, flags, env vars
- Artifacts: manpages, --help outputs, example scripts
- Probes: dry‑run execution, exit codes, argument parsing checks
- Tasks:
  - Command construction from task descriptions
  - Pipeline recipes across multiple commands with validated dry‑runs
  - Migration helpers: map deprecated flags to new equivalents with citations

DAGs / Orchestrators
- Program: dag_debugger
- Entities: tasks, operators, datasets, schedules
- Artifacts: DAG definitions, runtime logs, task metadata
- Probes: dry‑run, dependency resolution, lineage checks
- Tasks:
  - DAG debugging: explain failed task via dependency/contract evidence
  - Execution planning: propose parallelization/caching opportunities

Microservices / Messaging
- Program: svc_messaging_ops
- Entities: services, topics/queues, message types, consumers/producers
- Artifacts: AsyncAPI/specs, schemas, topology maps, error logs
- Probes: schema compatibility checks, DLQ scans, retry config simulation
- Tasks:
  - Contract gaps: detect missing producers/consumers or schema drifts
  - Retry/dead‑letter planning: propose safe backoff/timeout configs
  - SLO guidance: map symptoms to service‑specific reliability levers

ML / Feature Stores
- Program: feature_store_qa
- Entities: features, sources, transforms, training jobs
- Artifacts: feature specs, training configs, lineage graphs
- Probes: schema checks, offline/online parity, drift tests
- Tasks:
  - Feature lineage: trace features to sources and transformations
  - Training pipeline QA: verify schema/contract adherence end‑to‑end
  - Drift diagnostics: propose probes and mitigations

Data Integration (ETL/ELT)
- Program: etl_integrator
- Entities: sources, transforms, joins, schedules, SLAs
- Artifacts: mapping specs, SQL scripts, lineage, SLA docs
- Probes: schema validation, EXPLAIN plans, SLA simulators
- Tasks:
  - Mapping assistants: propose joins/transforms with schema checks
  - SLA planning: detect bottlenecks and propose schedule changes
  - Backfill plans: safe, chunked backfills with constraints

Security / SOAR
- Program: soar_playbooks
- Entities: playbooks, actions, rules, integrations
- Artifacts: runbooks, connector specs, alert samples
- Probes: dry‑run actions, policy lint, rule simulation
- Tasks:
  - Playbook validation: check steps against tool contracts
  - Alert triage: map rules/signatures to remediation tasks
  - Policy rollout: generate incremental deployment plans with guardrails

Scientific / HPC Schedulers
- Program: hpc_scheduler_planner
- Entities: queues, constraints, jobs, nodes
- Artifacts: SLURM/LSF configs, job scripts, performance metrics
- Probes: dry‑run submit, resource fit checks, queue policy simulation
- Tasks:
  - Job planning: resource requests, constraints, queue selection
  - DAG‑to‑SLURM translation: grounded script generation with checks
  - Performance hints: propose parallelization and I/O layouts

Multi‑Program Unions (Software Development domain)
- Program: repo_multiK_grounded_qa
  - Compose: K repositories (client/server/infra)
  - Tasks: answer spanning repos with interface table proving symbol ↔ endpoint ↔ config links; multi‑repo citations (repo:file:line)
- Program: repo_interface_mapper
  - Compose: K repositories forming a system
  - Tasks: map cross‑repo contracts (types/endpoints/events) and output a ContractMap table; verify with a golden‑path toy integration check
- Program: agent_contract_guard
  - Compose: K repositories with interfaces and service contracts
  - Tasks: detect cross‑repo interface incompatibilities; emit ContractGuardReport and FixPlan; validate with a ToyIntegrationCheck
- Program: dev_union_repo_api_db_ci
  - Compose: Code repo (Git) + OpenAPI + DB schema + CI configs
  - Tasks: End‑to‑end “implement/modify endpoint X” plan (code stubs + SQL + CI) and “what breaks if we rename column Y?” with grounded citations
- Program: dev_union_repo_k8s_configs
  - Compose: Repo + Helm/K8s manifests + service‑mesh policy
  - Tasks: Safe rollout plan (resources/limits/probes) and config/code drift detection + fixes
- Program: dev_union_repo_tests_coverage
  - Compose: Repo + test results + coverage + flaky logs
  - Tasks: Minimal test plan for touched modules (evidence: coverage hits) and flaky test triage with likely root cause and stabilizing steps

LLM‑in‑the‑loop Primitives (PCA roles)
- Program: webdom_single_arxiv
  - IO: ArxivBundle{ids|urls|query} → PaperDOM{title, authors[], abstract, sections[], references[], citations[]}
  - Probes/Θ: each DOM section has anchored citations to fetched HTML/text; size/structure checks
  - Notes: uses Selenium headless Chromium to fetch pages from [arXiv](https://arxiv.org/). Install: `pip install selenium webdriver-manager`; ensure Chrome/Chromium installed.
- Program: llm_as_router
  - IO: Subgoal{text} + CapabilityIndex → RouteDecision{program_id, confidence, rationale}
  - Probes/Θ: route‑regret via hindsight after verification; compare chosen route vs best verified route
  - Tasks: select most promising program(s) for a goal under budget constraints with rationale and fallbacks
- Program: router_meta_capability_select
  - IO: Subgoal{intent, repo, codegraph_slice?, constraints, budget} → Binding{program_id, adapter_mix{(adapter_id, w, rank)…}}
  - Probes/Θ: interface match to program’s schema; budget feasibility; adapter policy gates (mix_top_k, ranks)
  - Tasks: choose program and adapter mixture per subgoal; learn from verified passes (recent_pass_boost)
- Program: llm_as_agent
  - IO: Tools/Capabilities + Goal → ActionPlan{steps[], rationales[]} + Trace
  - Probes/Θ: step outputs validated by downstream program verifiers; ensure contracts/preconditions satisfied
  - Tasks: decompose and invoke tool‑like capabilities with typed arguments while respecting budgets and policies
- Program: llm_as_planner (optional)
  - IO: Goal → TypedDAG{subgoals[], edges[]}
  - Probes/Θ: schema/type checks against Γ; replay viability under budget
  - Tasks: produce DAG of subgoals with capability matches and minimal budgets
- Program: llm_as_judge (optional)
  - IO: ProgramOutputs → Verdict{ok, summary, citations}
  - Probes/Θ: cross‑checks with lints/tests/compile; disagreements logged as counter‑evidence
  - Tasks: augment/verbalize verifier results with human‑readable rationale and policy mapping
- Program: llm_as_packer (optional)
  - IO: Sources + Constraints → PackedWindows{slices[]}
  - Probes/Θ: citation density, hit rate vs ground truth windows
  - Tasks: choose evidence windows (diffs, spans, API slices) to maximize verifier pass per token

Notes
- Each program provides a ProgramGraph implementation and (optionally) probes that the runner can use to verify, cite, or repair answers.
- Use `examples/program_conditioned_adapter/run.py --pg-backend <module:Factory>` to load a backend; the PCA core remains program‑agnostic.

Software Development Program Examples
- Warehouse DAG Diagnose (ETL/DBT/Airflow)
  - Name: warehouse_multi_dag_diagnose
  - Program: U = DAG runs/logs; O = SLA root-causes + fix plan; Θ = replay on sample DAG and green task exits.
  - Tasks: diagnose SLA misses and propose actionable fixes with citations.
- Change Summarizer (diff-aware, symbol-linked)
  - Name: summarize_single_repo_change
  - Program: U = diffs, graph slices, tests touched; O = ChangeSummary{headline, risk_factors[], api_changes[], test_impacts[], citations[]}; Θ = every bullet has resolvable citations; size/structure limits.
  - Tasks: produce PR-ready summaries with anchors; optional Narrative.md for human review.
- Code Entity Graph (single repo)
  - Name: graph_single_code_entity
  - Program: U = files/symbols; E = imports/calls; O = symbols, edges, tests, owners, api_surface; Θ = parser success rate, import acyclicity per package, symbol→file existence, test discovery sanity.
  - Tasks: materialize CodeGraph caches; serve CodeGraphSlice given symbols and radius.
- CI‑Failure Triage (multi‑repo + logs)
  - Name: ci_multi_triage
  - Program: U = commits, PRs, CI jobs, build/test logs; E = PR→job, commit→artifact; O = failing tests, error types; T = step logs + stack traces; Θ = failure classifier + fix‑suggestion verifier.
  - Tasks: grounded failure explanation; “nearest fix” retrieval; patch sketch with citations to failing lines and prior fixes.
  - Adapter context: build logs + failing test windows + changed files.
  - Eval: retro‑triage accuracy, patch acceptance rate.
- PR‑Diff Review with Policy
  - Name: repo_single_pr_review
  - Program: U = diffs, CODEOWNERS, linters; C = style/security/complexity policy; O = lints/coverage deltas; Θ = policy checker + counterexample miner.
  - Tasks: comment generation with path:line citations; show policy violations; test suggestions for changed public APIs.
  - Eval: policy‑violation precision/recall, developer acceptance.
- Test Gap Discovery & Targeted Test Generation
  - Name: repo_single_testgen
  - Program: U = code, tests, coverage map; E = callgraph (prod↔test); O = uncovered public surface; Θ = generator + pytest verifier.
  - Tasks: rank uncovered critical functions; generate minimal tests; verify via local runner; export per‑module adapters.
  - Eval: new coverage, flake rate, mutation‑score uplift.
- Security Advisory & SAST‑Grounded Patching
  - Name: repo_multi20_sec_patch
  - Program: U = code + SBOM + advisories (CVEs), SAST outputs; E = dep graph; O = vulnerable ranges; T = scanner traces; Θ = advisory→callsite mapper + patch verifier.
  - Tasks: locate exploitation surfaces; propose patches; update dependencies with ripple‑check to build/test.
  - Eval: vuln recall@k, build pass rate post‑patch, diff locality.
- API Usage Migration (v1→v2)
  - Name: repo_single_api_migrate
  - Program: U = code, old/new API docs/changelogs; E = ref map old→new, callsites; C = migration rules; Θ = matcher + rewrite synthesizer + compile/test verifier.
  - Tasks: find callsites; propose rewrites; produce PR with cited diffs.
  - Eval: migration success rate, tests green on first try.
- Performance Regression Explainer
  - Name: repo_single_perf_explain
  - Program: U = benchmarks, profiles, code; O = perf deltas; T = profiler traces; Θ = hot‑path localizer + regressor explainer.
  - Tasks: link regression to commit diff + code paths; propose micro‑patches with citations.
  - Eval: % regressions with actionable RCA, perf regained.
- Runtime Error Crash Triage (Prod logs)
  - Name: logs_single_crash_triage
  - Program: U = logs/stack traces, build info; E = symbolication map; O = error clusters; T = trace spans; Θ = clusterer + “closest prior fix” retriever.
  - Tasks: cluster by signature; map to source regions; output RCA with repo citations; link to prior fixes.
  - Eval: mean time to RCA, duplicate bug rate.
- Database Schema & Query Assistant
  - Name: sql_single_grounded_qa
  - Program: U = DDL, ERDs, migrations; E = FK graph; O = query plans; T = slow query logs; Θ = plan analyzer + index advisor.
  - Tasks: grounded SQL Q&A; safe query synthesis; migration impact analysis.
  - Eval: plan cost reduction; correctness on DB eval set.
- Build System Doctor (Bazel/Make/CMake)
  - Name: build_single_doctor
  - Program: U = build files, cache, action graph; E = target deps; O = cache misses, action times; T = build logs; Θ = invalidation explainer + rule tuner.
  - Tasks: explain slow/fragile targets; propose rule fixes; cite BUILD files.
  - Eval: cache hit rate; rebuild time reduction.
- Dependency Hygiene & Supply‑Chain
  - Name: deps_multi_policy
  - Program: U = lockfiles/SBOM; E = transitive graph; C = org policy (licenses/versions); O = violations; Θ = resolver proposing compliant replacements.
  - Tasks: identify policy breaks; propose minimal diffs; smoke build.
  - Eval: policy conformance; PR merge latency.
- Notebook/Docs ↔ Code Consistency
  - Name: notebook_single_sync
  - Program: U = notebooks, README, tutorials, code; E = example→API links; O = drift; Θ = runnable‑snippet verifier.
  - Tasks: detect stale docs; generate updated examples with citations.
  - Eval: example pass rate in CI; doc PR churn decrease.
- Issue Triage (GitHub/Jira) with Evidence Lift
  - Name: issues_multi_triage
  - Program: U = issues, labels, commits; E = issue↔commit/test links; O = duplicates, missing repro; T = linked logs; Θ = deduper + repro‑steps synthesizer.
  - Tasks: dedupe; propose repro steps; owner routing.
  - Eval: duplicate closure time; first‑response quality.
- Cross‑Repo Architecture Map & Impact Analysis
  - Name: arch_multi20_impact
  - Program: U = many repos, service manifests, RPC/REST specs; E = service call graph; O = blast radius; Θ = impact estimator.
  - Tasks: “If I change X, what breaks?” with downstream citations.
  - Eval: precision of impacted targets; rollback avoidance.
- LLM‑in‑the‑Loop Refactor Planning
  - Name: repo_single_refactor_plan
  - Program: U = code + tests; E = cohesion/coupling graph; O = code smells; Θ = refactor planner + risk scorer + test selector.
  - Tasks: staged refactor plan with guardrails; per‑stage patches/tests with citations.
  - Eval: refactor acceptance; defect escape rate.
- Frontend UI Snapshot & Visual Diff QA
  - Name: ui_single_visual_qa
  - Program: U = storybook stories, screenshots; O = visual diffs; T = rendering logs; Θ = perceptual diff explainer + CSS/DOM patch suggester.
  - Tasks: explain DOM/CSS lineage for diffs; propose fix.
  - Eval: false‑positive reduction; fix time.
- Data‑Pipeline/DAG Grounded Assistant (Airflow/Prefect)
  - Name: dag_single_operator_qa
  - Program: U = DAGs, operators, logs; E = task deps; O = task failures/slowness; T = run logs; Θ = operator doctor + SLA planner.
  - Tasks: diagnose failing task; propose operator config fix with citations.
  - Eval: SLA adherence; re‑run success.

Scaffolding pattern
- ProgramGraph builder for entities/edges; hashed channels to produce z and subgraph z_sub.
- Two selectors: fast heuristic and evidence‑scored.
- Packing utilities: pack_heads and pack_windows.
- Adapters: base prior + on‑the‑fly subgraph adapters; target‑weight presets.
- Verifiers (Θ): pytest/compile/lints/schema/log replay; gate verified buffer for self‑tune.
- CLIs per example:
  - examples/programs/<name>/build.py
  - examples/programs/<name>/run_<backend>_enhanced.py
  - examples/programs/<name>/self_tune.py

AGI‑Style Meta‑Program (Coding AGI)
- Meta‑program M = ⟨P, Γ, R, Π, Λ, Ξ⟩ where:
  - P: installed programs {P1..Pn} (e.g., PR‑review, CI‑triage, SQL‑QA)
  - Γ: typed interfaces/adapters for inter‑program messaging (IO schemas, effect contracts)
  - R: router that selects/combines programs (policy over embeddings, graphs, evidence)
  - Π: planner/binder that decomposes tasks and binds subgoals to Pi with budgets
  - Λ: learning loop (self‑tune) using verifiers Θi as reward/certification
  - Ξ: global memory (episodic/semantic/procedural) with immutable evidence ledger
- Minimal orchestration loop (planner → binder → executor → verifier → tutor):
  - Receive goal → Π decomposes into typed subgoals
  - Bind subgoals to programs with budgets → execute (pack, run, cite)
  - Verify via Θi → on pass, Λ updates adapters/memory; on fail, re‑plan with bounded retries
- Key rules:
  - Only verifiers write to memory (no unverifiable beliefs)
  - Typed results per program (e.g., ReviewComments[], PatchDiff, SQLPlanAdvice) to chain safely
  - Budgets are first‑class (tokens, wall‑time, CI minutes); failures consume budget and trigger re‑planning
- Program composition (interfaces & contracts):
  - Program exposes capabilities with TypedIO signatures, cost models, pre/post‑conditions
  - Router selects plan; Planner binds subgoals to programs
- Router & adapter mixing:
  - Static capability index (embed capability cards + verified exemplars)
  - Contextual routing: score sim(goal, capability) + evidence_prior
  - Adapter mixing: blend subgraph adapters from relevant programs with rank caps; upweight programs whose verifiers passed in session
- Evidence‑first memory (Ξ):
  - Episodic: transcripts/artifacts/diffs with citations
  - Semantic: distilled facts promoted only after repeated verifier passes
  - Procedural: “what worked” policies (router priors, packer heuristics, budgets)
  - All writes include verifier hash and provenance (repo path, CI job, commit)
- Self‑tuning Λ (only checks, no labels):
  - Unit: (goal, plan, packed windows, outputs, verdict, cost)
  - Objective: maximize verifier pass rate / minimize cost
  - Knobs: adapter ranks, packer windows, router priors, tool ordering
  - Method: offline policy eval + online small delta LoRAs per skill; merge after stability
- Safety & rollback:
  - Dry‑run by default; real writes require actuation tokens and a green verifier chain
  - Non‑destructive diffs: patches as PRs with exact citations
  - Time‑boxed retries and circuit breakers; degrade to best single program
  - Auditability: every action links to evidence in the ledger
- Minimal code skeleton:
  ```python
  # core/meta/runtime.py
  class MetaRuntime:
      def __init__(self, programs, router, planner, memory, tuner):
          self.P, self.R, self.Pi, self.Xi, self.L = programs, router, planner, memory, tuner
      def run(self, goal):
          plan = self.Pi.decompose(goal)
          bound = self.Pi.bind(plan, self.P, self.R, self.Xi.budgets(goal))
          for step in bound.steps:
              packed = step.program.pack(step.inputs, self.Xi.context(goal))
              outs = step.program.run(packed)
              verdict = step.program.verify(outs)
              self.Xi.record(goal, step, outs, verdict)
              if not verdict.ok:
                  bound = self.Pi.replan(goal, step, verdict, self.P, self.R, self.Xi)
                  continue
          self.L.update(self.Xi.verified_trajectories())
          return self.Xi.report(goal)
  ```
- Capability graph (installed skills):
  - Programs expose capabilities with typed IO signatures, cost models, and pre/post conditions.
  - Canonical starter capabilities:
    - PRReview: Diff → ReviewComments[]
    - CITriage: FailLogs → RootCause{files[], lines[], reason}
    - TestGen: APISurface → Tests{files[], cmds[]}
    - RefactorPlan: CodeGraph → Plan{stages[], risks[]}
    - SecPatch: SBOM+Advisories → PatchDiff
    - PerfExplain: Profiles+Diff → Hotpath{frames[], regressors[]}
    - SQLQA: Question → SQL{query, safety, rationale}
    - CrashTriage: StackTraces → ClusteredRCA
- Typed interfaces (Γ) with JSON Schemas (abridged example):
  ```json
  {
    "$id": "RootCause",
    "type": "object",
    "required": ["reason", "files", "lines", "evidence"],
    "properties": {
      "reason": {"type": "string"},
      "files": {"type": "array", "items": {"type": "string"}},
      "lines": {"type": "array", "items": {"type": "integer"}},
      "evidence": {"type": "array", "items": {"$ref": "Citation"}}
    }
  }
  ```
- Policies (budgets/safety/adapters/router) — minimal excerpt:
  ```yaml
  budgets:
    default: {tokens: 64000, gpu_sec: 120, ci_min: 10}
    high_risk: {tokens: 96000, gpu_sec: 240, ci_min: 25}
  safety:
    actuation:
      apply_patch: requires: [tests_green, lints_ok, coverage_delta>=0]
      alter_schema: requires: [canary_env, rollback_plan, dba_approval]
  adapters:
    mix_top_k: 3
    ranks:
      repo_single_pr_review: 16
      ci_multi_triage: 8
      repo_single_testgen: 8
  router:
    boost_verified_recent_hours: 24
  ```
- Halt conditions:
  - Budget exhausted; repeated Θ‑fails on same subgoal; missing preconditions (Γ violation); unsafe actuation.
- On‑disk layout (skeleton):
  ```
  coding_agi/
    core/
      runtime.py         # MetaRuntime
      planner.py         # Π
      router.py          # 𝓡
      memory.py          # Ξ
      tuner.py           # Λ
      budgeter.py        # 𝓑
      actuator.py        # Ω
      interfaces/        # Γ JSON Schemas
    programs/            # Installed PCAs
      repo_single_pr_review/
      ci_multi_triage/
      repo_single_testgen/
      repo_single_refactor_plan/
      repo_multi20_sec_patch/
      repo_single_perf_explain/
      sql_single_grounded_qa/
      logs_single_crash_triage/
    packs/
      policies.yaml      # budgets, safety, ranks, gates
      capability_cards/  # router priors + exemplars
    cli/
      cap_build.py       # build base adapters
      cap_run.py         # execute goals end-to-end
      cap_self_tune.py   # Λ on verified buffers
  ```
- Canonical flows:
  - A) Green‑patch from failing CI
    - CITriage: FailLogBundle → RootCause{files, lines, evidence}
    - TestGen: APISurface(slice) → Tests{files, cmds}
    - RefactorPlan: CodeGraphSlice → PatchPlan + PatchDiff
    - Actuator: Open PR; run verifier chain; report
  - B) Feature request to refactor + tests
    - PRReview: Draft changes → ReviewComments + risks
    - TestGen: New APISurface coverage → Tests
    - PerfExplain: Profiles+Diff → Hotpath guard
    - Actuator: PR + passing verifier chain
- Concrete composite to ship first:
  - Name: devops_multi_agent_maintainer
  - Included programs: repo_single_pr_review, ci_multi_triage, repo_single_testgen, deps_multi_policy, logs_single_crash_triage, repo_single_refactor_plan, sql_single_grounded_qa
  - Verifier chain: lints → compile/build → tests → coverage/mutation → policy gates → (optional) staging canary
  - End‑to‑end tasks: take a bug report or failing CI, plan fix, propose patch, generate/adjust tests, tighten deps, explain root cause—with citations and a mergeable PR
- KPIs:
  - Verifier‑pass rate (per program & composite chains)
  - Patch acceptance rate and revert rate
  - MTTR for CI failures and prod crashes
  - Cost per successful action (tokens, GPU seconds, CI minutes)
  - Knowledge retention: re‑solve rate on previously seen failure classes

Quickstart: run installed program smokes
- Agent PR Autofix:
  - python examples/program_conditioned_adapter/examples/agent_pr_autofix/run_smoke_example.py
- Agent Contract Guard:
  - python examples/program_conditioned_adapter/examples/agent_contract_guard/run_smoke_example.py
 - Dataset‑Grounded Training:
  - python examples/program_conditioned_adapter/examples/dataset_grounded_training/run_smoke_example.py
 - Speech‑to‑Speech Adapter:
  - python examples/program_conditioned_adapter/examples/speech_s2s_adapter/run_smoke_example.py
 - Docs Truth Enforcer:
  - python examples/program_conditioned_adapter/examples/docs_truth_enforcer/run_smoke_example.py
- Program Composer Agent:
  - python examples/program_conditioned_adapter/examples/program_composer_agent/run_smoke_example.py
- Self‑Tune PCA:
  - python examples/program_conditioned_adapter/examples/self_tune_pca/run_smoke_example.py
- DOM Exec PCA (RPA):
  - python examples/program_conditioned_adapter/examples/dom_exec_pca/run_smoke_example.py
- Hypothesis Runner PCA:
  - python examples/program_conditioned_adapter/examples/hypothesis_runner_pca/run_smoke_example.py
- Temporal State Adapter:
  - python examples/program_conditioned_adapter/examples/temporal_state_adapter/run_smoke_example.py
- Counterfactual Adapter:
  - python examples/program_conditioned_adapter/examples/counterfactual_adapter/run_smoke_example.py
- Proof‑Carrying Adapter:
  - python examples/program_conditioned_adapter/examples/proof_carrying_adapter/run_smoke_example.py
- Tool Policy Adapter:
  - python examples/program_conditioned_adapter/examples/tool_policy_adapter/run_smoke_example.py
- Calibrated Decoder Adapter:
  - python examples/program_conditioned_adapter/examples/calibrated_decoder_adapter/run_smoke_example.py
- Skill Shard Distiller:
  - python examples/program_conditioned_adapter/examples/skill_shard_distiller/run_smoke_example.py
- Multi‑Repo Interface Mapper:
  - python examples/program_conditioned_adapter/examples/repo_interface_mapper/run_smoke_example.py
- PR Review:
  - python examples/program_conditioned_adapter/examples/repo_single_pr_review/run_smoke_example.py
- Test Generation:
  - python examples/program_conditioned_adapter/examples/repo_single_testgen/run_smoke_example.py
- CI Failure Triage:
  - python examples/program_conditioned_adapter/examples/ci_multi_triage/run_smoke_example.py
- Refactor Plan:
  - python examples/program_conditioned_adapter/examples/repo_single_refactor_plan/run_smoke_example.py
- Deps Policy:
  - python examples/program_conditioned_adapter/examples/deps_multi_policy/run_smoke_example.py
- Crash Triage:
  - python examples/program_conditioned_adapter/examples/logs_single_crash_triage/run_smoke_example.py
- API Migrate:
  - python examples/program_conditioned_adapter/examples/repo_single_api_migrate/run_smoke_example.py
- Perf Explain:
  - python examples/program_conditioned_adapter/examples/repo_single_perf_explain/run_smoke_example.py
- SQL Grounded QA:
  - python examples/program_conditioned_adapter/examples/sql_single_grounded_qa/run_smoke_example.py
- Notebook Sync:
  - python examples/program_conditioned_adapter/examples/notebook_single_sync/run_smoke_example.py

Coding AGI example (meta‑program)
- Location: examples/program_conditioned_adapter/examples/coding_agi/core
  - runtime.py: minimal MetaRuntime loop
  - planner.py: single‑step planner with simple retry
  - router.py: naive keyword router across installed programs
  - memory.py: evidence ledger with verified() filter
- Intent:
  - Demonstrates the orchestration shell that binds installed programs, routes a goal, executes, verifies, and records evidence.
  - Keep using individual program CLIs above for now; the meta‑program illustrates structure and contracts for composition.

Tips
- Use --structured and --require-citations when running run.py directly to get typed outputs with anchored evidence.
- Prefer --code-recall-preset for code‑centric tasks (review, triage, testgen) to improve path:line citation density.
- Reuse adapters across runs by pointing --adapters-dir to the same artifacts directory for faster iteration.

### Program‑Conditioned Adapter Taxonomy

1) Databases (SQL/OLAP/Vector)
- Adapters:
  - db_single_grounded_qa (schema‑aware, query‑cited answers)
  - db_change_planner (DDL/DML diff plans with rollbacks)
  - db_profile_router (route questions to the right DB/warehouse)
- Signals: schemas, foreign keys, indexes, stats, EXPLAIN plans, sample rows, lineage/owners
- Verifiers: dry‑run queries, EXPLAIN shape matching, row counts / checksum comparisons

2) APIs / Microservices (REST, gRPC, GraphQL)
- Adapters:
  - api_grounded_qa (contract + example‑driven answers)
  - api_orchestrator_planner (multi‑call plans with rate‑limit guards)
  - api_change_guard (breaking‑change detector from OpenAPI diffs)
- Signals: OpenAPI/IDLs, examples, Postman collections, latency/error SLOs, auth scopes
- Verifiers: contract validation, schema conformance, replayable mock calls

3) CLIs / Toolchains
- Adapters:
  - cli_grounded_runner (flag‑aware suggestions with manpages)
  - cli_batch_planner (compose safe command pipelines)
- Signals: --help/manpages, completion specs, exit codes, stdout/stderr exemplars
- Verifiers: sandbox execution, exit‑code + regex assertions, snapshot diffs

4) DAGs & Data Pipelines (Airflow/Prefect/DVC)
- Adapters:
  - dag_grounded_qa (task/source‑of‑truth answers)
  - dag_change_planner (safe task edits, dependency impact)
- Signals: task graph, schedules, upstream/downstream, run logs, assets/lineage
- Verifiers: dry‑run DAG parse, dependency reachability, test task replays

6) Logs / Telemetry / Traces (ELK, OpenTelemetry)
- Adapters:
  - telemetry_incident_summarizer
  - log_rootcause_router (signal‑to‑service triage)
- Signals: metrics, spans, error clusters, golden signals (latency, errors, saturation)
- Verifiers: query reproducibility, time‑bounded consistency, counterfactual checks

7) Notebooks & Repro Bundles (Jupyter, W&B/Runs)
- Adapters:
  - nb_grounded_qa (cell‑cited answers)
  - experiment_compare_planner (A/B/C run comparison & next‑step plan)
- Signals: executed cell graph, outputs, environment YAML, run artifacts
- Verifiers: re‑execute minimal cells, metric threshold checks

8) Spreadsheets / BI / Reports
- Adapters:
  - sheet_formula_assistant (range‑aware, versioned)
  - bi_card_explainer (dashboard element lineage to sources)
- Signals: sheets, named ranges, pivot/measure defs, data sources
- Verifiers: formula evaluation, sample recomputes, chart data parity

9) Document & Knowledge Graphs
- Adapters:
  - doc_citation_qa (page/line‑cited grounding)
  - kg_path_reasoner (graph‑constrained multi‑hop answers)
- Signals: chunked docs with anchors, KG nodes/edges, provenance metadata
- Verifiers: citation coverage, edge‑path validation, contradiction sweeps
 - Program: library_grounded_qa
   - Ask anything across a curated shelf; answer anchored to book:page:line with contradiction checks.
   - Signals: OCR’d PDFs/EPUBs, chapter/section graph, citation graph, footnotes, glossary/indices
   - Verifiers: page‑bounded quote windows, cross‑source agreement tests, edition disambiguation
   - Flow: Select → Pack (page anchors) → Adapt (shelf‑conditioned) → Generate → Verify → Cite
Tabular / Dataframe Grounding
- Program: data_table_grounded_qa
  - Signals: table schemas, samples, constraints, data contracts
  - Verifiers: in‑process replay of code/SQL, shape/type assertions
  - Output: TableAnswer{answer, code, result_preview}

10) Messaging & Workflows (Email, Slack, Ticketing)
- Adapters:
  - ticket_triage_router (assign/label with SLA logic)
  - email_grounded_summarizer (thread‑aware with links)
- Signals: threads, assignees, SLAs, labels, actions taken, checklists
- Verifiers: policy conformance, audit trail links, assignee availability

11) Browsers / DOM / RPA
- Adapters:
  - browser_dom_grounded_qa (selector‑cited DOM answers)
  - rpa_flow_planner (deterministic macro plans)
- Signals: DOM trees, ARIA roles, screenshot hashes, stable selectors
- Verifiers: headless replay, visual diff thresholds, selector stability

12) Build/CI/Test Systems
- Adapters:
  - ci_failure_router (map failing test ↔ culprit commit/module)
  - test_gap_planner (coverage‑guided test authoring)
- Signals: build graph, test results, coverage maps, flake profiles
- Verifiers: re‑run failing shards, coverage deltas, determinism checks

13) Robotics / Simulation / Game Engines
- Adapters:
  - sim_policy_planner (scenario scripts with safety gates)
  - robot_task_grounder (URDF/env‑aware task plans)
- Signals: URDF/SDF, scene graphs, sensor specs, reward curves, sim configs
- Verifiers: rollout metrics, safety constraints, reproducible seeds
 - Program: game_character_planner
   - Auto-plan a build, grind route, and respec path for a given playstyle; verify DPS/TTK vs boss profiles and route feasibility.

14) Media Pipelines (Video/Audio/CAD)
- Adapters:
  - media_render_planner (graph of transforms, codecs, budgets)
  - cad_param_qa (dimension‑cited answers)
  - speech_s2s_adapter (ASR→TTS with prosody/alignment/latency checks)
- Signals: timelines, node graphs, codec params, geometry/constraints
- Verifiers: hash‑based artifact checks, visual probes, parameter bounds

15) Geospatial / GIS
- Adapters:
  - gis_query_qa (layer/source‑cited geospatial answers)
  - route_plan_validator (cost/safety/weather constraints)
- Signals: layers, CRS, topology, rasters, traffic/weather feeds
- Verifiers: topology rules, cross‑layer joins, route feasibility sims

### Core Utility Adapters (compose with any domain)

- Memory
  - memory_grounded_recall: consolidates verified Q/A + traces into a compact, citation‑rich cache the PCA can inject.
  - memory_incremental_distill: turns verified interactions into tiny add‑on adapters (per‑topic/per‑module).
  - skill_shard_distiller: converts every verified solve into a persistent skill‑shard LoRA and updates the router; ships adapters/shards/*, router weights, and an eval curve.
- Temporal / Dynamics
  - temporal_state_adapter: encodes recent state transitions S_t…S_t−k so the model reasons over dynamics, not snapshots; signals: event logs, diffs, counters; verifiers: time‑bounded invariants, rollback replay.
- Counterfactual / What‑If
  - counterfactual_adapter: swaps contracts/policies {C→C′} and re‑computes Δθ to answer “under different rules”; signals: delta‑contracts; verifiers: consequence simulation, invariant maintenance.
- Proof‑Carrying
  - proof_carrying_adapter: stores minimal proof objects alongside Δθ (test IDs, pages, checksums) so any claim references a proof pointer; verifier: deterministic replay of proof pointer.
- Toolformer‑Binding
  - tool_policy_adapter: Δθ specialized to a whitelist of tools/APIs with schemas and rate limits; verifiers: mock replay, schema conformance, rate guard.
- Distribution‑Aware Decoding
  - calibrated_decoder_adapter: adjusts token posterior calibration (ECE↓); verifiers: held‑out Brier/ECE + factuality checks.
- Routing
  - router_program_selector: choose correct program adapter (DB vs API vs Repo) using program signatures + confidence bounds.
  - router_task_selector: select task adapters (qa vs planner vs summarizer vs change_guard).
- Graphing
  - program_graph_builder: normalize any program to ⟨Entities, Edges, Artifacts, Contracts, State, Observables, Traces⟩ and cache embeddings.
- Summarizing
  - program_state_summarizer: produce time‑boxed, versioned digests of state/metrics with provenance anchors.

### Canonical PCA Interface (what every adapter implements)

- Select(): question‑aware selection of subgraph/segments/windows from the program.
- Pack(): deterministic context packaging with anchored snippets (e.g., path:line or node:id).
- Embed(): multi‑factor embedding of program‑specific features (schemas, contracts, graphs, traces).
- Adapt(): produce or mix LoRA‑like deltas for the current LM layer targets (attention/MLP) with a stable gating schedule.
- Verify(): run program‑native checks (dry‑run SQL, API mock, DAG parse, test replay).
- Cite(): append anchors to every claim (files, schema tables, API endpoints, task nodes).
- Log(): emit minimal, privacy‑respecting telemetry for reproducibility and distillation.

### Naming Patterns

- <program_domain>_<scope>_<task>
- Examples to ship:
  - repo_single_grounded_planner
  - db_single_grounded_qa
  - api_multi10_orchestrator_planner
  - dag_single_change_planner
  - iac_stack_risk_analyzer
  - browser_dom_grounded_qa
  - sheet_single_formula_assistant
  - telemetry_incident_summarizer

### Recommended Build Order

1) Router + Memory (foundational)
   - router_program_selector, memory_grounded_recall
2) DB + API (broadest utility)
   - db_single_grounded_qa, api_grounded_qa, with strict contract/schema verifiers
3) DAG + CI/Test (devops leverage)
   - dag_grounded_qa, ci_failure_router
4) Browser/RPA (end‑to‑end usability)
   - browser_dom_grounded_qa, rpa_flow_planner
5) Logs/Telemetry (oncall leverage)
   - telemetry_incident_summarizer, log_rootcause_router

Each domain shares the same PCA skeleton, so once you’ve implemented Select/Pack/Embed/Adapt/Verify/Cite, new adapters are mostly feature mappers + verifiers.
