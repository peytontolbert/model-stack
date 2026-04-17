# Program-Conditioned Adapters White Paper

This file is now the short summary for PCA. The canonical long-form design lives in [program_adapter.md](program_adapter.md).

## Executive Summary

Program-Conditioned Adapters (PCA) specialize a pretrained language model to an executable system by:
- building a `ProgramGraph` over the system's entities, edges, artifacts, and contracts
- embedding that structure into a compact, channelized representation
- synthesizing a small reversible LoRA delta at inference time
- selecting evidence windows and enforcing artifact-backed citations
- optionally verifying outputs and keeping only the deltas that improve measurable behavior

The intended result is grounded, auditable generation that stays tied to the target program rather than relying on the base model's generic prior alone.

## Read Next

- [README.md](README.md): practical entry point and CLI overview
- [program_adapter.md](program_adapter.md): full theory, architecture, and evaluation writeup
- [examples/README.md](examples/README.md): checked-in examples and support scripts
