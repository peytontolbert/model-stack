# Smoke Repo for Repo-Grounded Adapters

This tiny repository contains a minimal attention example used to validate that
repo-grounded adapter selection, context packing, and generation behave
reasonably on a small, isolated codebase.

Files:
- `attention_demo.py`: small functions and docstrings describing attention.

Usage: point the runner's `--repo` to this directory and use prompts that
mention "attention" or the function names to drive selection.

