# Safety Package

`safety/` is the placeholder package for policy-as-code and runtime guardrails.
At the moment, the implemented surface is intentionally minimal:

- `guard.py` defines a `Verdict` shape with decisions `allow`, `block`, and
  `rewrite`.
- `assess(prompt, response)` is only a typed stub and does not currently enforce
  policy.

Do not treat this package as production moderation. It documents the intended
integration point, not a complete safety system.

## Intended Contract

A complete guard implementation should accept:

- prompt text
- optional model response text
- optional request metadata
- optional policy configuration

and return:

```python
Verdict(decision="allow" | "block" | "rewrite", reasons=[...])
```

The call should be deterministic for the same input and policy version.

## Integration Points

Expected future integration surfaces:

- `serve/`: request-time prompt and response guardrails
- `train/`: optional safety-aware loss shaping or data filtering
- `eval/`: red-team and policy regression suites
- `governance/`: policy version and safety evaluation metadata in release
  artifacts

## Implementation Requirements

Before using this package for real enforcement:

1. Replace the stub in `guard.py` with an implemented policy engine.
2. Add unit tests for allow/block/rewrite decisions.
3. Add fixtures for prompt-only, response-only, and prompt-response checks.
4. Add versioned policy metadata so release artifacts can state which policy was
   used.
5. Decide whether violations raise, return structured verdicts, or both.

Keep the safety layer explicit. Runtime code should call the guard at clear
boundaries instead of hiding moderation behavior inside unrelated model or
sampling utilities.
