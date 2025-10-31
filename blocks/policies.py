from typing import Literal

NormPolicy = Literal["prenorm", "postnorm", "parallel"]


def validate_policy(policy: NormPolicy) -> NormPolicy:
    if policy not in ("prenorm", "postnorm", "parallel"):
        raise ValueError(f"Unknown norm policy: {policy}")
    return policy


