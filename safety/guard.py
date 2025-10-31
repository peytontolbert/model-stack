# safety/guard.py
from typing import Literal, NamedTuple
class Verdict(NamedTuple):
    decision: Literal["allow","block","rewrite"]
    reasons: list[str]
def assess(prompt:str, response:str|None) -> Verdict: ...
