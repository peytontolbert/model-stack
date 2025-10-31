import torch
from tensor import S, unify, graph_shapes, infer, broadcast_plan


def test_infer_simple():
    sol, deriv = infer("B*H==G*Q", {"B": 2, "H": 8, "G": 4})
    assert sol.get("Q") == 4
    assert "Q" in deriv


def test_broadcast_plan():
    plan, deriv = broadcast_plan("B,H,T,D", "1,1,T,D")
    assert plan["dim[0]"] in ("expand_b", "expand_a", "match", "incompatible")

