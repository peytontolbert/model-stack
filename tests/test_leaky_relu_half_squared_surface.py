from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import autotune.bench.mlp as autotune_bench_mlp
from autotune import presets as autotune_presets
from autotune.spaces import Choice
from specs import ops as spec_ops


def test_specs_ops_exposes_leaky_relu_half_squared_activation():
    activations = spec_ops.list_ops("activations")["activations"]

    assert "leaky_relu_0p5_squared" in activations
    assert spec_ops.get_op("activations", "leaky_relu_0p5_squared") is not None


def test_blocks_wiring_search_space_includes_leaky_relu_half_squared():
    space = autotune_presets.blocks_wiring()
    activation = space.parameters["activation"]

    assert isinstance(activation, Choice)
    assert "leaky_relu_0p5_squared" in activation.options


def test_bench_mlp_default_activation_list_includes_leaky_relu_half_squared():
    benchmark_source = autotune_bench_mlp.bench_mlp.__code__.co_consts

    assert any(
        isinstance(const, tuple) and "leaky_relu_0p5_squared" in const
        for const in benchmark_source
    )
