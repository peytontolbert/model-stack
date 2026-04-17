from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import model.causal as model_causal_mod
import model.encoder as model_encoder_mod
import model.heads as model_heads_mod
import model.prefix_lm as model_prefix_lm_mod
import model.seq2seq as model_seq2seq_mod
import runtime as runtime_pkg
import runtime.causal as runtime_causal_mod
import runtime.encoder as runtime_encoder_mod
import runtime.heads as runtime_heads_mod
import runtime.prefix_lm as runtime_prefix_lm_mod
import runtime.seq2seq as runtime_seq2seq_mod


def test_model_core_modules_are_runtime_aliases():
    assert model_causal_mod is runtime_causal_mod
    assert model_prefix_lm_mod is runtime_prefix_lm_mod
    assert model_encoder_mod is runtime_encoder_mod
    assert model_seq2seq_mod is runtime_seq2seq_mod
    assert model_heads_mod is runtime_heads_mod


def test_runtime_package_exports_core_models_and_heads():
    assert runtime_pkg.CausalLM is runtime_causal_mod.CausalLM
    assert runtime_pkg.TransformerLM is runtime_causal_mod.TransformerLM
    assert runtime_pkg.PrefixCausalLM is runtime_prefix_lm_mod.PrefixCausalLM
    assert runtime_pkg.EncoderModel is runtime_encoder_mod.EncoderModel
    assert runtime_pkg.EncoderDecoderLM is runtime_seq2seq_mod.EncoderDecoderLM
    assert runtime_pkg.SequenceClassificationHead is runtime_heads_mod.SequenceClassificationHead
    assert runtime_pkg.TokenClassificationHead is runtime_heads_mod.TokenClassificationHead
