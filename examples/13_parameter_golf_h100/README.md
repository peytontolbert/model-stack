## Parameter Golf H100 Ternary Runtime Recipe

This example records the H100 Parameter Golf setup used for runtime-row ternary
training and export experiments. The Parameter Golf source lives at
`other_repos/parameter-golf`; the H100 copy used the same `train_gpt.py` and
`README.md` hashes as the local checkout.

Use the scripts in this directory for restartable H100 work:

```bash
# Build the local CUDA/native runtime for H100.
bash examples/13_parameter_golf_h100/build_runtime_h100.sh

# Launch the current best local ternary 8xH100 recipe.
PG_DIR=/root/parameter_golf \
PG_PRESET=runtime_row_1024x7_relu2_mlp3 \
bash examples/13_parameter_golf_h100/run_pg_8xh100.sh

# Export a trained ternary artifact into the runtime-row BitNet bundle.
PG_DIR=/root/parameter_golf \
ARTIFACT=/root/parameter_golf/final_model.ternary.ptz \
OUT=/root/transformer_10_h100/artifacts/parameter_golf/final_model.runtime_bitnet.pt \
bash examples/13_parameter_golf_h100/export_runtime_bundle.sh
```

The `h100_remote_snapshot/` directory preserves H100-side runtime files that
differed from local at the time of capture. Those are saved for recovery, not
merged, because the active local tree has newer BitNet decode changes.

### Current Best Restart Preset

Use `PG_PRESET=runtime_row_1024x7_relu2_mlp3` for the best restartable recipe
we have locally. It reflects the 8xH100 tuning pass after dropping the stale
9-layer smoke preset:

```bash
PG_DIR=/root/parameter_golf \
DATA_PATH=/root/parameter_golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/root/parameter_golf/data/tokenizers/fineweb_1024_bpe.model \
PG_PRESET=runtime_row_1024x7_relu2_mlp3 \
bash examples/13_parameter_golf_h100/run_pg_8xh100.sh
```

Key defaults:

```text
MODEL_DIM=1024
NUM_LAYERS=7
NUM_HEADS=16
NUM_KV_HEADS=4
MLP_MULT=3
ACTIVATION=relu2
TRAIN_SEQ_LEN=4096
TRAIN_BATCH_TOKENS=524288
BITNET_SCALE_LAYOUT=runtime_row
SLIDING_EVAL=1
SLIDING_EVAL_STRIDE=64
```

### Model Stack QAT Smoke

The active root `other_repos/parameter-golf/train_gpt.py` can now opt into
Model Stack trainable BitNet modules instead of plain `CastedLinear` layers:

```bash
PG_DIR=/root/parameter_golf \
MODEL_STACK_ROOT=/root/transformer_10_h100 \
MODEL_STACK_BITNET_QAT=1 \
MAX_WALLCLOCK_SECONDS=60 \
PG_PRESET=runtime_row_512 \
bash examples/13_parameter_golf_h100/run_pg_8xh100.sh
```

When enabled, training uses `compress.quantization.TrainableBitNetLinear` and
writes `final_model.model_stack_bitnet.pt` with packed `QuantizedLinearBitNet`
state for the QAT linear modules.

### BitNet Kernel Benchmark Gate

Do not treat Model Stack BitNet as an end-to-end speed win until this benchmark
shows the selected runtime path beating the dense dequantized baseline for the
active Parameter Golf shapes. The raw H100 int8 accumulation kernel now wins the
large prefill microbenchmark, but the dynamic activation-quantization and
rescale/output epilogue still prevent the direct-from-float path from winning.
The QAT training module currently uses a float shadow weight and PyTorch
`F.linear` for forward/backward; the packed BitNet CUDA kernels are runtime
kernels unless a trainable kernel path is added.

Local smoke on a free GPU:

```bash
conda run -n ai python examples/13_parameter_golf_h100/bench_pg_bitnet_kernels.py \
  --device cuda:1 \
  --preset runtime_row_1024x7_relu2_mlp3 \
  --rows 1,4096 \
  --warmup 10 \
  --iters 30 \
  --repeats 1 \
  --sync-wall \
  --consume-output
```

H100 gate before a full 8x run:

```bash
bash examples/13_parameter_golf_h100/run_bitnet_kernel_gate.sh
```

To prove the opt-in CUTLASS fused int8 backend against the current default
cuBLASLt full-output path, first build with CUTLASS and the experimental SM90a
target, then run the CUTLASS variant gate:

```bash
MODEL_STACK_CUTLASS_PATH=/root/cutlass \
MODEL_STACK_ENABLE_SM90A_EXPERIMENTAL=1 \
MODEL_STACK_CUDA_ARCH_LIST=9.0 \
python setup.py build_ext --inplace --force

INCLUDE_INT8_CUTLASS_FUSED_VARIANT=1 \
ROWS=65536 \
WARMUP=20 \
ITERS=100 \
REPEATS=3 \
MIN_CUTLASS_PREQUANT_SPEEDUP_VS_CURRENT=1.2 \
MIN_CUTLASS_PREQUANT_SPEEDUP_VS_DEQUANT=1.0 \
MIN_CUTLASS_DIRECT_MLP_SPEEDUP_VS_DEQUANT=1.0 \
MIN_DIRECT_INT8_SPEEDUP=0 \
MIN_PREQUANT_INT8_SPEEDUP=0 \
bash examples/13_parameter_golf_h100/run_bitnet_kernel_gate.sh
```

Use `INCLUDE_INT8_BACKEND_VARIANTS=1` only when intentionally checking the
known-slower no-cublasLt and SM90a WGMMA variants.

The Parameter Golf `relu2` MLP3 preset should also pass the composed MLP-pair
gate. This includes the activation between `fc` and `proj`, so it catches cases
where faster individual linears are erased by activation or requantization
overhead:

```bash
/usr/local/bin/python examples/13_parameter_golf_h100/bench_pg_bitnet_mlp_subgraph.py \
  --device cuda:0 \
  --preset runtime_row_1024x7_relu2_mlp3 \
  --rows 65536 \
  --activation-quant dynamic_int8 \
  --warmup 30 \
  --iters 200 \
  --repeats 5 \
  --consume-output \
  --include-int8-cutlass-fused-variant \
  --min-cutlass-speedup-vs-current 1.05 \
  --min-cutlass-speedup-vs-dequant 1.05 \
  --jsonl
```

The important columns are:

```text
dense_bitnet_dequant_ms: dense fallback on the exact dequantized BitNet weight
model_stack_auto_ms: the default QuantizedLinearBitNet runtime path
model_stack_forced_backend_ms: forced packed BitNet backend
direct_bitnet_int8_from_float_ms: direct native BitNet W2A8 path without module wrapper overhead
activation_int8_quantize_ms: native CUDA activation quantization frontend alone when available, excluding int8 matmul
prequant_int8_accum_only_ms: raw int8 Tensor Core accumulation only, before dequantizing scales/output dtype
prequant_int8_matmul_ms: int8 matmul after activation quantization is already done
*_speedup_vs_dequant: must be >1.0x to claim that path is faster than dense
```

The wrapper runs two contracts:

```text
activation_quant=none: exact packed runtime-row BitNet math
activation_quant=dynamic_int8: faster W2A8 candidate with activation quantization error
```

Observed single-H100 gate on `NVIDIA H100 80GB HBM3` after building the native
extension for `sm_90`/`sm_90a`:

```text
activation_quant=none forced backend:
  rows=65536 attn_out/lm_head 1024x1024: ~45 ms vs dense ~0.18 ms
  rows=65536 mlp_up/mlp_down 1024x3072/3072x1024: ~133-135 ms vs dense ~0.53 ms

activation_quant=dynamic_int8 forced backend:
  direct native BitNet now routes to the same no-pre-scale W2A8 CUDA frontend
  as native int8, so direct_bitnet and native_int8 are effectively equal.
  latest tuned rows=65536 raw accum-only kernel:
    mlp_down 3072x1024: ~0.278 ms vs dense ~0.527 ms, ~1.90x faster
    mlp_up 1024x3072: ~0.366 ms vs dense ~0.533 ms, ~1.46x faster
    attn_out 1024x1024: ~0.135 ms vs dense ~0.186 ms, ~1.38x faster
  latest tuned rows=65536 full prequantized path:
    mlp_down 3072x1024: ~0.430 ms vs dense ~0.527 ms, ~1.22x faster
    mlp_up 1024x3072: ~0.853 ms vs dense ~0.533 ms, not yet faster
    attn_out 1024x1024: ~0.301 ms vs dense ~0.186 ms, not yet faster
  latest tuned CUTLASS fused rows=65536 full prequantized path
  (`/tmp/pg_quant_default_final_h100_20260426.jsonl`,
  128x256x128 tile, 2x1x1 cooperative cluster, default 8-row warp
  activation quant frontend with shared-cache dynamic quantization, 2-warp
  frontend for >2048 columns):
    attn_qkv_fused 1024x1536: 0.208 ms vs current 0.444 ms, 2.13x faster; vs dense 0.273 ms, 1.31x faster
    attn_out 1024x1024: 0.141 ms vs current 0.307 ms, 2.18x faster; vs dense 0.198 ms, 1.40x faster
    mlp_up 1024x3072: 0.381 ms vs current 0.856 ms, 2.25x faster; vs dense 0.530 ms, 1.39x faster
    mlp_down 3072x1024: 0.300 ms vs current 0.437 ms, 1.45x faster; vs dense 0.543 ms, 1.81x faster
    lm_head 1024x1024: 0.141 ms vs current 0.308 ms, 2.19x faster; vs dense 0.198 ms, 1.41x faster
    gate passed with MIN_CUTLASS_PREQUANT_SPEEDUP_VS_CURRENT=1.2
    rejected in this sweep: 256x256x128 tile, 128x512x128 tile, 1x2x1,
    2x2x1, and 4x1x1 clusters, and pingpong schedule
  native activation quantization frontend alone, same log:
    1024-wide rows: ~0.081 ms
    3072-wide rows: ~0.226 ms
    disabling shared-cache quantization with
    MODEL_STACK_DISABLE_INT8_QUANT_SHARED_CACHE=1 returned to ~0.084 ms and
    ~0.253 ms in `/tmp/pg_sharedcache_disabled_h100_20260426.jsonl`
  latest tuned rows=65536 direct-from-float path:
    default warp activation quantization cuts the direct CUTLASS path to:
    attn_qkv_fused 0.286 ms, attn_out 0.218 ms, mlp_up 0.469 ms,
    mlp_down 0.522 ms, lm_head 0.218 ms.
    This makes mlp_up and mlp_down end-to-end speed wins vs dense; qkv,
    attention output, and lm_head are close but still below dense.
    Disable with MODEL_STACK_DISABLE_INT8_QUANT_WARP_ROW=1 for rollback.
    MODEL_STACK_INT8_QUANT_WARP_ROWS_PER_BLOCK can select 4, 8, or 16 rows
    per warp-row CTA; 8 remains the default. Rejected quant frontend variants:
    4-row and 16-row defaults, using the 2-warp wide frontend for 1024-column
    rows, and the opt-in MODEL_STACK_ENABLE_INT8_QUANT_VEC4=1 packed-store
    path, which regressed 1024-wide quantization to ~0.086 ms.
  static/provided-scale activation quantization now has a one-pass native
  frontend, but it did not close the 1024-wide direct-path gap in the synthetic
  benchmark. It is mainly useful if QAT/calibration proves a static scale keeps
  Parameter Golf validation quality.

activation_quant=dynamic_int8 auto backend:
  the current/default int8 backend keeps the dense cached-weight fallback on
  Hopper prefill, because the direct W2A8 path loses there. With the opt-in
  CUTLASS fused int8 backend enabled, `QuantizedLinearBitNet.runtime_linear`
  and the native C++ `linear_module_forward` path now route the measured
  Parameter Golf MLP shapes through direct W2A8:
    mlp_up 1024x3072: auto 0.470 ms, native auto 0.469 ms vs dense 0.534 ms, ~1.14x faster
    mlp_down 3072x1024: auto 0.524 ms, native auto 0.522 ms vs dense 0.544 ms, ~1.04x faster
  QKV, attention output, and lm_head remain dense/packed-policy cases until
  the direct-from-float path beats dense there too. The prequantized CUTLASS
  matmul already beats dense for all listed shapes, so the remaining direct
  gap is activation quantization/launch overhead, not the int8 GEMM itself.
  Gate log with native module dispatch:
    `/tmp/pg_quant_native_module_gate_h100_20260426.jsonl`
  Rebuilt gate after the fused `relu2` MLP quant path:
    `/tmp/pg_quant_native_module_gate_h100_20260426_after_relu2_fuse.jsonl`
  composed `relu2` MLP3 subgraph after fusing `relu2` with down-projection
  activation quantization:
    log: `/tmp/pg_bitnet_mlp_subgraph_h100_20260426_after_relu2_fuse.jsonl`
    current/default model_stack_mlp_module: 1.155 ms vs dense 1.516 ms, ~1.31x faster
    CUTLASS fused model_stack_mlp_module: 0.989 ms vs dense 1.518 ms, ~1.53x faster
    CUTLASS fused vs current/default: ~1.17x faster
    native linear-pair without fused activation quantization remains ~1.50 ms,
    so the MLP-pair gain comes from the fused activation-quant down path plus
    the CUTLASS int8 linears.

activation_quant=dynamic_int8 SM90a WGMMA variant:
  requires MODEL_STACK_ENABLE_SM90A_EXPERIMENTAL=1 at build time and
  MODEL_STACK_ENABLE_INT8_LINEAR_WGMMA=1 with cublasLt disabled at runtime.
  rows=65536 attn_out 1024x1024: ~73 ms vs dense ~0.18 ms
  rows=65536 mlp_up/mlp_down 1024x3072/3072x1024: ~210 ms vs dense ~0.47-0.50 ms
  this variant is much slower than both dense fallback and the cublasLt int8
  path, so it remains opt-in only. It is also disabled below its 64-row tile
  size because sub-tile rows produced incorrect rows=8 results during tuning.
```

This means the raw int8 GEMM kernel and the opt-in CUTLASS fused scale/output
kernel are speed wins on H100 for the large Parameter Golf prefill shapes
relative to both the previous full-output cuBLASLt path and dense dequantized
BitNet. The composed PG `relu2` MLP path is now also a speed win after fusing
activation with down-projection quantization. The remaining gap is not the
prequantized GEMM kernel anymore; it is the dynamic activation quantization and
full direct-from-float/module path for the standalone 1024-wide attention and
lm-head projections.

Profile a single candidate on H100 with Nsight Compute:

```bash
ACTIVATION_QUANT=dynamic_int8 \
SHAPE=mlp_up:1024:3072 \
ROWS=65536 \
bash examples/13_parameter_golf_h100/profile_bitnet_kernel_h100.sh
```

For the current 1024-wide preset, distributed 8xH100 mostly changes rows per
rank, not the per-rank linear kernel shape. With the default
`TRAIN_BATCH_TOKENS=524288` and `NPROC_PER_NODE=8`, the training-sized row count
is roughly `65536` per rank. Current 1024/3072 feature sizes do not enter the
split-K path because that dispatcher gate starts at `in_features >= 4096`; tune
decode with `MODEL_STACK_BITNET_DECODE_CTA_MULTIPLIER`, and tune prefill by
editing/rebuilding the tiled kernel constants or adding a better kernel.

Observed local artifact/logs:

```text
best standard roundtrip: val_bpb=1.2332
trusted fixed sliding run: val_bpb=1.2221
artifact budget: under 16 MB with code and packed ternary weights
```

Notes from the 8xH100 sweep:

```text
TRAIN_BATCH_TOKENS=786432 used more VRAM but reduced tokens/sec.
TRAIN_BATCH_TOKENS=1048576 was unstable in the remote session.
Naive learned bigram and strict token PPM mixers hurt BPB.
XSA-all was much worse early and was aborted.
Standalone packed ternary matmul is still not the win condition for 512/1024-wide PG layers; fused end-to-end runtime is.
```

### Train 512-Wide Runtime-Row Ternary

This is the 300 second run that produced a 4.19 MB packed artifact with full
validation after the wall-clock stop.

```bash
cd other_repos/parameter-golf

RUN_ID=ternary_runtime_row_sp1024_ctx4096_300s_fix \
DATA_PATH=/root/parameter_golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/root/parameter_golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
TRAIN_SEQ_LEN=4096 \
YARN_MAX_LEN=4096 \
MAX_WALLCLOCK_SECONDS=300 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=524288 \
TRAIN_BATCH_TOKENS=524288 \
TRAIN_LOG_EVERY=200 \
WARMUP_STEPS=1 \
BITNET_GROUP_SIZE=64 \
BITNET_SCALE_LAYOUT=runtime_row \
FP_STORAGE=0 \
MODEL_DIM=512 \
NUM_LAYERS=9 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=2 \
SEQ_SCHEDULE_FRACTION=0.33 \
BATCH_SCHEDULE_FRACTION=0.33 \
MATRIX_LR=0.04 \
SCALAR_LR=0.04 \
TIED_EMBED_LR=0.05 \
HEAD_LR=0.008 \
MUON_BACKEND_STEPS=5 \
LOGIT_SOFTCAP=30 \
QK_GAIN_INIT=1.5 \
python train_gpt.py
```

Observed H100 result:

```text
params:17585224 L:9 d:512 h:8 kv:4 ws:1 ga:8 s:1337
step:580/10000 val_loss:2.5454 val_bpb:1.5323 train_time:301009ms zero_frac:0.370
artifact:4.19MB ternary:16515072(3455409B) fp:545864(1091728B) code:72921
budget:4260641/16000000 (4.26/16.00MB) FITS
final_ternary_roundtrip val_loss:2.5455 val_bpb:1.5323
```

### Train Wider 1024 Smoke

This run checks the wider/fewer-layer direction that better matches packed
kernel economics while staying under the 16 MB artifact budget.

```bash
cd other_repos/parameter-golf

RUN_ID=ternary_runtime_row_sp1024_ctx4096_9x1024_smoke \
VOCAB_SIZE=1024 \
TRAIN_SEQ_LEN=4096 \
YARN_MAX_LEN=4096 \
MAX_WALLCLOCK_SECONDS=120 \
VAL_LOSS_EVERY=0 \
VAL_BATCH_SIZE=524288 \
TRAIN_BATCH_TOKENS=524288 \
TRAIN_LOG_EVERY=10 \
WARMUP_STEPS=1 \
BITNET_GROUP_SIZE=64 \
BITNET_SCALE_LAYOUT=runtime_row \
FP_STORAGE=0 \
MODEL_DIM=1024 \
NUM_LAYERS=9 \
NUM_HEADS=16 \
NUM_KV_HEADS=4 \
MLP_MULT=2 \
TRAINING_DEPTH_RECURRENCE=1 \
EVAL_DEPTH_RECURRENCE=1 \
ROPE_TYPE=yarn \
SEQ_SCHEDULE_FRACTION=0.33 \
BATCH_SCHEDULE_FRACTION=0.33 \
MATRIX_LR=0.04 \
SCALAR_LR=0.04 \
TIED_EMBED_LR=0.05 \
HEAD_LR=0.008 \
MUON_BACKEND_STEPS=5 \
LOGIT_SOFTCAP=30 \
QK_GAIN_INIT=1.5 \
python train_gpt.py
```

Observed H100 result:

```text
params:63480976 L:9 d:1024 h:16 kv:4 ws:1 ga:8 s:1337
step:140/10000 val_loss:3.7287 val_bpb:2.2446 train_time:125734ms zero_frac:0.343
artifact:14.09MB ternary:61341696(12878511B) fp:1090704(2181408B) code:72921
budget:14159617/16000000 (14.16/16.00MB) FITS
final_ternary_roundtrip val_loss:3.7288 val_bpb:2.2447
```

### Export To Runtime Packed Bundle

After training, export the `.ternary.ptz` artifact into the local runtime-row
BitNet bundle format and verify exact reconstruction.

```bash
cd /data/transformer_10

python tests/bench_parameter_golf_bitnet_export.py \
  --pg-script other_repos/parameter-golf/train_gpt.py \
  --artifact other_repos/parameter-golf/final_model.ternary.ptz \
  --export-packed artifacts/parameter_golf/final_model.runtime_bitnet.pt \
  --verify-export \
  --summary-json
```

Then benchmark selected tensors through the runtime:

```bash
python tests/bench_parameter_golf_bitnet_export.py \
  --pg-script other_repos/parameter-golf/train_gpt.py \
  --artifact other_repos/parameter-golf/final_model.ternary.ptz \
  --modules blocks.0.attn.c_qkv.weight,blocks.0.attn.proj.weight,blocks.0.mlp.fc.weight,blocks.0.mlp.proj.weight \
  --rows 1,16,4096,65536 \
  --backend auto \
  --warmup 10 \
  --iters 50
```

Standalone packed ternary matmul is not expected to beat H100 BF16 GEMM on the
small 512/1024-wide Parameter Golf layers. Treat that benchmark as a routing
diagnostic. The actual runtime target is fused packed execution and end-to-end
decode at 1024/2048/4096 context.
