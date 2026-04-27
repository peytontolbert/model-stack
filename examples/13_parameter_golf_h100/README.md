## Parameter Golf H100 Ternary Runtime Recipe

This example records the H100 Parameter Golf setup used for runtime-row ternary
training and export experiments. The Parameter Golf source lives at
`other_repos/parameter-golf`.

This recipe originally used another Parameter Golf pull as its base training
script:
`other_repos/parameter-golf/records/track_non_record_16mb/2026-04-25_ModelStack_RuntimeRow_BitNet_65M_SP1024_CTX4096/train_gpt.py`.
That source script identifies itself as:

```text
Ternary training script for OpenAI's Parameter Golf Challenge. Ciprian-Florin Ifrim - 24 March 2026
```

The local H100 work changed that base into a Model Stack runtime-row BitNet
experiment. The important local changes are:

- switched the target recipe to the restartable 65M-style
  `runtime_row_1024x7_relu2_mlp3` H100 preset;
- made runtime-row ternary scaling the default artifact/runtime layout for this
  example;
- added Model Stack trainable BitNet QAT integration and export of packed
  `QuantizedLinearBitNet` bundles;
- added H100 gates and microbenchmarks for packed BitNet forward, W2A8
  trainable forward, backward candidates, attention variants, and full training
  blocks;
- added launcher defaults for MLP-gated dynamic-int8 training, batched Muon,
  Parallel Muon reduce-scatter/all-gather exchange, MuonEq-R row normalization,
  and fused Adam/AdamW parameter group handling.

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
MODEL_STACK_BITNET_QAT=1
MODEL_STACK_TRAINABLE_BITNET_TRAINING_FORWARD=dynamic_int8_ste
MODEL_STACK_TRAINABLE_BITNET_SHAPE_GATE=pg_h100_mlp
MODEL_STACK_TRAINABLE_BITNET_BACKWARD_GRAD_INPUT=dynamic_int8_explicit_scale
MODEL_STACK_TRAINABLE_BITNET_BACKWARD_GRAD_WEIGHT=dynamic_int8_transpose
MODEL_STACK_INT8_QUANT_WARP_ROWS_PER_BLOCK=4
MODEL_STACK_INT8_COLUMN_QUANT_THREADS_X=32
MODEL_STACK_INT8_COLUMN_QUANT_ROWS_PER_BLOCK=128
MODEL_STACK_MUON_BATCHED=1
MODEL_STACK_MUON_BATCHED_MIN_BUCKET=2
MODEL_STACK_MUON_COMPILE=1
MODEL_STACK_MUON_ROW_NORM=1
MODEL_STACK_MUON_PARALLEL=1
MODEL_STACK_MUON_DISTRIBUTED_EXCHANGE=all_gather
MODEL_STACK_MUON_DISTRIBUTED_SHARDING=shape_bucket
SLIDING_EVAL=1
SLIDING_EVAL_STRIDE=64
```

### Model Stack QAT Smoke

The active root `other_repos/parameter-golf/train_gpt.py` can use Model Stack
trainable BitNet modules instead of plain `CastedLinear` layers. The
`run_pg_8xh100.sh` launcher now defaults to that Model Stack QAT path; set
`MODEL_STACK_BITNET_QAT=0` only when intentionally measuring the non-Model-Stack
baseline.

```bash
PG_DIR=/root/parameter_golf \
MODEL_STACK_ROOT=/root/transformer_10_h100 \
MAX_WALLCLOCK_SECONDS=60 \
PG_PRESET=runtime_row_512 \
bash examples/13_parameter_golf_h100/run_pg_8xh100.sh
```

With the default enabled path, training uses
`compress.quantization.TrainableBitNetLinear`, writes
`final_model.model_stack_bitnet.pt` with packed `QuantizedLinearBitNet` state,
and applies the measured dynamic-int8 training gate to the Parameter Golf MLP
projections.
`MODEL_STACK_TRAINABLE_BITNET_BACKWARD_GRAD_INPUT=dynamic_int8_explicit_scale`
plus
`MODEL_STACK_TRAINABLE_BITNET_BACKWARD_GRAD_WEIGHT=dynamic_int8_transpose`
is the current local SM80-family speed candidate. It keeps the dynamic-int8 path
shape-gated to the MLP projections and uses native int8 kernels in both
backward matrix products. These backward paths are approximate, so short
training-quality checks are still required before treating a full run as
record-ready.
The native pre-scale grad-input path is faster in the isolated local backward
microbenchmark, but the composed compiled block still favored explicit scaling
on the latest RTX 3090 run:

```text
rows=65536 mlp_up grad-input:
  dense 5.73 ms
  dynamic_int8_explicit_scale 4.57 ms, 1.25x
  dynamic_int8_pre_scale 4.55 ms, 1.26x

rows=65536 mlp_up grad-weight:
  dense 5.81 ms
  native composed int8 transpose 4.24 ms, 1.37x
  MODEL_STACK_INT8_COLUMN_QUANT_THREADS_X=32 was the best local amax layout.

compiled PG block d1024 h16 kv4 mlp3, batch=16, seq=4096:
  pg_h100_expansion + explicit_scale + int8 grad_weight:
    87.06 ms vs dense 91.61 ms, 1.052x
  pg_h100_mlp + explicit_scale + int8 grad_weight:
    86.36 ms vs dense 91.04 ms, 1.054x

compiled relu2 MLP pair d1024->3072->1024, rows=65536:
  pg_h100_mlp + explicit_scale + int8 grad_weight:
    31.31 ms vs dense 36.94 ms, 1.18x in the full compiled component harness
```

Muon optimizer work is now also measured separately. Single-rank seven-block
synthetic optimizer timing improved from `39.77 ms` for the old flat allocation
path to `33.23 ms` for shape-bucketed batched Muon, about `1.20x`. With
MuonEq-R row normalization enabled, the same local seven-block optimizer bench
measured `32.05 ms` for the batched path versus `38.33 ms` for the old flat
allocation path; disabling row norm measured `31.10 ms` versus `37.07 ms`.
The default
`MODEL_STACK_MUON_BATCHED_MIN_BUCKET=2` keeps small same-shape distributed
chunks eligible for the batched Newton-Schulz kernel without hurting the local
single-rank benchmark. The distributed Muon path now defaults to PR #399-style
`MODEL_STACK_MUON_PARALLEL=1`: DDP gradient sync is disabled during measured
training, matrix gradients are owner-packed and async reduce-scattered, Adam-side
replicated parameters are all-reduced and stepped while the matrix reduce-scatter
is in flight, then owner-side batched Newton-Schulz updates are all-gathered.
Set `MODEL_STACK_MUON_PARALLEL=0` to restore the DDP-first path. In that fallback
path, `MODEL_STACK_MUON_DISTRIBUTED_EXCHANGE=all_gather` packs only the update
shard owned by each rank and all-gathers those shards instead of zero-filling a
full update vector and all-reducing it; set it to `all_reduce` to restore the old
exchange.
`MODEL_STACK_MUON_DISTRIBUTED_SHARDING=shape_bucket` assigns same-shape matrix
chunks to ranks before the exchange, so the batched Newton-Schulz kernel can
still fire on 8-way distributed runs; set it to `index` to restore the simple
`parameter_index % world_size` ownership rule. With the 65M PG matrix shapes and
8 ranks, this produced an estimated owner load ratio of about `1.14x` max/min
instead of `3.0x` for 4-matrix chunks. The Newton-Schulz polynomial uses
`addmm`/`baddbmm` fused scale-add matmuls, which measured about `31.01 ms` for
the local seven-block synthetic Muon step versus `33.46 ms` before that cleanup
and `40.03 ms` for the old flat path.
Adam and AdamW parameter groups already use PyTorch `fused=True`; the record
script also now keeps the AdamW matrix LR/momentum scheduling separate from
Muon.
`MODEL_STACK_ATTENTION_REPEAT_KV=repeat` or `expand_reshape` is also available
as an attention-core experiment: it materializes GQA KV heads before SDPA instead
of using PyTorch `enable_gqa=True`. Local attention-only benchmarks sometimes
improve, but the full block has not kept a stable win, so this remains opt-in.

### BitNet Kernel Benchmark Gate

Do not treat Model Stack BitNet as an end-to-end speed win until this benchmark
shows the selected runtime path beating the dense dequantized baseline for the
active Parameter Golf shapes. The raw H100 int8 accumulation kernel now wins the
large prefill microbenchmark, but the dynamic activation-quantization and
rescale/output epilogue still prevent the direct-from-float path from winning.
The QAT training module defaults to a float shadow weight and PyTorch
`F.linear` for forward/backward. The opt-in
`MODEL_STACK_TRAINABLE_BITNET_TRAINING_FORWARD=dynamic_int8_ste` path routes
the forward through Model Stack's native runtime-row quantizer and W2A8
from-float kernel, but it must still beat the compiled dense STE training path
before it can be treated as a Parameter Golf training-speed win.

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

### Training Step Kernel Gate

Runtime packed forward wins do not automatically imply faster training, because
backward, attention, optimizer, and DDP still dominate the full step. Use this
gate to measure the Model Stack trainable BitNet layer as a training primitive:
The QAT path also routes runtime-row weight quantization through the native
`bitnet_runtime_row_quantize` CUDA helper when the extension is rebuilt, so the
gate measures the fused forward path without Python/Torch row-reduction overhead
masking it.

```bash
cd /data/transformer_10

MODEL_STACK_ENABLE_INT8_LINEAR_CUTLASS_FUSED=1 \
python examples/13_parameter_golf_h100/bench_pg_bitnet_training_step.py \
  --device cuda:0 \
  --preset runtime_row_1024x7_swiglu_mlp2 \
  --rows 65536 \
  --warmup 10 \
  --iters 50 \
  --repeats 3 \
  --include-int8-grad-weight \
  --jsonl
```

The benchmark compares the default `dense_ste` training path with the opt-in
`MODEL_STACK_TRAINABLE_BITNET_TRAINING_FORWARD=dynamic_int8_ste` path. Treat a
result as actionable only when `train_step_speedup_vs_dense_ste` improves, not
just `forward_speedup_vs_dense_ste`. Also watch `weight_quantize_ms`; after
rebuilding the extension it should reflect the native runtime-row quantizer, not
the old Python/Torch reduction path.

Parameter Golf compiles the model forward with `torch.compile(...,
fullgraph=True)`, so also run the compile-aware gate:

```bash
MODEL_STACK_ENABLE_INT8_LINEAR_CUTLASS_FUSED=1 \
python examples/13_parameter_golf_h100/bench_pg_bitnet_training_step.py \
  --device cuda:0 \
  --preset runtime_row_1024x7_relu2_mlp3 \
  --rows 65536 \
  --compile-module \
  --warmup 5 \
  --iters 20 \
  --repeats 2 \
  --include-relu2-mlp-pair \
  --jsonl
```

The compiled dynamic-int8 STE path is guarded by
`MODEL_STACK_TRAINABLE_BITNET_COMPILED_INT8_STE=1`. The benchmark sets that
flag only for the explicit dynamic-int8 variant and resets the Dynamo cache
before compiling each variant, because this gate compares env-selected training
graphs in one process.

To measure the actual 65M PG block geometry, include attention and the ReLU2 MLP
in one compiled block:

```bash
MODEL_STACK_ENABLE_INT8_LINEAR_CUTLASS_FUSED=1 \
MODEL_STACK_INT8_QUANT_WARP_ROWS_PER_BLOCK=4 \
MODEL_STACK_TRAINABLE_BITNET_SHAPE_GATE=pg_h100_mlp \
python examples/13_parameter_golf_h100/bench_pg_bitnet_training_step.py \
  --device cuda:0 \
  --shape skip:64:64 \
  --rows 1 \
  --dtype bf16 \
  --compile-module \
  --warmup 10 \
  --iters 20 \
  --repeats 3 \
  --include-pg-block \
  --block-batch-size 16 \
  --block-seq-len 4096 \
  --block-dim 1024 \
  --block-num-heads 16 \
  --block-num-kv-heads 4 \
  --block-mlp-mult 2 \
  --jsonl
```

`MODEL_STACK_TRAINABLE_BITNET_SHAPE_GATE=pg_h100_mlp` enables the ReLU2 MLP up
and down projections while keeping attention projections on dense STE. The
safer `pg_h100_expansion` gate remains available if a quality smoke says the
approximate down-projection backward is too noisy. The MLP gate is launch-safe
for float32 down-projection activations; the rowwise shared-cache guard reserves
static shared-memory overhead and falls back to the non-cached wide quantizer
instead of launching over the default shared memory limit.

For training-side bottlenecks, use the component profiler before changing
defaults:

```bash
MODEL_STACK_ENABLE_INT8_LINEAR_CUTLASS_FUSED=1 \
MODEL_STACK_TRAINABLE_BITNET_SHAPE_GATE=pg_h100_mlp \
MODEL_STACK_INT8_QUANT_WARP_ROWS_PER_BLOCK=4 \
conda run -n ai python examples/13_parameter_golf_h100/bench_pg_training_components.py \
  --dtype bf16 \
  --compile-module \
  --batch-size 8 \
  --seq-len 4096 \
  --dim 1024 \
  --num-heads 16 \
  --num-kv-heads 4 \
  --mlp-mult 2 \
  --jsonl
```

On the local RTX 3090 at `batch=8, seq=4096, dim=1024, mlp_mult=2`, this showed
attention training around 25 ms, MLP training around 12 ms, block training around
40-42 ms, and a Muon-like matrix optimizer step around 5 ms for one block. That
keeps the H100 priority on backward/optimizer/block-level gates rather than
forward-only GEMM wins.

The training loader now keeps token distribution rank-local: each rank skips
other ranks' spans in the shared token stream, copies one uint16 token span to
GPU, then forms `x` and `y` as shifted int64 GPU views. This preserves the old
rank slicing contract and avoids materializing a full world-size chunk plus two
separate host-to-device copies. On the local RTX 3090 with the real FineWeb
SP1024 shards and an 8-rank-style batch, this path measured about `0.35 ms`
versus `1.00 ms` for the previous active loader logic. It is useful cleanup, but
not large enough to explain 100 ms-class training steps.

Current local SM80-family candidate:

```bash
MODEL_STACK_TRAINABLE_BITNET_SHAPE_GATE=pg_h100_mlp \
MODEL_STACK_TRAINABLE_BITNET_BACKWARD_GRAD_INPUT=dynamic_int8_explicit_scale \
MODEL_STACK_TRAINABLE_BITNET_BACKWARD_GRAD_WEIGHT=dynamic_int8_transpose \
MODEL_STACK_INT8_QUANT_WARP_ROWS_PER_BLOCK=4 \
MODEL_STACK_INT8_COLUMN_QUANT_THREADS_X=32 \
MODEL_STACK_INT8_COLUMN_QUANT_ROWS_PER_BLOCK=128 \
conda run -n ai python examples/13_parameter_golf_h100/bench_pg_bitnet_training_step.py \
  --device cuda:0 \
  --shape skip:64:64 \
  --rows 1 \
  --dtype fp16 \
  --compile-module \
  --warmup 3 \
  --iters 5 \
  --repeats 3 \
  --include-pg-block \
  --block-batch-size 16 \
  --block-seq-len 4096 \
  --block-dim 1024 \
  --block-num-heads 16 \
  --block-num-kv-heads 4 \
  --block-mlp-mult 3 \
  --jsonl
```

On the RTX 3090 this candidate measured about `1.05x` full PG block train-step
speedup for `MLP_MULT=3` at both 32k and 65k token rows. The safer variant with
only `MODEL_STACK_TRAINABLE_BITNET_BACKWARD_GRAD_INPUT=dynamic_int8_explicit_scale`
keeps `grad_weight` exact and measured about `1.03x` on the 65k-row MLP3 block.
The faster variant has larger gradient deviation (`param_grad_max_abs_err` around
`1e3` in the synthetic block benchmark), so it is a speed candidate, not yet a
training-quality proof.

To isolate GQA attention-core variants:

```bash
conda run -n ai python examples/13_parameter_golf_h100/bench_pg_attention_variants.py \
  --dtype bf16 \
  --compile-module \
  --batch-size 8 \
  --seq-len 4096 \
  --num-heads 16 \
  --num-kv-heads 4 \
  --head-dim 64 \
  --jsonl
```

If `repeat` or `expand_reshape` wins this isolated benchmark, rerun the component
profiler with `MODEL_STACK_ATTENTION_REPEAT_KV` set before changing training
defaults; the extra KV materialization can erase the attention-only gain inside
the full PG block.

Observed on single H100 after switching the compiled trainable path to the
direct autograd Function over the compile-safe runtime-row quantizer and W2A8
forward:

```text
eager ReLU2 MLP pair:
  mlp_up train_step_speedup_vs_dense_ste: ~1.11x
  mlp_down train_step_speedup_vs_dense_ste: ~1.04x

compiled ReLU2 MLP pair:
  current with input gradients:
    mlp_up train_step_speedup_vs_dense_ste: ~1.08x
    mlp_down train_step_speedup_vs_dense_ste: ~1.02x
  current no-input-grad upper bound:
    mlp_up train_step_speedup_vs_dense_ste: ~1.09x
    mlp_down train_step_speedup_vs_dense_ste: ~1.03x

compiled ReLU2 MLP pair as one module (`1024 -> 3072 -> 1024`):
  no-input-grad train_step_speedup_vs_dense_ste: ~1.04x
  with input gradients train_step_speedup_vs_dense_ste: ~1.04x

compiled SwiGLU MLP pair:
  swiglu_gate_up train_step_speedup_vs_dense_ste: ~0.95x
  swiglu_down train_step_speedup_vs_dense_ste: ~0.98x

compiled full PG block (`batch=16, seq=4096, dim=1024, heads=16, kv_heads=4, mlp_mult=2`):
  all dynamic-int8 linears: slower, ~0.91-0.92x train_step_speedup_vs_dense_ste
  fused dynamic-int8 QKV prototype: still slower, ~0.92x
  expansion-only gate + 4-row quant tuning: ~1.006-1.012x train_step_speedup_vs_dense_ste
  expansion-only gate + 4-row quant tuning: ~1.02-1.03x forward_speedup_vs_dense_ste

The ReLU2 MLP pair is now a compiled training-primitive win. It is still not an
end-to-end Parameter Golf proof: attention, optimizer, DDP, and validation
quality still need measurement before spending an 8xH100 full run.
```

This means the current Model Stack W2A8 forward kernel is a runtime/prepacked
kernel win, an eager trainable-layer win on selected MLP shapes, a compiled
ReLU2 MLP training-primitive win, and a small compiled block-level win only when
the H100 shape gate is enabled. A quick H100 prototype that reused the existing
W2A8 forward kernel for approximate `grad_input` was slower than the compiled
BF16 backward GEMM (`~0.50-0.87x` depending on shape), so larger training
speedups still need better backward strategy or graph-level reuse, not just more
forward dispatch wiring.

Use the backward component benchmark before building another training kernel:

```bash
MODEL_STACK_ENABLE_INT8_LINEAR_CUTLASS_FUSED=1 \
python examples/13_parameter_golf_h100/bench_pg_bitnet_backward.py \
  --device cuda:0 \
  --preset runtime_row_1024x7_relu2_mlp3 \
  --rows 65536 \
  --warmup 10 \
  --iters 50 \
  --repeats 3 \
  --include-int8-grad-weight \
  --include-int8-grad-weight-upper-bound \
  --jsonl
```

The current negative results on single H100:

```text
existing W2A8-from-float reused for grad_input:
  relu2_up: ~0.50x dense
  swiglu_gate_up: ~0.58x dense
  relu2_down: ~0.87x dense
  swiglu_down: ~0.78x dense

existing packed BitNet forward reused for grad_input:
  ~0.004x dense, not viable for training backward

CUTLASS 55_hopper_int4_bf16_gemm as a candidate grad_input kernel:
  direct convert is roughly tied with dense before row-scale handling
  per-column scale mode is slower than dense on all PG backward shapes tested

W8A8 grad_weight prototype:
  prequantized int8 GEMM can beat dense grad_weight, but preparing the
  transposed int8 operands dominates. On rows=65536, transposing and quantizing
  a 1024-wide activation costs ~0.77 ms before the int8 GEMM starts; 3072-wide
  costs ~2.33 ms and 4096-wide costs ~3.11 ms. Full W8A8 grad_weight is only
  ~0.13-0.16x dense in the tested shapes.

fused columnwise transpose-quantize prototype:
  `int8_quantize_activation_transpose_forward` writes the transpose-friendly
  int8 layout directly and returns one scale per original column. This matches
  the TransformerEngine-style rowwise/columnwise representation pattern and is
  much faster than `x.t().contiguous()` followed by rowwise quantization:
    1024-wide: ~0.19 ms vs ~0.78 ms
    2048-wide: ~0.36 ms vs ~1.56 ms
    3072-wide: ~0.54 ms vs ~2.33 ms
    4096-wide: ~0.72 ms vs ~3.10 ms
  Full W8A8 grad_weight improves from ~2.66-4.23 ms to ~0.87-1.28 ms, but
  still loses to dense BF16 grad_weight on the PG shapes. A delayed-scale upper
  bound, where the column scales are already known and the kernel only
  quantizes into transposed layout, improves further to ~0.68-0.95 ms, but
  still only reaches ~0.56-0.81x dense. This makes W8A8 grad_weight a useful
  diagnostic but not the current winning backward path.

raw original-rowwise int8 grad_weight upper bound:
  quantizing the original row-major `x` and `grad_out` is cheap
  (`mlp_up`: x ~0.076 ms, grad_out ~0.224 ms; `swiglu_gate_up`: x ~0.076 ms,
  grad_out ~0.294 ms), but `torch._int_mm(qgrad_out.t(), qx)` on the strided
  transpose is much slower than dense (`mlp_up`: ~4.33 ms vs dense ~0.54 ms;
  `swiglu_gate_up`: ~4.34 ms vs dense ~0.78 ms). Making the lhs contiguous is
  slower still because it reintroduces the transpose copy. This path is also
  not scale-correct: original row-wise activation scales multiply inside the
  reduction, so a single output row/column rescale cannot recover dense
  grad_weight. A correct winning grad_weight kernel would need to fuse or
  avoid the transposed column-quantization/layout step, not just call a
  strided int8 matmul.

compiled runtime-row quantization:
  the native `bitnet_runtime_row_quantize` custom op is faster than a pure
  Torch compiled row quantizer for the tested dense STE forward shapes
  (`mlp_up` ~0.574 ms vs ~0.605 ms, `swiglu_gate_up` ~0.766 ms vs ~0.848 ms),
  so the custom op is not the source of the compiled dense baseline win.
```
