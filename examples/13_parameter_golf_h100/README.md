## Parameter Golf H100 Ternary Runtime Recipe

This example records the H100 Parameter Golf setup used for runtime-row ternary
training and export experiments. The Parameter Golf source lives at
`other_repos/parameter-golf`; the H100 copy used the same `train_gpt.py` and
`README.md` hashes as the local checkout.

Use the scripts in this directory for restartable H100 work:

```bash
# Build the local CUDA/native runtime for H100.
bash examples/13_parameter_golf_h100/build_runtime_h100.sh

# Launch competition-style 8xH100 Parameter Golf training.
PG_DIR=/root/parameter_golf \
PG_PRESET=runtime_row_1024 \
bash examples/13_parameter_golf_h100/run_pg_8xh100.sh

# Export a trained ternary artifact into the runtime-row BitNet bundle.
PG_DIR=/root/parameter_golf \
ARTIFACT=/root/parameter_golf/final_model.ternary.ptz \
OUT=/root/transformer_10_h100/artifacts/parameter_golf/final_model.runtime_bitnet.pt \
bash examples/13_parameter_golf_h100/export_runtime_bundle.sh
```

The `h100_remote_snapshot/` directory also preserves H100-side runtime files
that differed from local at the time of capture. Those are saved for recovery,
not merged, because the active local tree has newer BitNet decode changes.

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
