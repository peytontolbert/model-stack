metrics, benchmarks, throughput/latency harness, regression dashboards

## Perplexity on token shards

```bash
python -m eval.cli ppl \
  --model-dir ckpts/my_model \
  --shards data/val_tokens \
  --batch-size 16 --seq-len 1024 --num-workers 2
```

Or pass a model factory:

```bash
python -m eval.cli ppl \
  --model mypkg.models:build_small_lm \
  --shards data/val_tokens
```

Outputs JSON with nll/ppl/acc/token counts.

## Throughput/latency benchmarks

Forward tokens/sec:
```bash
python -m eval.cli bench-forward \
  --model-dir ckpts/my_model \
  --batch-size 8 --seq-len 1024 --warmup 5 --steps 20 --outdir .eval
```

Decode throughput (new tokens/sec):
```bash
python -m eval.cli bench-generate \
  --model-dir ckpts/my_model \
  --batch-size 1 --seq-len 256 --max-new-tokens 256 --outdir .eval
```

## Calibration (ECE)

```bash
python -m eval.cli ece \
  --model-dir ckpts/my_model \
  --shards data/val_tokens --n-bins 15 --outdir .eval
```

All commands can optionally log to viz with `--viz-log-dir .viz`.

## Sequence metrics (BLEU, ROUGE-L, EM, token F1)

```bash
python -m eval.cli seq \
  --hyp outputs.txt \
  --ref references.txt \
  --outdir .eval --viz-log-dir .viz
```

`outputs.txt` and `references.txt` are one example per line.

## Latency percentiles

```bash
python -m eval.cli latency --model-dir ckpts/my_model --mode forward --repeats 200
python -m eval.cli latency --model-dir ckpts/my_model --mode generate --repeats 100 --max-new-tokens 1
```

## Memory footprint

```bash
python -m eval.cli mem --model-dir ckpts/my_model --peak --outdir .eval
```

## Suite (ppl, benches, ece)

```bash
python -m eval.cli suite \
  --model-dir ckpts/my_model \
  --shards data/val_tokens \
  --batch-size 16 --seq-len 1024 --outdir .eval
```
