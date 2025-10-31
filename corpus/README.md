corpus: manifests, redaction, dedup, shard builder

Build shards from raw text/jsonl

```bash
python -m corpus.cli build \
  --input ./raw_corpus \
  --outdir ./corpus/shards \
  --tokenizer gpt2 \
  --shard-size-tokens 1048576 \
  --redact-pii --dedup
```

Outputs `.bin` shards (int32 token streams) and `manifest.json`:

```json
{
  "version": 1,
  "tokenizer": {"type": "hf", "name_or_path": "gpt2"},
  "total_tokens": 123456789,
  "shards": [{"path": ".../shard_000000.bin", "num_tokens": 1048576}]
}
```

Stats

```bash
python -m corpus.cli stats --manifest ./corpus/shards/manifest.json
```

PII & dedup

- Simple regex redaction for emails/phones/IPs
- Exact content-hash dedup baseline

Data curation & quality pipeline

Web/FS connectors, license & provenance tracking, dedup (MinHash/LSH), near-dup clustering, PII scrubbing, heuristic filters, weak labeling, corpus manifests.

Emits shard manifests consumed by data