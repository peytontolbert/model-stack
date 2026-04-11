# transformer_10 Data, Checkpoint, And Tokenizer C++ Spec

This document defines the target C++ ownership for `data/`, `model/checkpoint.py`, `dist/checkpoint.py`, `tensor/io_safetensors.py`, `tensor/io_utils.py`, `model/hf_llama_loader.py`, and `model/hf_snapshot.py`.

## 1. Scope

The runtime must own:

- tokenizer loading and encode/decode
- dataset and shard iteration
- batch and packed-batch descriptors
- pinned-memory and async H2D staging
- safetensors and sharded checkpoint I/O
- HF snapshot import and local weight binding

## 2. Tokenizer API

## `t10::data::Tokenizer`

Required interface:

- `encode(std::string_view text) -> std::vector<int32_t>`
- `decode(span<const int32_t> ids, DecodeOptions opts) -> std::string`
- `vocab_size()`
- `special_tokens()`
- `serialize_info()`

Required implementations:

- `SentencePieceTokenizer`
  - for `tokenizer.model`
- `TokenizerJsonUnigramTokenizer`
  - for `tokenizer.json`
- `WhitespaceTokenizer`
  - testing/reference only
- `HfTokenizerAdapter`
  - Python binding fallback during transition only

Required metadata:

- BOS/EOS/UNK/PAD ids
- special token strings
- normalization policy
- byte-level or metaspace policy

Mapping from current code:

- `data/tokenizer.py::LocalLlamaTokenizer` -> `SentencePieceTokenizer` or `TokenizerJsonUnigramTokenizer`
- `data/tokenizer.py::PureLlamaTokenizer` -> `TokenizerJsonUnigramTokenizer` reference for exactness
- `data/tokenizer.py::HFTokenizer` -> Python compatibility adapter only

## 3. Batch And Dataset API

## `t10::data::Batch`

Required fields:

- `input_ids`
- `attn_mask` or compressed mask descriptor
- optional `lengths`
- optional `position_ids`
- optional `labels`
- optional `packed_offsets`
- optional `segment_ids`

## `t10::data::Dataset`

Required dataset types:

- `MapDataset`
  - indexed non-streaming dataset from token shards
- `StreamingDataset`
  - shard-streaming iterable dataset
- `DistributedDatasetView`
  - rank-aware shard slicing

## `t10::data::ShardReader`

Supported sources:

- `.bin`
- `.npy`
- `.txt`
- later optional memory-mapped packed binary formats

Requirements:

- deterministic shard discovery
- zero-copy or memory-mapped reading where possible
- no Python DataLoader dependency on the hot path

## 4. Loader And H2D Staging

## `t10::data::DataLoader`

Responsibilities:

- host-side batch collation
- pinned host allocation
- asynchronous H2D staging
- prefetch depth control
- device-targeted batch materialization

## `t10::data::DistributedDataLoader`

Responsibilities:

- rank/world shard partitioning
- deterministic shuffle seeds
- epoch reset hooks
- integration with `t10::dist::RankTopology`

Important rule:

- the data runtime owns batch layout decisions that CUDA kernels expect
- packed and ragged descriptors are part of the runtime contract, not an afterthought

## 5. Checkpoint API

## `t10::checkpoint::CheckpointReader`

Required functions:

- load model config
- enumerate tensors without materializing all data
- load selected tensors by key
- load shard metadata
- validate dtype and shape

## `t10::checkpoint::CheckpointWriter`

Required functions:

- save config and runtime metadata
- write safetensors or sharded tensor files
- persist compression deltas and optimizer state when needed

## `t10::checkpoint::WeightBinder`

Responsibilities:

- map checkpoint tensor names to internal parameter storage
- transpose or reshape where required exactly once during import
- apply quant or LoRA metadata if requested

## 6. Safetensors Support

Replace `tensor/io_safetensors.py` with:

- `SafeTensorReader`
- `SafeTensorWriter`
- `StableTensorHash`

Requirements:

- CPU-path metadata parse
- staged H2D load
- optional selective tensor materialization
- checksum hooks for governance and registry metadata

## 7. HF Import Path

Required import runtime:

- `HfSnapshotResolver`
- `HfLlamaImporter`
- `SnapshotCache`

Responsibilities:

- locate snapshot files
- read config/tokenizer/weights
- map HF naming to internal parameter layout
- stack or split fused gate/up projections as needed
- preserve exact model metadata for parity testing

This replaces the logic scattered across:

- `model/hf_llama_loader.py`
- `model/hf_snapshot.py`
- `model/llama_bootstrap.py`

## 8. Distributed Checkpointing

Required:

- global checkpoint metadata
- shard manifest by rank and parameter partition
- restore under TP/DP/SP layouts different from save layout where supported
- rank-local streaming reads

This replaces `dist/checkpoint.py` as a true runtime subsystem.

## 9. File Mapping

| Current file | Target implementation |
|---|---|
| `data/batch.py` | `t10::data::Batch` |
| `data/loader.py` | `t10::data::MapDataset`, `DataLoader` |
| `data/iterable.py` | `t10::data::StreamingDataset` |
| `data/tokenizer.py` | `t10::data::TokenizerFactory` and tokenizer implementations |
| `model/checkpoint.py` | `t10::checkpoint::CheckpointReader/Writer` |
| `dist/checkpoint.py` | `t10::checkpoint::ShardedCheckpointCoordinator` |
| `tensor/io_safetensors.py` | `t10::checkpoint::SafeTensorReader/Writer` |
| `tensor/io_utils.py` | `t10::data::PinnedTransfer` |
| `model/hf_llama_loader.py` | `t10::checkpoint::HfLlamaImporter` |
| `model/hf_snapshot.py` | `t10::checkpoint::SnapshotCache` |

## 10. Migration Order

1. local config + safetensors read/write
2. local tokenizer load and encode/decode
3. map-style shard dataset and batch loader
4. streaming loader and pinned-memory prefetch
5. HF import path
6. distributed checkpointing and distributed data loading

## 11. Definition Of Coverage

Coverage is complete when:

- no model load path requires `torch.load`, HF Python internals, or Python-owned tensor rebinding for normal operation
- tokenizer choice is represented in C++ config and runtime types
- batch descriptors match kernel expectations directly
- checkpoint and data systems are explicit runtime-owned subsystems
