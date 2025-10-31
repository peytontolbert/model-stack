RAG: Retrieval-Augmented Generation

This package provides a small, composable RAG stack:
- Config schemas in `config.py`
- Pluggable components in `components/` (splitter, embedder, store, retriever, reranker)
- A simple `pipeline.py` wiring components together
- A CLI in `cli.py` for quick indexing and query

Example:
```python
from rag.pipeline import RAGPipeline
from rag.config import RAGConfig

cfg = RAGConfig()
pipe = RAGPipeline.from_config(cfg)
pipe.index_texts(["hello world", "lorem ipsum dolor sit amet"], ids=["a","b"]) 
print(pipe.query("hello"))
```

