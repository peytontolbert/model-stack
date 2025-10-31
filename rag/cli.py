import argparse
from .config import RAGConfig
from .pipeline import RAGPipeline


def main():
    p = argparse.ArgumentParser("rag")
    sub = p.add_subparsers(dest="cmd", required=True)

    idx = sub.add_parser("index", help="index a newline-separated text file")
    idx.add_argument("--infile", required=True)

    q = sub.add_parser("query", help="query the index")
    q.add_argument("--query", required=True)
    q.add_argument("--k", type=int, default=4)

    args = p.parse_args()
    cfg = RAGConfig()
    pipe = RAGPipeline.from_config(cfg)

    if args.cmd == "index":
        with open(args.infile, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        pipe.index_texts(texts)
        print(f"Indexed {len(texts)} texts")
        return

    if args.cmd == "query":
        res = pipe.query(args.query, k=args.k)
        for i, (doc_id, score, text) in enumerate(res):
            print(f"{i+1}. id={doc_id} score={score:.4f} text={text[:80]}")


if __name__ == "__main__":
    main()


