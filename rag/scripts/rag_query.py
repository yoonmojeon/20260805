"""
Query RAG index: single-document or unified multi-doc collection.

Examples:
  python scripts/rag_query.py --doc-id kr_1_2025 --query "902절 탈급 절차는?"
  python scripts/rag_query.py --doc-id kr_1_2025 -i --full-text --top-k 3
  python scripts/rag_query.py --unified pilot_100 --source KR --query "탈급 절차"
  python scripts/rag_query.py --unified pilot_100 -i --full-text --top-k 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from embedding_policy import DEFAULT_EMBEDDING_PRESET, embed_texts_local, resolve_embedding_config
from rag_eval_lib import load_chunk_text_map
from retrieval_search import enrich_query_for_embedding, query_with_hybrid_ranking


def load_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Index manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_index_paths(
    *,
    index_dir: Path,
    doc_id: str | None,
    unified_id: str | None,
) -> tuple[Path, dict]:
    if unified_id:
        root = index_dir / f"unified_{unified_id}"
    elif doc_id:
        root = index_dir / doc_id
    else:
        raise ValueError("Provide --doc-id or --unified")
    manifest = load_manifest(root / "index_manifest.json")
    return root, manifest


def chunk_body(
    chunk_id: str,
    doc: str | None,
    chunk_texts: dict[str, str] | None,
    *,
    full_text: bool,
    preview_chars: int,
) -> str:
    text = (chunk_texts or {}).get(chunk_id) or doc or ""
    if full_text:
        if len(text) > preview_chars:
            return text[:preview_chars] + "\n... (truncated)"
        return text
    return text.replace("\n", " ")[:200]


def query_collection(
    collection,
    query: str,
    model_name: str,
    top_k: int,
    *,
    chunk_texts: dict[str, str] | None = None,
    full_text: bool = False,
    preview_chars: int = 2500,
    source: str | None = None,
    filter_doc_id: str | None = None,
) -> None:
    embed_query = enrich_query_for_embedding(query, model_name)
    vector = embed_texts_local([embed_query], model_name, for_query=True)[0]
    results = query_with_hybrid_ranking(
        collection,
        query,
        vector,
        top_k=top_k,
        source=source,
        doc_id=filter_doc_id,
    )

    ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    documents = results["documents"][0]
    hints = results.get("clause_hints") or []

    filter_note = ""
    if source:
        filter_note += f" source={source}"
    if filter_doc_id:
        filter_note += f" doc={filter_doc_id}"
    print(f"\nQ: {query}{filter_note}")
    if hints:
        print(f"  (clause hints: {', '.join(hints)})")

    for rank, (chunk_id, distance, meta, doc) in enumerate(
        zip(ids, distances, metadatas, documents), start=1
    ):
        meta = meta or {}
        clause = meta.get("clause_number") or meta.get("article_number") or ""
        clause_part = f" clause={clause}" if clause else ""
        src = meta.get("source", "")
        did = meta.get("doc_id", "")
        etype = meta.get("element_type", "")
        emode = meta.get("embedding_mode", "")
        print(
            f"  {rank}. {chunk_id} | {src}/{did} | p{meta.get('page_number')}"
            f"{clause_part} | {etype}/{emode} | dist={distance:.4f}"
        )
        body = chunk_body(
            chunk_id, doc, chunk_texts, full_text=full_text, preview_chars=preview_chars
        )
        if full_text:
            print("     ---")
            for line in body.splitlines():
                print(f"     {line}")
            print("     ---")
        else:
            print(f"     {body}")


def run_interactive(
    collection,
    model_name: str,
    top_k: int,
    *,
    chunk_texts: dict[str, str] | None,
    full_text: bool,
    preview_chars: int,
    source: str | None,
    filter_doc_id: str | None,
    unified_id: str | None,
) -> None:
    scope = f"unified={unified_id}" if unified_id else "single-doc"
    print(f"대화형 검색 ({scope}) | top_k={top_k} | full_text={full_text}")
    if source:
        print(f"  source filter: {source}")
    if filter_doc_id:
        print(f"  doc filter: {filter_doc_id}")
    print("종료: quit / exit / q / 빈 Enter")
    while True:
        try:
            query = input("\n질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break
        if not query or query.lower() in {"quit", "exit", "q"}:
            print("종료합니다.")
            break
        query_collection(
            collection,
            query,
            model_name,
            top_k,
            chunk_texts=chunk_texts,
            full_text=full_text,
            preview_chars=preview_chars,
            source=source,
            filter_doc_id=filter_doc_id,
        )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Query single-doc or unified RAG index.")
    parser.add_argument("--doc-id", type=str, default=None, help="Single-document index")
    parser.add_argument(
        "--unified",
        type=str,
        default=None,
        metavar="COLLECTION_ID",
        help="Unified index id (e.g. pilot_100)",
    )
    parser.add_argument("--source", type=str, default=None, help="Filter: KR, DNV, ABS, LR, MSC, MEPC")
    parser.add_argument("--filter-doc-id", type=str, default=None, help="Filter to one doc_id")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("-i", "--interactive", action="store_true")
    parser.add_argument("--questions", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--full-text", action="store_true")
    parser.add_argument("--preview-chars", type=int, default=2500)
    parser.add_argument("--index-dir", type=Path, default=Path("data/processed/index"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    parser.add_argument("--embedding-preset", type=str, default=None)
    args = parser.parse_args()

    if bool(args.doc_id) == bool(args.unified):
        parser.error("Provide exactly one of: --doc-id OR --unified")

    if sum(bool(x) for x in (args.query, args.interactive, args.questions)) > 1:
        parser.error("Use only one of: --query, -i/--interactive, --questions")

    _, manifest = resolve_index_paths(
        index_dir=args.index_dir,
        doc_id=args.doc_id,
        unified_id=args.unified,
    )
    preset = args.embedding_preset or manifest.get("embedding_preset", DEFAULT_EMBEDDING_PRESET)
    embed_config = resolve_embedding_config(preset, str(manifest.get("embedding_model", "")))
    model_name = str(embed_config["model"])

    chunk_texts = None
    if args.full_text and args.doc_id and not args.unified:
        chunks_path = args.chunks_dir / args.doc_id / "chunks.jsonl"
        if chunks_path.exists():
            chunk_texts = load_chunk_text_map(chunks_path)

    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=manifest["chroma_path"],
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(manifest["collection_name"])

    common = {
        "chunk_texts": chunk_texts,
        "full_text": args.full_text,
        "preview_chars": args.preview_chars,
        "source": args.source.upper() if args.source else None,
        "filter_doc_id": args.filter_doc_id,
    }

    if args.interactive:
        run_interactive(
            collection,
            model_name,
            args.top_k,
            unified_id=args.unified,
            **common,
        )
    elif args.questions:
        with args.questions.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                q_source = row.get("gold_source") or args.source
                q_doc = row.get("gold_doc_id") or args.filter_doc_id
                query_collection(
                    collection,
                    str(row["question"]),
                    model_name,
                    args.top_k,
                    source=str(q_source).upper() if q_source else common["source"],
                    filter_doc_id=str(q_doc) if q_doc else common["filter_doc_id"],
                    chunk_texts=chunk_texts,
                    full_text=args.full_text,
                    preview_chars=args.preview_chars,
                )
    elif args.query:
        query_collection(collection, args.query, model_name, args.top_k, **common)
    else:
        parser.error("Provide --query, -i/--interactive, or --questions")


if __name__ == "__main__":
    main()
