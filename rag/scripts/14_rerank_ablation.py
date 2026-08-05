"""
Rerank parameter ablation on pilot validation questions.

  python scripts/14_rerank_ablation.py
  python scripts/14_rerank_ablation.py --top-k 8 --fetch-k 40
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from rag_answer_lib import RetrievedChunk, load_questions, load_unified_collection, retrieve_for_question
from rag_retrieval_metrics import compute_retrieval_metrics
from retrieval_diversity import DiversityConfig, diversity_rerank


@dataclass(frozen=True)
class AblationSetting:
    name: str
    label: str
    use_rerank: bool
    max_chunks_per_page: int = 1
    max_chunks_per_doc: int = 8
    enable_mmr: bool = True


SETTINGS: tuple[AblationSetting, ...] = (
    AblationSetting("A", "baseline", use_rerank=False),
    AblationSetting(
        "B",
        "rerank_page_only",
        use_rerank=True,
        max_chunks_per_page=1,
        max_chunks_per_doc=8,
        enable_mmr=True,
    ),
    AblationSetting(
        "C",
        "rerank_balanced",
        use_rerank=True,
        max_chunks_per_page=1,
        max_chunks_per_doc=5,
        enable_mmr=True,
    ),
    AblationSetting(
        "D",
        "rerank_strict",
        use_rerank=True,
        max_chunks_per_page=1,
        max_chunks_per_doc=3,
        enable_mmr=True,
    ),
)

METRIC_COLUMNS = (
    "source_hit_at_5",
    "gold_doc_hit_at_5",
    "gold_page_set_hit_at_5",
    "topic_hit_at_k",
    "keyword_coverage",
    "duplicate_doc_ratio",
    "duplicate_page_ratio",
    "ambiguous_count",
)


def _clone_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return copy.deepcopy(chunks)


def _apply_setting(
    pool: list[RetrievedChunk],
    *,
    row: dict,
    top_k: int,
    setting: AblationSetting,
) -> list[RetrievedChunk]:
    if not setting.use_rerank:
        return pool[:top_k]
    cfg = DiversityConfig(
        max_chunks_per_doc=setting.max_chunks_per_doc,
        max_chunks_per_page=setting.max_chunks_per_page,
        enable_mmr=setting.enable_mmr,
    )
    return diversity_rerank(
        _clone_chunks(pool),
        top_k=top_k,
        category=str(row.get("category", "")),
        config=cfg,
    )


def _aggregate_row(metrics_list: list[dict], *, eval_k: int, top_k: int) -> dict:
    n = len(metrics_list)
    if not n:
        return {col: 0 for col in METRIC_COLUMNS}

    def rate(key: bool) -> float:
        return sum(1 for m in metrics_list if m.get(key)) / n

    return {
        "source_hit_at_5": rate("source_hit_at_5"),
        "gold_doc_hit_at_5": rate("gold_doc_hit_at_5"),
        "gold_page_set_hit_at_5": rate("gold_page_set_hit_at_5"),
        "topic_hit_at_k": rate("topic_hit_at_k"),
        "keyword_coverage": sum(float(m.get("keyword_coverage", 0)) for m in metrics_list) / n,
        "duplicate_doc_ratio": sum(float(m.get("duplicate_doc_ratio", 0)) for m in metrics_list) / n,
        "duplicate_page_ratio": sum(float(m.get("duplicate_page_ratio", 0)) for m in metrics_list) / n,
        "ambiguous_count": sum(1 for m in metrics_list if m.get("is_ambiguous")),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "setting_id",
        "setting_label",
        "max_chunks_per_page",
        "max_chunks_per_doc",
        "enable_mmr",
        "num_questions",
        "top_k",
        "eval_k",
        "fetch_k",
        *METRIC_COLUMNS,
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_txt(path: Path, rows: list[dict], meta: dict) -> None:
    lines = [
        "Rerank Ablation Study",
        f"Generated: {meta['generated_at']}",
        f"Questions: {meta['num_questions']} | top_k={meta['top_k']} | eval_k={meta['eval_k']} | fetch_k={meta['fetch_k']}",
        f"Corpus: unified_{meta['unified_id']}",
        "",
    ]
    header = f"{'Setting':<22} {'src@5':>7} {'doc@5':>7} {'page@5':>7} {'topic':>7} {'kw_cov':>7} {'dup_d':>7} {'dup_p':>7} {'ambig':>6}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        lines.append(
            f"{r['setting_label']:<22} "
            f"{r['source_hit_at_5']:>6.1%} "
            f"{r['gold_doc_hit_at_5']:>6.1%} "
            f"{r['gold_page_set_hit_at_5']:>6.1%} "
            f"{r['topic_hit_at_k']:>6.1%} "
            f"{r['keyword_coverage']:>6.1%} "
            f"{r['duplicate_doc_ratio']:>6.1%} "
            f"{r['duplicate_page_ratio']:>6.1%} "
            f"{r['ambiguous_count']:>6d}"
        )
    lines.extend(["", "Config detail:"])
    for r in rows:
        mmr = "on" if r.get("enable_mmr") else "off"
        lines.append(
            f"  {r['setting_id']} {r['setting_label']}: "
            f"rerank={'yes' if r['setting_label'] != 'baseline' else 'no'} "
            f"max/page={r['max_chunks_per_page']} max/doc={r['max_chunks_per_doc']} mmr={mmr}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Rerank parameter ablation study.")
    parser.add_argument("--unified", type=str, default="full_corpus_715_v1")
    parser.add_argument("--questions", type=Path, default=Path("data/eval/pilot_validation_questions.jsonl"))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--eval-k", type=int, default=5)
    parser.add_argument("--fetch-k", type=int, default=40)
    parser.add_argument("--index-dir", type=Path, default=Path("data/processed/index"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/processed/chunks"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/logs/pilot_validation"),
    )
    args = parser.parse_args()

    collection, embed_model, _manifest = load_unified_collection(args.unified, args.index_dir)
    questions = load_questions(args.questions)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Cache fetch pool once per question (same raw retrieval for all rerank variants)
    pools: dict[str, list[RetrievedChunk]] = {}
    for row in questions:
        qid = str(row["question_id"])
        print(f"Fetching pool: {qid}", flush=True)
        pools[qid] = retrieve_for_question(
            collection,
            embed_model,
            row,
            top_k=args.fetch_k,
            fetch_k=args.fetch_k,
            chunks_dir=args.chunks_dir,
        )

    summary_rows: list[dict] = []
    per_setting_detail: dict[str, list[dict]] = {}

    for setting in SETTINGS:
        print(f"\n=== {setting.name} {setting.label} ===", flush=True)
        metrics_list: list[dict] = []
        per_q: list[dict] = []

        for row in questions:
            qid = str(row["question_id"])
            retrieved = _apply_setting(pools[qid], row=row, top_k=args.top_k, setting=setting)
            m = compute_retrieval_metrics(row, retrieved, top_k=args.top_k, eval_k=args.eval_k)
            md = m.to_dict()
            metrics_list.append(md)
            per_q.append(
                {
                    "question_id": qid,
                    "pages": [c.page_number for c in retrieved],
                    "num_chunks": len(retrieved),
                    **{k: md.get(k) for k in METRIC_COLUMNS if k != "ambiguous_count"},
                    "is_ambiguous": md.get("is_ambiguous"),
                }
            )
            print(
                f"  {qid}: kw={md.get('keyword_coverage', 0):.0%} "
                f"dup_d={md.get('duplicate_doc_ratio', 0):.0%} "
                f"page_set={md.get('gold_page_set_hit_at_5')} "
                f"n={len(retrieved)} pages={ [c.page_number for c in retrieved[:5]] }",
                flush=True,
            )

        agg = _aggregate_row(metrics_list, eval_k=args.eval_k, top_k=args.top_k)
        row_out = {
            "setting_id": setting.name,
            "setting_label": setting.label,
            "max_chunks_per_page": setting.max_chunks_per_page if setting.use_rerank else "",
            "max_chunks_per_doc": setting.max_chunks_per_doc if setting.use_rerank else "",
            "enable_mmr": setting.enable_mmr if setting.use_rerank else "",
            "num_questions": len(questions),
            "top_k": args.top_k,
            "eval_k": args.eval_k,
            "fetch_k": args.fetch_k,
            **agg,
        }
        summary_rows.append(row_out)
        per_setting_detail[setting.label] = per_q

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "unified_id": args.unified,
        "num_questions": len(questions),
        "top_k": args.top_k,
        "eval_k": args.eval_k,
        "fetch_k": args.fetch_k,
        "embedding_model": embed_model,
    }

    csv_path = args.output_dir / "rerank_ablation.csv"
    txt_path = args.output_dir / "rerank_ablation.txt"
    json_path = args.output_dir / "rerank_ablation.json"

    _write_csv(csv_path, summary_rows)
    _write_txt(txt_path, summary_rows, meta)
    json_path.write_text(
        json.dumps(
            {"meta": meta, "summary": summary_rows, "per_question": per_setting_detail},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nCSV:  {csv_path}", flush=True)
    print(f"TXT:  {txt_path}", flush=True)
    print(f"JSON: {json_path}", flush=True)


if __name__ == "__main__":
    main()
