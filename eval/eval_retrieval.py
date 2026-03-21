from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from PIL import Image

from backend.agent import is_generic_visual_query
from backend.config import get_settings
from backend.embeddings import ensure_indexes_loaded, load_catalog, resolve_catalog_image_path
from backend.retrieval import retrieve_products, retrieve_products_debug


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUERIES_PATH = REPO_ROOT / "eval" / "queries.jsonl"
settings = get_settings()

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
if settings.hf_token:
    os.environ.setdefault("HF_TOKEN", settings.hf_token)


def load_queries(path: Path) -> list[dict]:
    queries: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            queries.append(json.loads(line))
    return queries


def build_catalog_index() -> dict[str, dict]:
    return {str(product["id"]): product for product in load_catalog()}


def load_query_image(query: dict, catalog_by_id: dict[str, dict]) -> Image.Image:
    image_product_id = str(query["image_product_id"])
    product = catalog_by_id[image_product_id]
    image_path = resolve_catalog_image_path(product.get("image_path") or product.get("image"))
    if not image_path:
        raise FileNotFoundError(f"Could not resolve image for product {image_product_id}")
    with Image.open(image_path) as image:
        return image.convert("RGB")


def run_query(query: dict, top_k: int, catalog_by_id: dict[str, dict]) -> tuple[list[dict], float]:
    start = time.perf_counter()
    mode = query["mode"]
    if mode == "text":
        products = retrieve_products(
            query=query["query"],
            user_profile=None,
            top_k=top_k,
            mode="text",
        )
    elif mode == "image":
        image = load_query_image(query, catalog_by_id)
        products = retrieve_products(
            image=image,
            user_profile=None,
            top_k=top_k,
            mode="image",
        )
    elif mode == "multimodal":
        text_query = query.get("query", "")
        image = load_query_image(query, catalog_by_id)
        products = retrieve_products(
            query=text_query,
            image=image,
            user_profile=None,
            top_k=top_k,
            mode="multimodal",
            generic_visual_query=is_generic_visual_query(text_query),
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    latency_ms = (time.perf_counter() - start) * 1000
    return products, latency_ms


def run_query_debug(query: dict, top_k: int, catalog_by_id: dict[str, dict]) -> tuple[dict, float]:
    start = time.perf_counter()
    mode = query["mode"]
    if mode == "text":
        payload = retrieve_products_debug(
            query=query["query"],
            user_profile=None,
            top_k=top_k,
            mode="text",
        )
    elif mode == "image":
        image = load_query_image(query, catalog_by_id)
        payload = retrieve_products_debug(
            image=image,
            user_profile=None,
            top_k=top_k,
            mode="image",
        )
    elif mode == "multimodal":
        text_query = query.get("query", "")
        image = load_query_image(query, catalog_by_id)
        payload = retrieve_products_debug(
            query=text_query,
            image=image,
            user_profile=None,
            top_k=top_k,
            mode="multimodal",
            generic_visual_query=is_generic_visual_query(text_query),
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    latency_ms = (time.perf_counter() - start) * 1000
    return payload, latency_ms


def product_text(product: dict) -> str:
    return " ".join(
        [
            product.get("name", ""),
            product.get("brand", ""),
            product.get("category", ""),
            product.get("subcategory", ""),
            product.get("description", ""),
            product.get("visual_description", ""),
            " ".join(product.get("features", [])),
            " ".join(product.get("tags", [])),
            " ".join(product.get("use_cases", [])),
        ]
    ).lower()


def reciprocal_rank(results: list[dict], positive_ids: set[str]) -> float:
    if not positive_ids:
        return 0.0
    for rank, product in enumerate(results, start=1):
        if str(product["id"]) in positive_ids:
            return 1.0 / rank
    return 0.0


def evaluate_query(query: dict, results: list[dict], latency_ms: float) -> dict:
    result_ids = [str(product["id"]) for product in results]
    top_result = results[0] if results else None
    positive_ids = {str(product_id) for product_id in query.get("positive_product_ids", [])}
    expected_categories = set(query.get("expected_categories", []))
    expected_subcategories = set(query.get("expected_subcategories", []))
    expected_keywords = [keyword.lower() for keyword in query.get("expected_keywords", [])]

    strict_hit_at_3 = int(any(product_id in positive_ids for product_id in result_ids[:3])) if positive_ids else 0
    strict_hit_at_5 = int(any(product_id in positive_ids for product_id in result_ids[:5])) if positive_ids else 0
    strict_mrr = reciprocal_rank(results, positive_ids)

    semantic_category_hit = int(
        any(product.get("category") in expected_categories for product in results[:3])
    ) if expected_categories else 0
    semantic_subcategory_hit = int(
        any(product.get("subcategory") in expected_subcategories for product in results[:3])
    ) if expected_subcategories else 0
    semantic_keyword_hit = int(
        any(
            any(keyword in product_text(product) for keyword in expected_keywords)
            for product in results[:3]
        )
    ) if expected_keywords else 0

    return {
        "id": query["id"],
        "mode": query["mode"],
        "query": query.get("query", ""),
        "latency_ms": round(latency_ms, 2),
        "top_result_id": str(top_result["id"]) if top_result else None,
        "top_result_name": top_result["name"] if top_result else None,
        "top_result_subcategory": top_result.get("subcategory") if top_result else None,
        "strict_hit_at_3": strict_hit_at_3,
        "strict_hit_at_5": strict_hit_at_5,
        "strict_mrr": round(strict_mrr, 4),
        "semantic_category_hit_at_3": semantic_category_hit,
        "semantic_subcategory_hit_at_3": semantic_subcategory_hit,
        "semantic_keyword_hit_at_3": semantic_keyword_hit,
        "result_ids": result_ids,
    }


def summarize(rows: list[dict]) -> dict:
    def avg(key: str) -> float:
        return round(statistics.mean(row[key] for row in rows), 4) if rows else 0.0

    summary = {
        "query_count": len(rows),
        "strict_hit_at_3": avg("strict_hit_at_3"),
        "strict_hit_at_5": avg("strict_hit_at_5"),
        "strict_mrr": avg("strict_mrr"),
        "semantic_category_hit_at_3": avg("semantic_category_hit_at_3"),
        "semantic_subcategory_hit_at_3": avg("semantic_subcategory_hit_at_3"),
        "semantic_keyword_hit_at_3": avg("semantic_keyword_hit_at_3"),
        "avg_latency_ms": round(statistics.mean(row["latency_ms"] for row in rows), 2) if rows else 0.0,
    }

    by_mode: dict[str, dict] = {}
    for mode in sorted({row["mode"] for row in rows}):
        mode_rows = [row for row in rows if row["mode"] == mode]
        by_mode[mode] = {
            "query_count": len(mode_rows),
            "strict_hit_at_3": round(statistics.mean(row["strict_hit_at_3"] for row in mode_rows), 4),
            "strict_hit_at_5": round(statistics.mean(row["strict_hit_at_5"] for row in mode_rows), 4),
            "strict_mrr": round(statistics.mean(row["strict_mrr"] for row in mode_rows), 4),
            "semantic_category_hit_at_3": round(statistics.mean(row["semantic_category_hit_at_3"] for row in mode_rows), 4),
            "semantic_subcategory_hit_at_3": round(statistics.mean(row["semantic_subcategory_hit_at_3"] for row in mode_rows), 4),
            "semantic_keyword_hit_at_3": round(statistics.mean(row["semantic_keyword_hit_at_3"] for row in mode_rows), 4),
            "avg_latency_ms": round(statistics.mean(row["latency_ms"] for row in mode_rows), 2),
        }
    summary["by_mode"] = by_mode
    return summary


def print_summary(summary: dict, rows: list[dict]) -> None:
    print("Retrieval Evaluation Summary")
    print("==========================")
    print(f"Queries: {summary['query_count']}")
    print(f"Strict Hit@3: {summary['strict_hit_at_3']:.3f}")
    print(f"Strict Hit@5: {summary['strict_hit_at_5']:.3f}")
    print(f"Strict MRR: {summary['strict_mrr']:.3f}")
    print(f"Semantic Category Hit@3: {summary['semantic_category_hit_at_3']:.3f}")
    print(f"Semantic Subcategory Hit@3: {summary['semantic_subcategory_hit_at_3']:.3f}")
    print(f"Semantic Keyword Hit@3: {summary['semantic_keyword_hit_at_3']:.3f}")
    print(f"Average Latency (ms): {summary['avg_latency_ms']:.2f}")
    print()
    print("By Mode")
    print("-------")
    for mode, mode_summary in summary["by_mode"].items():
        print(
            f"{mode:10} queries={mode_summary['query_count']:>2} "
            f"strict_hit@3={mode_summary['strict_hit_at_3']:.3f} "
            f"strict_mrr={mode_summary['strict_mrr']:.3f} "
            f"semantic_subcategory@3={mode_summary['semantic_subcategory_hit_at_3']:.3f} "
            f"latency_ms={mode_summary['avg_latency_ms']:.2f}"
        )
    print()
    print("Per Query")
    print("---------")
    for row in rows:
        print(
            f"{row['id']:28} mode={row['mode']:10} "
            f"top1={row['top_result_subcategory'] or '-':15} "
            f"strict_hit@3={row['strict_hit_at_3']} "
            f"strict_mrr={row['strict_mrr']:.3f} "
            f"semantic_subcat@3={row['semantic_subcategory_hit_at_3']} "
            f"latency_ms={row['latency_ms']:.2f}"
        )


def _compact_product(product: dict) -> dict:
    return {
        "id": str(product.get("id")),
        "name": product.get("name"),
        "category": product.get("category"),
        "subcategory": product.get("subcategory"),
        "match_debug": product.get("match_debug", {}),
    }


def build_debug_dump(
    queries: list[dict],
    query_ids: set[str],
    top_k: int,
    catalog_by_id: dict[str, dict],
) -> list[dict]:
    dumps: list[dict] = []
    for query in queries:
        if query["id"] not in query_ids:
            continue
        payload, latency_ms = run_query_debug(query, top_k=top_k, catalog_by_id=catalog_by_id)
        dumps.append(
            {
                "id": query["id"],
                "mode": query["mode"],
                "query": query.get("query", ""),
                "latency_ms": round(latency_ms, 2),
                "candidate_pool": [_compact_product(product) for product in payload["candidate_pool"]],
                "heuristic_reranked": [_compact_product(product) for product in payload["heuristic_reranked"]],
                "llm_reranked": [_compact_product(product) for product in payload["llm_reranked"]],
                "final_results": [_compact_product(product) for product in payload["final_results"]],
            }
        )
    return dumps


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality for the AI Commerce Agent.")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_PATH, help="Path to a JSONL query set.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieval results to score per query.")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save full JSON results.")
    parser.add_argument(
        "--debug-query",
        action="append",
        default=[],
        help="Query id to dump candidate pool and rerank stages for. Can be passed multiple times.",
    )
    parser.add_argument(
        "--debug-output",
        type=Path,
        default=None,
        help="Optional path to save rerank debug dumps as JSON.",
    )
    args = parser.parse_args()

    ensure_indexes_loaded(force_rebuild=False)
    catalog_by_id = build_catalog_index()
    queries = load_queries(args.queries)

    rows: list[dict] = []
    for query in queries:
        results, latency_ms = run_query(query, top_k=args.top_k, catalog_by_id=catalog_by_id)
        rows.append(evaluate_query(query, results, latency_ms))

    summary = summarize(rows)
    print_summary(summary, rows)

    debug_dumps = []
    if args.debug_query:
        debug_dumps = build_debug_dump(
            queries=queries,
            query_ids=set(args.debug_query),
            top_k=args.top_k,
            catalog_by_id=catalog_by_id,
        )
        print()
        print("Debug Dumps")
        print("-----------")
        for dump in debug_dumps:
            print(f"{dump['id']} ({dump['mode']})")
            print(f"  candidate_pool: {[item['subcategory'] + ':' + item['id'] for item in dump['candidate_pool'][:5]]}")
            print(f"  heuristic:      {[item['subcategory'] + ':' + item['id'] for item in dump['heuristic_reranked'][:5]]}")
            print(f"  llm:            {[item['subcategory'] + ':' + item['id'] for item in dump['llm_reranked'][:5]]}")
            print(f"  final:          {[item['subcategory'] + ':' + item['id'] for item in dump['final_results'][:5]]}")

    if args.output:
        payload = {
            "queries_path": str(args.queries),
            "top_k": args.top_k,
            "summary": summary,
            "rows": rows,
            "debug_dumps": debug_dumps,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.debug_output:
        args.debug_output.parent.mkdir(parents=True, exist_ok=True)
        args.debug_output.write_text(json.dumps(debug_dumps, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
