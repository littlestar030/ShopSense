from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import numpy as np

from .config import get_settings
from .embeddings import (
    find_similar_products_by_image,
    find_similar_products_by_text,
    get_image_embedding,
    get_text_embedding,
    load_catalog,
)
from .reranking import INTENT_SUBCATEGORY_RULES, detect_query_intents, llm_rerank_products, rerank_products


settings = get_settings()

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "better",
    "but",
    "by",
    "can",
    "could",
    "find",
    "for",
    "from",
    "i",
    "if",
    "in",
    "is",
    "it",
    "like",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "show",
    "similar",
    "something",
    "that",
    "the",
    "these",
    "this",
    "to",
    "want",
    "with",
    "you",
}


def _tokenize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in STOPWORDS]


def _product_text(product: dict[str, Any]) -> str:
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
            product.get("color", ""),
            product.get("material", ""),
            product.get("style", ""),
            product.get("gender", ""),
            product.get("season", ""),
            product.get("price_band", ""),
        ]
    )


def _keyword_score(query: str, product: dict[str, Any]) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0
    product_tokens = _tokenize(_product_text(product))
    product_counts = Counter(product_tokens)
    score = 0.0
    for token in query_tokens:
        if token in product_counts:
            score += 1.0 + math.log1p(product_counts[token])
    return score / len(query_tokens)


def _normalize_score_map(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    min_score = min(values)
    max_score = max(values)
    if math.isclose(min_score, max_score):
        return {key: 1.0 for key in scores}
    return {key: (value - min_score) / (max_score - min_score) for key, value in scores.items()}


def _collect_dense_text_scores(query: str, top_k: int) -> dict[str, float]:
    embedding = get_text_embedding(query)
    results = find_similar_products_by_text(embedding, top_k=top_k)
    return {product["id"]: 1.0 / (rank + 1) for rank, product in enumerate(results)}


def _collect_image_scores(image, top_k: int) -> dict[str, float]:
    embedding = get_image_embedding(image)
    results = find_similar_products_by_image(embedding, top_k=top_k)
    return {product["id"]: 1.0 / (rank + 1) for rank, product in enumerate(results)}


def _collect_keyword_scores(query: str) -> dict[str, float]:
    scores = {}
    for product in load_catalog():
        score = _keyword_score(query, product)
        if score > 0:
            scores[product["id"]] = score
    return scores


def _intent_seed_scores(query: str, intents: set[str]) -> dict[str, float]:
    if not intents:
        return {}

    query_tokens = set(_tokenize(query))
    scores: dict[str, float] = {}
    for product in load_catalog():
        subcategory = product.get("subcategory", "")
        seed_score = 0.0
        for intent in intents:
            rules = INTENT_SUBCATEGORY_RULES[intent]
            if subcategory in rules["allow"]:
                seed_score = max(seed_score, 0.12)
                text_blob = _product_text(product).lower()
                if query_tokens & set(_tokenize(text_blob)):
                    seed_score += 0.08
        if seed_score > 0:
            scores[product["id"]] = max(scores.get(product["id"], 0.0), seed_score)
    return scores


def _merge_candidates(
    keyword_scores: dict[str, float],
    dense_scores: dict[str, float],
    image_scores: dict[str, float],
    seed_scores: dict[str, float],
    candidate_pool_size: int,
    text_weight: float,
    lexical_weight: float,
    image_weight: float,
) -> list[dict[str, Any]]:
    keyword_scores = _normalize_score_map(keyword_scores)
    dense_scores = _normalize_score_map(dense_scores)
    image_scores = _normalize_score_map(image_scores)
    seed_scores = _normalize_score_map(seed_scores)
    product_map = {product["id"]: product for product in load_catalog()}

    merged: list[dict[str, Any]] = []
    for product_id in set(keyword_scores) | set(dense_scores) | set(image_scores) | set(seed_scores):
        product = dict(product_map[product_id])
        text_score = dense_scores.get(product_id, 0.0)
        lexical_score = keyword_scores.get(product_id, 0.0)
        image_score = image_scores.get(product_id, 0.0)
        seed_score = seed_scores.get(product_id, 0.0)

        base_hybrid_score = (
            text_weight * text_score
            + lexical_weight * lexical_score
            + image_weight * image_score
            + 0.20 * seed_score
        )
        product["match_debug"] = {
            "text_score": round(text_score, 4),
            "lexical_score": round(lexical_score, 4),
            "image_score": round(image_score, 4),
            "intent_seed_score": round(seed_score, 4),
            "base_hybrid_score": round(base_hybrid_score, 4),
            "hybrid_score": round(base_hybrid_score, 4),
            "matched_modalities": [
                modality
                for modality, enabled in [("text", text_score > 0 or lexical_score > 0), ("image", image_score > 0)]
                if enabled
            ],
            "match_reasons": [],
        }
        merged.append(product)

    merged.sort(key=lambda product: product["match_debug"]["base_hybrid_score"], reverse=True)
    return merged[:candidate_pool_size]


def _score_margin(candidates: list[dict[str, Any]]) -> float:
    if len(candidates) < 2:
        return float("inf")
    first = float(candidates[0].get("match_debug", {}).get("hybrid_score", 0.0))
    second = float(candidates[1].get("match_debug", {}).get("hybrid_score", 0.0))
    return abs(first - second)


def _should_apply_llm_reranking(query: str, mode: str, candidates: list[dict[str, Any]]) -> bool:
    if not settings.enable_llm_reranking or not query.strip():
        return False
    if mode.lower() not in settings.normalized_llm_rerank_modes:
        return False
    if len(candidates) <= 1:
        return False

    # Multimodal cases are the main high-value target. For other enabled modes,
    # only spend LLM cost when the heuristic ranking is genuinely ambiguous.
    if mode == "multimodal":
        return True
    return _score_margin(candidates) <= settings.llm_rerank_margin_threshold


def retrieve_products(
    query: str | None = None,
    image=None,
    user_profile: dict[str, Any] | None = None,
    top_k: int | None = None,
    mode: str = "text",
    generic_visual_query: bool = False,
) -> list[dict[str, Any]]:
    top_k = top_k or max(settings.text_top_k, settings.image_top_k)
    query = (query or "").strip()

    if not query and image is None:
        return []

    candidate_pool_size = max(settings.rerank_candidate_pool_size, top_k * 2, 6)
    intents = detect_query_intents(query)
    keyword_scores = _collect_keyword_scores(query) if query else {}
    dense_scores = _collect_dense_text_scores(query, top_k=candidate_pool_size) if query else {}
    image_scores = _collect_image_scores(image, top_k=candidate_pool_size) if image is not None else {}
    seed_scores = _intent_seed_scores(query, intents) if query and mode in {"text", "multimodal"} else {}

    if mode == "image":
        text_weight, lexical_weight, image_weight = 0.05, 0.0, 0.95
        personalization_scale = 0.1
        rerank_scale = 0.06
    elif mode == "multimodal":
        if generic_visual_query:
            text_weight, lexical_weight, image_weight = 0.15, 0.05, 0.80
            personalization_scale = 0.2
        else:
            text_weight, lexical_weight, image_weight = 0.35, 0.15, 0.50
            personalization_scale = 0.45
        rerank_scale = 0.10
    else:
        text_weight, lexical_weight, image_weight = 0.50, 0.30, 0.20
        personalization_scale = 1.0
        rerank_scale = 0.15

    candidates = _merge_candidates(
        keyword_scores,
        dense_scores,
        image_scores,
        seed_scores,
        candidate_pool_size=candidate_pool_size,
        text_weight=text_weight,
        lexical_weight=lexical_weight,
        image_weight=image_weight,
    )
    if not settings.enable_reranking:
        return candidates[:top_k]

    reranked = rerank_products(
        query=query or "image search",
        candidates=candidates,
        user_profile=user_profile,
        mode=mode,
        personalization_scale=personalization_scale,
        rerank_scale=rerank_scale,
        top_k=top_k,
    )
    if _should_apply_llm_reranking(query, mode, reranked):
        reranked = llm_rerank_products(
            query=query,
            candidates=reranked,
            mode=mode,
            top_n=min(settings.llm_rerank_top_n, len(reranked)),
        )
    return reranked[:top_k]


def retrieve_products_debug(
    query: str | None = None,
    image=None,
    user_profile: dict[str, Any] | None = None,
    top_k: int | None = None,
    mode: str = "text",
    generic_visual_query: bool = False,
) -> dict[str, Any]:
    top_k = top_k or max(settings.text_top_k, settings.image_top_k)
    query = (query or "").strip()

    if not query and image is None:
        return {
            "mode": mode,
            "query": query,
            "candidate_pool": [],
            "heuristic_reranked": [],
            "llm_reranked": [],
            "final_results": [],
        }

    candidate_pool_size = max(settings.rerank_candidate_pool_size, top_k * 2, 6)
    intents = detect_query_intents(query)
    keyword_scores = _collect_keyword_scores(query) if query else {}
    dense_scores = _collect_dense_text_scores(query, top_k=candidate_pool_size) if query else {}
    image_scores = _collect_image_scores(image, top_k=candidate_pool_size) if image is not None else {}
    seed_scores = _intent_seed_scores(query, intents) if query and mode in {"text", "multimodal"} else {}

    if mode == "image":
        text_weight, lexical_weight, image_weight = 0.05, 0.0, 0.95
        personalization_scale = 0.1
        rerank_scale = 0.06
    elif mode == "multimodal":
        if generic_visual_query:
            text_weight, lexical_weight, image_weight = 0.15, 0.05, 0.80
            personalization_scale = 0.2
        else:
            text_weight, lexical_weight, image_weight = 0.35, 0.15, 0.50
            personalization_scale = 0.45
        rerank_scale = 0.10
    else:
        text_weight, lexical_weight, image_weight = 0.50, 0.30, 0.20
        personalization_scale = 1.0
        rerank_scale = 0.15

    candidates = _merge_candidates(
        keyword_scores,
        dense_scores,
        image_scores,
        seed_scores,
        candidate_pool_size=candidate_pool_size,
        text_weight=text_weight,
        lexical_weight=lexical_weight,
        image_weight=image_weight,
    )
    if not settings.enable_reranking:
        final_results = candidates[:top_k]
        return {
            "mode": mode,
            "query": query,
            "candidate_pool": candidates,
            "heuristic_reranked": candidates,
            "llm_reranked": candidates,
            "final_results": final_results,
        }

    heuristic_reranked = rerank_products(
        query=query or "image search",
        candidates=candidates,
        user_profile=user_profile,
        mode=mode,
        personalization_scale=personalization_scale,
        rerank_scale=rerank_scale,
        top_k=top_k,
    )
    llm_reranked = heuristic_reranked
    if _should_apply_llm_reranking(query, mode, heuristic_reranked):
        llm_reranked = llm_rerank_products(
            query=query,
            candidates=heuristic_reranked,
            mode=mode,
            top_n=min(settings.llm_rerank_top_n, len(heuristic_reranked)),
        )

    final_results = llm_reranked[:top_k]
    return {
        "mode": mode,
        "query": query,
        "candidate_pool": candidates,
        "heuristic_reranked": heuristic_reranked,
        "llm_reranked": llm_reranked,
        "final_results": final_results,
    }
