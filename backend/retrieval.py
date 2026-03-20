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


settings = get_settings()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


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
    scores = {}
    for rank, product in enumerate(results):
        scores[product["id"]] = 1.0 / (rank + 1)
    return scores


def _collect_image_scores(image, top_k: int) -> dict[str, float]:
    embedding = get_image_embedding(image)
    results = find_similar_products_by_image(embedding, top_k=top_k)
    scores = {}
    for rank, product in enumerate(results):
        scores[product["id"]] = 1.0 / (rank + 1)
    return scores


def _collect_keyword_scores(query: str) -> dict[str, float]:
    scores = {}
    for product in load_catalog():
        score = _keyword_score(query, product)
        if score > 0:
            scores[product["id"]] = score
    return scores


def _reason_strings(query: str, product: dict[str, Any], modalities: list[str]) -> list[str]:
    query_tokens = set(_tokenize(query))
    reasons: list[str] = []

    feature_hits = [feature for feature in product.get("features", []) if query_tokens & set(_tokenize(feature))]
    tag_hits = [tag for tag in product.get("tags", []) if query_tokens & set(_tokenize(tag))]
    use_case_hits = [use_case for use_case in product.get("use_cases", []) if query_tokens & set(_tokenize(use_case))]

    if feature_hits:
        reasons.append(f"feature match: {', '.join(feature_hits[:2])}")
    if tag_hits:
        reasons.append(f"tag match: {', '.join(tag_hits[:2])}")
    if use_case_hits:
        reasons.append(f"use case match: {', '.join(use_case_hits[:2])}")
    if "image" in modalities:
        reasons.append("visual similarity match")
    if "text" in modalities and not reasons:
        reasons.append("semantic text similarity match")

    return reasons[:3]


def _query_alignment_adjustment(query: str, product: dict[str, Any]) -> tuple[float, list[str]]:
    tokens = set(_tokenize(query))
    if not tokens:
        return 0.0, []

    product_text = _product_text(product).lower()
    score = 0.0
    reasons: list[str] = []

    if tokens & {"run", "running", "jog", "jogging", "runner"}:
        if any(term in product_text for term in ["sports_shoes", "sports shoes", "training", "cushion", "grippy outsole", "active wear"]):
            score += 0.35
            reasons.append("better aligned with running use")
        if any(term in product_text for term in ["formal_shoes", "formal shoes", "office wear", "formal occasions", "polished styling"]):
            score -= 0.60
            reasons.append("less suitable for running")
        elif any(term in product_text for term in ["sandals", "slippers", "flip flops", "flats", "heels"]):
            score -= 0.32
            reasons.append("less suitable for running")
        elif any(term in product_text for term in ["casual_shoes", "casual shoes"]):
            score -= 0.12
            reasons.append("less performance-oriented for running")

    if tokens & {"basketball", "court", "hoops"}:
        if any(term in product_text for term in ["sports_shoes", "sports shoes", "court", "cushion", "grip"]):
            score += 0.28
            reasons.append("better aligned with court use")
        if any(term in product_text for term in ["formal_shoes", "formal shoes", "office wear", "formal occasions"]):
            score -= 0.45
            reasons.append("less suitable for court use")
        elif any(term in product_text for term in ["sandals", "slippers", "flip flops", "flats"]):
            score -= 0.28
            reasons.append("less suitable for court use")

    if tokens & {"hike", "hiking", "trail", "trek"}:
        if any(term in product_text for term in ["sports_shoes", "sports shoes", "backpacks", "durable", "grip"]):
            score += 0.22
            reasons.append("better aligned with outdoor use")
        if any(term in product_text for term in ["heels", "flats", "formal"]):
            score -= 0.24
            reasons.append("less suitable for outdoor use")

    return score, reasons[:2]


def _personalization_score(product: dict[str, Any], user_profile: dict[str, Any] | None) -> tuple[float, list[str]]:
    if not user_profile:
        return 0.0, []

    score = 0.0
    reasons: list[str] = []
    category_prefs = user_profile.get("preferred_categories", {})
    tag_prefs = user_profile.get("preferred_tags", {})
    use_case_prefs = user_profile.get("preferred_use_cases", {})
    price_band = user_profile.get("price_band")

    category = product.get("category")
    if category and category in category_prefs:
        score += min(category_prefs[category] * 0.08, 0.24)
        reasons.append(f"matches your recent interest in {category}")

    matching_tags = [tag for tag in product.get("tags", []) if tag in tag_prefs]
    if matching_tags:
        score += min(sum(tag_prefs[tag] for tag in matching_tags) * 0.04, 0.20)
        reasons.append(f"aligned with your preferred tags: {', '.join(matching_tags[:2])}")

    matching_use_cases = [use_case for use_case in product.get("use_cases", []) if use_case in use_case_prefs]
    if matching_use_cases:
        score += min(sum(use_case_prefs[item] for item in matching_use_cases) * 0.04, 0.20)
        reasons.append(f"aligned with your recent use cases: {', '.join(matching_use_cases[:2])}")

    price = product.get("price", 0)
    if price_band == "low" and price <= 80:
        score += 0.10
        reasons.append("fits your budget preference")
    elif price_band == "high" and price >= 120:
        score += 0.10
        reasons.append("fits your premium preference")

    return score, reasons[:2]


def _merge_candidates(
    query: str,
    keyword_scores: dict[str, float],
    dense_scores: dict[str, float],
    image_scores: dict[str, float],
    user_profile: dict[str, Any] | None,
    top_k: int,
    text_weight: float,
    lexical_weight: float,
    image_weight: float,
    personalization_scale: float,
    rerank_scale: float,
) -> list[dict[str, Any]]:
    keyword_scores = _normalize_score_map(keyword_scores)
    dense_scores = _normalize_score_map(dense_scores)
    image_scores = _normalize_score_map(image_scores)
    product_map = {product["id"]: product for product in load_catalog()}

    merged: list[dict[str, Any]] = []
    for product_id in set(keyword_scores) | set(dense_scores) | set(image_scores):
        product = dict(product_map[product_id])
        text_score = dense_scores.get(product_id, 0.0)
        lexical_score = keyword_scores.get(product_id, 0.0)
        visual_score = image_scores.get(product_id, 0.0)

        hybrid_score = (
            text_weight * text_score
            + lexical_weight * lexical_score
            + image_weight * visual_score
        )
        modalities = []
        if text_score > 0 or lexical_score > 0:
            modalities.append("text")
        if visual_score > 0:
            modalities.append("image")
        base_reasons = _reason_strings(query, product, modalities)
        rerank_bonus = rerank_scale * len(base_reasons)
        personalization_bonus, personalization_reasons = _personalization_score(product, user_profile)
        personalization_bonus *= personalization_scale
        alignment_bonus, alignment_reasons = _query_alignment_adjustment(query, product)
        final_score = hybrid_score + rerank_bonus + personalization_bonus + alignment_bonus

        product["match_debug"] = {
            "text_score": round(text_score, 4),
            "lexical_score": round(lexical_score, 4),
            "image_score": round(visual_score, 4),
            "personalization_score": round(personalization_bonus, 4),
            "query_alignment_score": round(alignment_bonus, 4),
            "hybrid_score": round(final_score, 4),
            "matched_modalities": modalities,
            "match_reasons": (base_reasons + alignment_reasons + personalization_reasons)[:4],
        }
        merged.append(product)

    merged.sort(key=lambda product: product["match_debug"]["hybrid_score"], reverse=True)
    return merged[:top_k]


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

    keyword_scores = _collect_keyword_scores(query) if query else {}
    dense_scores = _collect_dense_text_scores(query, top_k=max(top_k * 2, 6)) if query else {}
    image_scores = _collect_image_scores(image, top_k=max(top_k * 2, 6)) if image is not None else {}

    if not query and image is None:
        return []

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

    return _merge_candidates(
        query or "image search",
        keyword_scores,
        dense_scores,
        image_scores,
        user_profile,
        top_k,
        text_weight=text_weight,
        lexical_weight=lexical_weight,
        image_weight=image_weight,
        personalization_scale=personalization_scale,
        rerank_scale=rerank_scale,
    )
