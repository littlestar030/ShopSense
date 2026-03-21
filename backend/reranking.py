from __future__ import annotations

import json
import re
from typing import Any

from .config import get_settings


settings = get_settings()
_openai_client: Any | None = None

INTENT_SUBCATEGORY_RULES: dict[str, dict[str, set[str]]] = {
    "running": {
        "allow": {"sports_shoes"},
        "penalize": {"formal_shoes", "sandals", "casual_shoes"},
    },
    "basketball": {
        "allow": {"sports_shoes"},
        "penalize": {"formal_shoes", "sandals", "casual_shoes"},
    },
    "hiking": {
        "allow": {"sports_shoes", "backpacks"},
        "penalize": {"formal_shoes", "sandals", "heels", "flats"},
    },
}

COLOR_TOKENS = {
    "black",
    "blue",
    "brown",
    "green",
    "grey",
    "gray",
    "navy",
    "olive",
    "pink",
    "purple",
    "red",
    "silver",
    "tan",
    "white",
    "yellow",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _get_openai_client():
    global _openai_client
    if settings.use_mock_openai or not settings.openai_api_key:
        return None
    if _openai_client is None:
        from openai import OpenAI

        _openai_client = OpenAI(api_key=settings.openai_api_key)
    return _openai_client


def product_text(product: dict[str, Any]) -> str:
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


def reason_strings(query: str, product: dict[str, Any], modalities: list[str]) -> list[str]:
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


def detect_query_intents(query: str) -> set[str]:
    tokens = set(_tokenize(query))
    intents: set[str] = set()
    if tokens & {"run", "running", "jog", "jogging", "runner"}:
        intents.add("running")
    if tokens & {"basketball", "court", "hoops"}:
        intents.add("basketball")
    if tokens & {"hike", "hiking", "trail", "trek"}:
        intents.add("hiking")
    return intents


def query_alignment_adjustment(query: str, product: dict[str, Any]) -> tuple[float, list[str]]:
    tokens = set(_tokenize(query))
    if not tokens:
        return 0.0, []

    text = product_text(product).lower()
    score = 0.0
    reasons: list[str] = []

    if tokens & {"run", "running", "jog", "jogging", "runner"}:
        if any(term in text for term in ["sports_shoes", "sports shoes", "training", "cushion", "grippy outsole", "active wear"]):
            score += 0.35
            reasons.append("better aligned with running use")
        if any(term in text for term in ["formal_shoes", "formal shoes", "office wear", "formal occasions", "polished styling"]):
            score -= 0.60
            reasons.append("less suitable for running")
        elif any(term in text for term in ["sandals", "slippers", "flip flops", "flats", "heels"]):
            score -= 0.32
            reasons.append("less suitable for running")
        elif any(term in text for term in ["casual_shoes", "casual shoes"]):
            score -= 0.12
            reasons.append("less performance-oriented for running")

    if tokens & {"basketball", "court", "hoops"}:
        if any(term in text for term in ["sports_shoes", "sports shoes", "court", "cushion", "grip"]):
            score += 0.28
            reasons.append("better aligned with court use")
        if any(term in text for term in ["formal_shoes", "formal shoes", "office wear", "formal occasions"]):
            score -= 0.45
            reasons.append("less suitable for court use")
        elif any(term in text for term in ["sandals", "slippers", "flip flops", "flats"]):
            score -= 0.28
            reasons.append("less suitable for court use")

    if tokens & {"hike", "hiking", "trail", "trek"}:
        if any(term in text for term in ["sports_shoes", "sports shoes", "backpacks", "durable", "grip"]):
            score += 0.22
            reasons.append("better aligned with outdoor use")
        if any(term in text for term in ["heels", "flats", "formal"]):
            score -= 0.24
            reasons.append("less suitable for outdoor use")

    return score, reasons[:2]


def query_attribute_adjustment(query: str, product: dict[str, Any]) -> tuple[float, list[str]]:
    tokens = set(_tokenize(query))
    if not tokens:
        return 0.0, []

    text = product_text(product).lower()
    color = (product.get("color") or "").lower()
    score = 0.0
    reasons: list[str] = []

    requested_colors = tokens & COLOR_TOKENS
    if requested_colors:
        if any(token in color or token in text for token in requested_colors):
            score += 0.12
            reasons.append("matches requested color")
        else:
            score -= 0.05
            reasons.append("less aligned with requested color")

    if tokens & {"comfortable", "comfort", "comfy"}:
        if any(term in text for term in ["comfortable", "comfort", "easy movement", "lightweight", "easy slip-on", "comfortable fit"]):
            score += 0.10
            reasons.append("matches comfort preference")

    if tokens & {"training", "gym", "workout", "workouts", "active", "athletic"}:
        if any(term in text for term in ["training", "gym sessions", "active wear", "performance", "sports"]):
            score += 0.18
            reasons.append("fits performance use case")
        elif any(term in text for term in ["daily wear", "weekend styling", "travel", "casual"]):
            score -= 0.08
            reasons.append("less focused on performance use")

    if tokens & {"travel", "commute"}:
        if any(term in text for term in ["travel", "daily wear", "weekend styling", "organized storage", "easy carry"]):
            score += 0.16
            reasons.append("fits travel or carry use")

    if tokens & {"summer", "warm", "weather", "hot"}:
        if any(term in text for term in ["summer", "lightweight", "easy slip-on", "sandals", "travel"]):
            score += 0.14
            reasons.append("fits warm-weather use")

    return score, reasons[:2]


def intent_candidate_adjustment(
    product: dict[str, Any],
    intents: set[str],
    mode: str,
) -> tuple[float, list[str]]:
    if not intents:
        return 0.0, []

    subcategory = product.get("subcategory", "")
    category = product.get("category", "")
    score = 0.0
    reasons: list[str] = []

    for intent in intents:
        rules = INTENT_SUBCATEGORY_RULES[intent]
        allowed = rules["allow"]
        penalized = rules["penalize"]

        if subcategory in allowed:
            score += 0.45 if mode == "multimodal" else 0.30
            reasons.append(f"subcategory fits {intent} intent")
        elif subcategory in penalized:
            score -= 0.85 if mode == "multimodal" else 0.55
            reasons.append(f"subcategory conflicts with {intent} intent")
        elif intent in {"running", "basketball"} and category == "footwear":
            score -= 0.10
        elif intent == "hiking" and category not in {"footwear", "accessories"}:
            score -= 0.12

    return score, reasons[:2]


def personalization_score(product: dict[str, Any], user_profile: dict[str, Any] | None) -> tuple[float, list[str]]:
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


def rerank_products(
    query: str,
    candidates: list[dict[str, Any]],
    user_profile: dict[str, Any] | None,
    mode: str,
    personalization_scale: float,
    rerank_scale: float,
    top_k: int,
) -> list[dict[str, Any]]:
    intents = detect_query_intents(query)
    rescored: list[dict[str, Any]] = []

    for original in candidates:
        product = dict(original)
        match_debug = dict(product.get("match_debug", {}))
        text_score = float(match_debug.get("text_score", 0.0))
        lexical_score = float(match_debug.get("lexical_score", 0.0))
        image_score = float(match_debug.get("image_score", 0.0))
        base_score = float(match_debug.get("base_hybrid_score", match_debug.get("hybrid_score", 0.0)))

        modalities = []
        if text_score > 0 or lexical_score > 0:
            modalities.append("text")
        if image_score > 0:
            modalities.append("image")

        base_reasons = reason_strings(query, product, modalities)
        rerank_bonus = rerank_scale * len(base_reasons)
        alignment_bonus, alignment_reasons = query_alignment_adjustment(query, product)
        attribute_bonus, attribute_reasons = query_attribute_adjustment(query, product)
        intent_bonus, intent_reasons = intent_candidate_adjustment(product, intents, mode)
        personalization_bonus, personalization_reasons = personalization_score(product, user_profile)
        personalization_bonus *= personalization_scale

        final_score = base_score + rerank_bonus + alignment_bonus + attribute_bonus + intent_bonus + personalization_bonus
        match_debug.update(
            {
                "base_hybrid_score": round(base_score, 4),
                "personalization_score": round(personalization_bonus, 4),
                "query_alignment_score": round(alignment_bonus, 4),
                "query_attribute_score": round(attribute_bonus, 4),
                "intent_filter_score": round(intent_bonus, 4),
                "rerank_bonus": round(rerank_bonus, 4),
                "hybrid_score": round(final_score, 4),
                "match_reasons": (
                    base_reasons + alignment_reasons + attribute_reasons + intent_reasons + personalization_reasons
                )[:5],
            }
        )
        product["match_debug"] = match_debug
        rescored.append(product)

    rescored.sort(key=lambda product: product["match_debug"]["hybrid_score"], reverse=True)

    if intents:
        protected_subcategories = set()
        for intent in intents:
            protected_subcategories.update(INTENT_SUBCATEGORY_RULES[intent]["allow"])
        protected = [product for product in rescored if product.get("subcategory") in protected_subcategories]
        if mode == "multimodal" and protected:
            head = protected[:top_k]
            protected_ids = {product["id"] for product in head}
            tail = [product for product in rescored if product["id"] not in protected_ids]
            rescored = head + tail

    return rescored[:top_k]


def llm_rerank_products(
    query: str,
    candidates: list[dict[str, Any]],
    mode: str,
    top_n: int,
) -> list[dict[str, Any]]:
    client = _get_openai_client()
    if client is None or not query.strip() or mode not in {"text", "multimodal"}:
        return candidates

    subset = candidates[:top_n]
    if len(subset) <= 1:
        return candidates

    candidate_descriptions = []
    for product in subset:
        candidate_descriptions.append(
            {
                "id": str(product["id"]),
                "name": product.get("name", ""),
                "category": product.get("category", ""),
                "subcategory": product.get("subcategory", ""),
                "description": product.get("description", ""),
                "features": product.get("features", [])[:3],
                "use_cases": product.get("use_cases", [])[:3],
                "tags": product.get("tags", [])[:5],
                "base_score": product.get("match_debug", {}).get("hybrid_score", 0.0),
            }
        )

    prompt = (
        "Re-rank these shopping candidates for the user query. "
        "Prioritize product usefulness, intent fit, and likely recommendation quality. "
        "Pay special attention to activity intent such as running, basketball, hiking, office wear, or commute. "
        "Return only a JSON array of candidate IDs ordered best to worst.\n\n"
        f"Mode: {mode}\n"
        f"Query: {query}\n"
        f"Candidates: {json.dumps(candidate_descriptions, ensure_ascii=True)}"
    )

    try:
        response = client.chat.completions.create(
            model=settings.llm_rerank_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a product reranker. Return only a JSON array of IDs.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content.strip()
        match = re.search(r"\[[^\]]*\]", content, flags=re.DOTALL)
        if not match:
            return candidates
        reranked_ids = json.loads(match.group(0))
        if not isinstance(reranked_ids, list):
            return candidates
    except Exception:
        return candidates

    subset_by_id = {str(product["id"]): product for product in subset}
    ordered_subset: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for product_id in reranked_ids:
        product_id = str(product_id)
        if product_id in subset_by_id and product_id not in seen_ids:
            product = subset_by_id[product_id]
            product = dict(product)
            match_debug = dict(product.get("match_debug", {}))
            match_debug["llm_rerank_applied"] = True
            match_debug["llm_rerank_model"] = settings.llm_rerank_model
            match_debug["llm_rerank_position"] = len(ordered_subset) + 1
            product["match_debug"] = match_debug
            ordered_subset.append(product)
            seen_ids.add(product_id)

    for product in subset:
        product_id = str(product["id"])
        if product_id not in seen_ids:
            product = dict(product)
            match_debug = dict(product.get("match_debug", {}))
            match_debug["llm_rerank_applied"] = True
            match_debug["llm_rerank_model"] = settings.llm_rerank_model
            match_debug["llm_rerank_position"] = len(ordered_subset) + 1
            product["match_debug"] = match_debug
            ordered_subset.append(product)
            seen_ids.add(product_id)

    return ordered_subset + candidates[top_n:]
