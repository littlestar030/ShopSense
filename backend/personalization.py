from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from contextlib import contextmanager
from typing import Any

from .config import get_settings


settings = get_settings()

PRICE_HINTS = {
    "cheap": "low",
    "budget": "low",
    "affordable": "low",
    "inexpensive": "low",
    "premium": "high",
    "expensive": "high",
    "luxury": "high",
}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@contextmanager
def _connect():
    connection = sqlite3.connect(settings.profile_db_path)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS session_profiles (
                user_id TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        yield connection
        connection.commit()
    finally:
        connection.close()


def default_profile() -> dict[str, Any]:
    return {
        "preferred_categories": {},
        "preferred_tags": {},
        "preferred_use_cases": {},
        "price_band": None,
        "last_query": None,
    }


def load_profile(user_id: str) -> dict[str, Any]:
    if not settings.enable_personalization:
        return default_profile()

    with _connect() as connection:
        row = connection.execute(
            "SELECT profile_json FROM session_profiles WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    if not row:
        return default_profile()
    return json.loads(row[0])


def save_profile(user_id: str, profile: dict[str, Any]) -> None:
    if not settings.enable_personalization:
        return

    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO session_profiles (user_id, profile_json, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                profile_json = excluded.profile_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (user_id, json.dumps(profile)),
        )


def _bump(counter_map: dict[str, int], values: list[str], amount: int = 1) -> dict[str, int]:
    counter = Counter(counter_map)
    counter.update({value: amount for value in values})
    return dict(counter)


def _extract_price_band(query: str) -> str | None:
    query_tokens = set(_tokenize(query))
    for token, band in PRICE_HINTS.items():
        if token in query_tokens:
            return band
    return None


def update_profile_from_query(user_id: str, query: str) -> dict[str, Any]:
    profile = load_profile(user_id)
    price_band = _extract_price_band(query)
    if price_band:
        profile["price_band"] = price_band
    profile["last_query"] = query
    save_profile(user_id, profile)
    return profile


def update_profile_from_products(user_id: str, products: list[dict[str, Any]]) -> dict[str, Any]:
    profile = load_profile(user_id)
    categories = [product.get("category") for product in products if product.get("category")]
    tags = [tag for product in products for tag in product.get("tags", [])]
    use_cases = [use_case for product in products for use_case in product.get("use_cases", [])]

    if categories:
        profile["preferred_categories"] = _bump(profile.get("preferred_categories", {}), categories)
    if tags:
        profile["preferred_tags"] = _bump(profile.get("preferred_tags", {}), tags)
    if use_cases:
        profile["preferred_use_cases"] = _bump(profile.get("preferred_use_cases", {}), use_cases)

    if products:
        avg_price = sum(product.get("price", 0) for product in products) / len(products)
        if avg_price <= 70:
            profile["price_band"] = profile.get("price_band") or "low"
        elif avg_price >= 120:
            profile["price_band"] = profile.get("price_band") or "high"

    save_profile(user_id, profile)
    return profile


def profile_summary(profile: dict[str, Any]) -> str | None:
    categories = sorted(
        profile.get("preferred_categories", {}).items(),
        key=lambda item: item[1],
        reverse=True,
    )
    tags = sorted(
        profile.get("preferred_tags", {}).items(),
        key=lambda item: item[1],
        reverse=True,
    )

    parts = []
    if categories:
        parts.append(f"leans toward {categories[0][0]}")
    if tags:
        parts.append(f"often likes {tags[0][0]}")
    if profile.get("price_band"):
        parts.append(f"current budget preference: {profile['price_band']}")
    return "; ".join(parts) if parts else None
