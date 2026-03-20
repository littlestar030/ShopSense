from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "backend" / "data" / "catalog" / "products.json"
DATASET_DIR = REPO_ROOT / "dataset"
STYLES_CSV = DATASET_DIR / "styles.csv"
IMAGES_DIR = DATASET_DIR / "images"

ALLOWED_MASTER_CATEGORIES = {
    "apparel",
    "accessories",
    "footwear",
    "sporting goods",
}

USAGE_TO_STYLE = {
    "casual": "casual",
    "sports": "performance",
    "formal": "polished",
    "smart casual": "smart-casual",
    "party": "statement",
    "ethnic": "heritage-inspired",
    "travel": "travel-ready",
}

USAGE_TO_USE_CASES = {
    "casual": ["daily wear", "weekend styling", "travel"],
    "sports": ["training", "active wear", "gym sessions"],
    "formal": ["office wear", "formal occasions", "polished styling"],
    "smart casual": ["smart casual", "office-to-evening", "social outings"],
    "party": ["party wear", "evening outings", "statement styling"],
    "ethnic": ["festive wear", "cultural occasions", "celebrations"],
    "travel": ["travel", "all-day wear", "on-the-go styling"],
}

FEATURE_MAP = {
    "sports shoes": ["breathable upper", "all-day cushioning", "grippy outsole"],
    "casual shoes": ["versatile styling", "durable sole", "everyday comfort"],
    "formal shoes": ["polished finish", "structured support", "dress-ready silhouette"],
    "sandals": ["easy slip-on wear", "lightweight comfort", "versatile styling"],
    "slippers": ["soft footbed", "easy slip-on wear", "at-home comfort"],
    "flip flops": ["easy slip-on wear", "lightweight feel", "casual comfort"],
    "tshirts": ["soft hand feel", "breathable fabric", "easy layering"],
    "shirts": ["clean silhouette", "versatile styling", "all-day comfort"],
    "tops": ["lightweight feel", "easy styling", "everyday comfort"],
    "jeans": ["durable denim feel", "everyday versatility", "comfortable fit"],
    "track pants": ["easy movement", "soft comfort", "athleisure-ready fit"],
    "shorts": ["lightweight feel", "easy movement", "warm-weather comfort"],
    "jackets": ["layering-friendly", "weather-ready coverage", "structured silhouette"],
    "sweatshirts": ["cozy warmth", "soft comfort", "casual layering"],
    "handbags": ["organized storage", "structured carry", "daily versatility"],
    "backpacks": ["organized storage", "easy carry", "daily versatility"],
    "wallets": ["compact organization", "durable finish", "daily essentials carry"],
    "watches": ["statement styling", "daily wearability", "refined detailing"],
    "sunglasses": ["sun-ready coverage", "lightweight frame", "style-forward finish"],
    "belts": ["adjustable fit", "durable finish", "polished styling"],
}

MATERIAL_MAP = {
    "footwear": "synthetic mesh and rubber",
    "apparel": "cotton blend fabric",
    "accessories": "durable mixed materials",
    "sporting goods": "performance-oriented composite materials",
}

ARTICLE_MATERIAL_OVERRIDES = {
    "jeans": "durable denim",
    "shirts": "woven cotton blend",
    "tshirts": "soft cotton jersey",
    "tops": "lightweight knit fabric",
    "track pants": "stretch performance knit",
    "jackets": "weather-ready shell fabric",
    "sweatshirts": "brushed fleece fabric",
    "handbags": "structured synthetic leather",
    "backpacks": "durable woven fabric",
    "wallets": "textured synthetic leather",
    "watches": "metal and mineral glass",
    "sunglasses": "lightweight acetate and tinted lenses",
    "belts": "coated leather-look material",
}

PRICE_BASE_BY_CATEGORY = {
    "apparel": 42,
    "footwear": 88,
    "accessories": 55,
    "sporting_goods": 72,
}

PRICE_BONUS_BY_ARTICLE = {
    "sports shoes": 28,
    "casual shoes": 18,
    "formal shoes": 22,
    "watches": 70,
    "handbags": 30,
    "backpacks": 18,
    "jeans": 12,
    "jackets": 24,
    "sweatshirts": 10,
}

STOP_BRAND_TOKENS = {
    "men",
    "mens",
    "women",
    "womens",
    "boys",
    "girls",
    "boy",
    "girl",
    "unisex",
    "kids",
    "kid",
}


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "unknown"


def normalize_label(value: str) -> str:
    return slugify(value).replace("-", "_")


def title_or_none(value: str) -> str:
    cleaned = (value or "").strip()
    return cleaned.title() if cleaned else ""


def extract_brand(product_name: str) -> str:
    tokens = product_name.split()
    if tokens and tokens[0].isupper():
        return tokens[0]

    brand_tokens: list[str] = []
    for token in tokens:
        normalized = re.sub(r"[^a-z0-9&]+", "", token.lower())
        if normalized in STOP_BRAND_TOKENS:
            break
        brand_tokens.append(token)
        if len(brand_tokens) == 2:
            break
    if not brand_tokens:
        return "Generic"
    return " ".join(brand_tokens)


def infer_style(usage: str, article_type: str, master_category: str) -> str:
    usage_key = usage.lower()
    if usage_key in USAGE_TO_STYLE:
        return USAGE_TO_STYLE[usage_key]
    article_key = article_type.lower()
    if "watch" in article_key:
        return "refined"
    if "shoe" in article_key:
        return "sporty" if "sport" in article_key else "versatile"
    if master_category.lower() == "accessories":
        return "statement"
    return "modern"


def infer_use_cases(usage: str, article_type: str, sub_category: str) -> list[str]:
    usage_key = usage.lower()
    if usage_key in USAGE_TO_USE_CASES:
        return USAGE_TO_USE_CASES[usage_key]
    article_key = article_type.lower()
    if "shoe" in article_key:
        return ["daily wear", "commuting", "weekend outings"]
    if sub_category.lower() == "bags":
        return ["daily carry", "commuting", "travel"]
    return ["daily wear", "easy styling", "all-day comfort"]


def infer_features(article_type: str, category: str) -> list[str]:
    article_key = article_type.lower()
    if article_key in FEATURE_MAP:
        return FEATURE_MAP[article_key]
    if category == "footwear":
        return ["durable build", "comfortable fit", "daily versatility"]
    if category == "accessories":
        return ["daily utility", "durable finish", "easy styling"]
    return ["comfortable fit", "versatile styling", "everyday wearability"]


def infer_material(article_type: str, category: str) -> str:
    article_key = article_type.lower()
    if article_key in ARTICLE_MATERIAL_OVERRIDES:
        return ARTICLE_MATERIAL_OVERRIDES[article_key]
    return MATERIAL_MAP.get(category, "mixed materials")


def infer_price(category: str, article_type: str, product_id: str) -> float:
    base = PRICE_BASE_BY_CATEGORY.get(category, 48)
    bonus = PRICE_BONUS_BY_ARTICLE.get(article_type.lower(), 0)
    digest = hashlib.sha256(product_id.encode("utf-8")).digest()
    spread = digest[0] % 35
    return round(base + bonus + spread, 2)


def price_band(price: float) -> str:
    if price < 60:
        return "budget"
    if price < 120:
        return "mid-range"
    return "premium"


def inventory_status(product_id: str) -> str:
    statuses = ["in_stock", "in_stock", "low_stock", "in_stock", "preorder"]
    digest = hashlib.sha256(f"inventory-{product_id}".encode("utf-8")).digest()
    return statuses[digest[0] % len(statuses)]


def inferred_rating(product_id: str) -> float:
    digest = hashlib.sha256(f"rating-{product_id}".encode("utf-8")).digest()
    return round(3.8 + (digest[0] % 13) * 0.1, 1)


def inferred_review_count(product_id: str) -> int:
    digest = hashlib.sha256(f"reviews-{product_id}".encode("utf-8")).digest()
    return 20 + digest[0] * 3


def build_tags(
    gender: str,
    color: str,
    season: str,
    usage: str,
    category: str,
    subcategory: str,
    article_type: str,
    style: str,
    brand: str,
) -> list[str]:
    raw_tags = [
        gender.lower().replace("'s", ""),
        color.lower(),
        season.lower(),
        usage.lower(),
        category.replace("_", " "),
        subcategory.replace("_", " "),
        article_type.lower(),
        style.lower(),
        brand.lower(),
    ]
    tags = []
    for tag in raw_tags:
        normalized = re.sub(r"\s+", " ", tag).strip()
        if normalized and normalized not in tags:
            tags.append(normalized)
    return tags


def build_description(name: str, brand: str, color: str, article_type: str, style: str, usage: str, material: str) -> str:
    return (
        f"{name} by {brand} is a {color.lower()} {article_type.lower()} with {style} styling, "
        f"{material}, and a profile suited for {usage.lower()} moments."
    )


def build_visual_description(color: str, article_type: str, style: str) -> str:
    return f"{title_or_none(color)} {article_type.lower()} with a {style} silhouette and catalog-style product photography."


def normalize_gender(value: str) -> str:
    lowered = value.strip().lower()
    if lowered == "men":
        return "Men's"
    if lowered == "women":
        return "Women's"
    if lowered == "boys":
        return "Boys'"
    if lowered == "girls":
        return "Girls'"
    return title_or_none(value)


def is_eligible(row: dict[str, str]) -> bool:
    master = row.get("masterCategory", "").strip().lower()
    product_id = row.get("id", "").strip()
    if master not in ALLOWED_MASTER_CATEGORIES:
        return False
    if not product_id:
        return False
    if not (IMAGES_DIR / f"{product_id}.jpg").exists():
        return False
    return True


def load_rows() -> list[dict[str, str]]:
    if not STYLES_CSV.exists():
        raise FileNotFoundError(f"Could not find styles.csv at {STYLES_CSV}")
    with STYLES_CSV.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader if is_eligible(row)]


def sample_rows(
    rows: list[dict[str, str]],
    max_products: int | None,
    max_per_article: int,
    min_article_count: int,
    seed: int,
) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        article_key = row.get("articleType", "").strip() or row.get("subCategory", "").strip() or "Unknown"
        grouped[article_key].append(row)

    rng = random.Random(seed)
    pools: list[list[dict[str, str]]] = []
    for article_key, group in grouped.items():
        if len(group) < min_article_count:
            continue
        ordered = sorted(group, key=lambda item: int(item["id"]))
        rng.shuffle(ordered)
        pools.append(ordered[:max_per_article])

    pools.sort(key=lambda items: (items[0].get("masterCategory", ""), items[0].get("articleType", "")))

    selected: list[dict[str, str]] = []
    while pools and (max_products is None or len(selected) < max_products):
        next_round: list[list[dict[str, str]]] = []
        for pool in pools:
            if not pool:
                continue
            selected.append(pool.pop())
            if max_products is not None and len(selected) >= max_products:
                break
            if pool:
                next_round.append(pool)
        pools = next_round

    return sorted(selected, key=lambda item: int(item["id"]))


def row_to_product(row: dict[str, str]) -> dict[str, object]:
    product_id = row["id"].strip()
    name = row["productDisplayName"].strip()
    brand = extract_brand(name)
    gender = normalize_gender(row.get("gender", ""))
    color = row.get("baseColour", "").strip() or "Neutral"
    season = row.get("season", "").strip() or "All-Season"
    usage = row.get("usage", "").strip() or "Casual"
    master_category = row.get("masterCategory", "").strip()
    sub_category = row.get("subCategory", "").strip() or "General"
    article_type = row.get("articleType", "").strip() or sub_category
    category = normalize_label(master_category)
    subcategory = normalize_label(article_type or sub_category)
    style = infer_style(usage, article_type, master_category)
    features = infer_features(article_type, category)
    use_cases = infer_use_cases(usage, article_type, sub_category)
    material = infer_material(article_type, category)
    price = infer_price(category, article_type, product_id)

    return {
        "id": product_id,
        "sku": f"FPI-{product_id}",
        "name": name,
        "brand": brand,
        "category": category,
        "subcategory": subcategory,
        "gender": gender,
        "color": title_or_none(color),
        "material": material,
        "style": style,
        "season": title_or_none(season),
        "tags": build_tags(gender, color, season, usage, category, subcategory, article_type, style, brand),
        "use_cases": use_cases,
        "description": build_description(name, brand, color, article_type, style, usage, material),
        "visual_description": build_visual_description(color, article_type, style),
        "features": features,
        "price": price,
        "price_band": price_band(price),
        "rating": inferred_rating(product_id),
        "review_count": inferred_review_count(product_id),
        "inventory_status": inventory_status(product_id),
        "image_path": f"images/{product_id}.jpg",
        "source": {
            "dataset": "fashion-product-images-small",
            "master_category": master_category,
            "sub_category": sub_category,
            "article_type": article_type,
            "usage": usage,
            "year": row.get("year", "").strip(),
        },
    }


def build_catalog(
    max_products: int | None,
    max_per_article: int,
    min_article_count: int,
    seed: int,
) -> list[dict[str, object]]:
    rows = load_rows()
    selected_rows = sample_rows(
        rows=rows,
        max_products=max_products,
        max_per_article=max_per_article,
        min_article_count=min_article_count,
        seed=seed,
    )
    return [row_to_product(row) for row in selected_rows]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a cleaned catalog from the Kaggle fashion product dataset.")
    parser.add_argument("--max-products", type=int, default=1200, help="Maximum number of products to export.")
    parser.add_argument("--max-per-article", type=int, default=60, help="Maximum number of products per article type.")
    parser.add_argument("--min-article-count", type=int, default=20, help="Minimum raw examples required to keep an article type.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    products = build_catalog(
        max_products=args.max_products,
        max_per_article=args.max_per_article,
        min_article_count=args.min_article_count,
        seed=args.seed,
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(products, indent=2), encoding="utf-8")
    print(f"Wrote {len(products)} products to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
