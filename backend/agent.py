from __future__ import annotations

import json
import re
from collections import defaultdict

from openai import OpenAI
from PIL import Image

from .config import get_settings
from .embeddings import (
    load_catalog,
    resolve_catalog_image_path,
)
from .personalization import (
    load_profile,
    profile_summary,
    update_profile_from_products,
    update_profile_from_query,
)
from .retrieval import retrieve_products


settings = get_settings()

USE_MOCK = settings.use_mock_openai
if USE_MOCK:
    class DummyOpenAI:
        def __init__(self):
            self.chat = self
            self.completions = self
            self.embeddings = self
            self.files = self
            self.responses = self

        def create(self, **kwargs):
            if "model" in kwargs and "messages" in kwargs:
                return type(
                    "Obj",
                    (),
                    {
                        "choices": [
                            type(
                                "C",
                                (),
                                {
                                    "message": type(
                                        "M",
                                        (),
                                        {"content": "Sorry, I'm a dummy response."},
                                    )
                                },
                            )
                        ]
                    },
                )
            if "input" in kwargs:
                return type("Obj", (), {"output_text": "[]"})
            if (
                "input" in kwargs
                and "model" in kwargs
                and kwargs["model"].startswith("text-embedding")
            ):
                return type("Obj", (), {"data": [type("D", (), {"embedding": [0.0] * 1536})]})
            return type("Obj", (), {"id": "file-123"})

    client = DummyOpenAI()
else:
    client = OpenAI(api_key=settings.openai_api_key)


CONVERSATION_MEMORY = defaultdict(list)
GENERIC_VISUAL_QUERY_PATTERNS = (
    "something like this",
    "something similar",
    "similar to this",
    "like this",
    "find something like this",
    "find me something like this",
    "could you find me something like this",
    "show me something similar",
)


def get_product_by_id(product_id: str) -> dict | None:
    for product in load_catalog():
        if str(product.get("id")) == str(product_id):
            return product
    return None


def add_to_memory(user_id: str, role: str, content: str):
    """Add a message to the user's conversation memory."""
    CONVERSATION_MEMORY[user_id].append({"role": role, "content": content})
    CONVERSATION_MEMORY[user_id] = CONVERSATION_MEMORY[user_id][-settings.memory_turn_limit :]


def is_generic_visual_query(user_input: str) -> bool:
    normalized = " ".join((user_input or "").strip().lower().split())
    if not normalized:
        return True
    return any(pattern in normalized for pattern in GENERIC_VISUAL_QUERY_PATTERNS)


def _grounded_product_summary(product: dict) -> str:
    use_case = ", ".join(product.get("use_cases", [])[:2])
    features = ", ".join(product.get("features", [])[:2])
    category = product.get("category", "item")
    parts = [f"{product['name']} is a strong {category} option"]
    if use_case:
        parts.append(f"for {use_case}")
    if features:
        parts.append(f"with {features}")
    return " ".join(parts) + "."


def _join_product_names(products: list[dict]) -> str:
    names = [product["name"] for product in products[:3]]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{names[0]}, {names[1]}, and {names[2]}"


def _grounded_recommendation_intro(prefix: str, products: list[dict]) -> str:
    names = _join_product_names(products)
    if not names:
        return prefix
    return f"{prefix} Top matches include {names}."


def get_system_prompt(user_name: str = "customer") -> str:
    return (
        "Your name is BOT. "
        "You are the intelligent shopping assistant for WilsonWear, a sportswear and lifestyle apparel company. "
        "Your job is to help users find the best products from our catalog using text or image queries, "
        "and to answer any general questions in a helpful, branded, friendly tone. "
        f"You are currently helping {user_name} browse and discover products."
    )


def classify_intent(user_input: str) -> str:
    prompt = (
        'You are an AI assistant that helps route user queries. '
        'Classify the following message as either "product_recommendation" or "general_chat".\n\n'
        f'Message: "{user_input}"\nClassification (just output one word):'
    )
    try:
        response = client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {"role": "system", "content": "You are a concise intent classifier."},
                {"role": "user", "content": prompt},
            ],
        )
        intent = response.choices[0].message.content.strip().lower()
    except Exception:
        intent = "general_chat"
    return intent


def clarify_user_query(user_input: str, user_id: str = "default") -> str:
    prompt = (
        "You are an intelligent shopping assistant. A user just said:\n\n"
        f"\"{user_input}\"\n\n"
        "Rewrite this input as a clear and concise product search query."
    )
    response = client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {
                "role": "system",
                "content": "You rewrite vague user queries into precise product search phrases.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    clarified_query = response.choices[0].message.content.strip()
    add_to_memory(user_id, "assistant", f"(Clarified query: '{clarified_query}')")
    return clarified_query


def explain_recommendation(user_query: str, products: list, user_id: str = "default") -> str:
    if not products:
        return "I'm sorry, we couldn't find any products for that query."
    product_info = "\n".join(
        [
            f"- {p['name']}: {p['description']} | Features: {', '.join(p['features'])} | Why: {', '.join(p.get('match_debug', {}).get('match_reasons', []))}"
            for p in products
        ]
    )
    prompt = (
        f'A user searched for: "{user_query}".\n\n'
        f"The following products were recommended:\n{product_info}\n\n"
        "Write a short, customer-facing recommendation summary. "
        "Focus on fit, style, use case, and standout features. "
        "Do not mention system reasoning, retrieval pipelines, personalization logic, or internal analysis. "
        "Write 2 to 4 sentences as a single natural paragraph. Do not use bullet points, numbering, or per-product mini-sections."
    )
    summary = profile_summary(load_profile(user_id))
    if summary:
        prompt += (
            f"\nUse this preference context quietly when helpful: {summary}. "
            "Do not explicitly mention stored preferences or profile summaries."
        )
    response = client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {
                "role": "system",
                "content": "You are a friendly AI shopping assistant explaining recommendations.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    reasoning = response.choices[0].message.content.strip()
    add_to_memory(
        user_id,
        "user",
        f"(User asked: Why were these products recommended for '{user_query}'?)",
    )
    add_to_memory(user_id, "assistant", reasoning)
    return reasoning


def explain_search(products: list, user_id: str = "default") -> str:
    if not products:
        return "I'm sorry, we couldn't find any products for that image."
    intro = _grounded_recommendation_intro(
        "I found a few visually similar options that should fit the same overall style.",
        products,
    )
    details = " ".join(_grounded_product_summary(product) for product in products[:3])
    reasoning = f"{intro} {details}".strip()
    add_to_memory(user_id, "user", "(User asked: Why were these products shown for my image?)")
    add_to_memory(user_id, "assistant", reasoning)
    return reasoning


def explain_multimodal_search(user_query: str, products: list, user_id: str = "default") -> str:
    if not products:
        return "I'm sorry, we couldn't find any products that matched your text and image together."
    query_hint = f'For "{user_query}", ' if user_query else ""
    intro = _grounded_recommendation_intro(
        f"{query_hint}these options are the closest overall match in both look and likely use.",
        products,
    )
    details = " ".join(_grounded_product_summary(product) for product in products[:3])
    reasoning = f"{intro} {details}".strip()
    add_to_memory(user_id, "assistant", reasoning)
    return reasoning


def refine_text_matches(query: str, candidates: list) -> list:
    product_info = "\n".join(
        [
            f"- {p['name']} (ID: {p['id']}, Category: {p['category']}, Features: {', '.join(p.get('features', []))}) - {p['description']}"
            for p in candidates
        ]
    )
    prompt = (
        f'A user asked for: "{query}".\n\n'
        "Here are candidate product matches:\n"
        f"{product_info}\n\n"
        'Return only an array of product IDs that are good matches (e.g., ["p001", "p002"]).'
    )
    response = client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {
                "role": "system",
                "content": "You are a product filter that selects relevant product matches.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    try:
        valid_ids = re.search(r"\[.*?\]", response.choices[0].message.content.strip(), flags=re.DOTALL)
        json_valid_ids = json.loads(valid_ids.group(0))
        if not json_valid_ids:
            return []
        return [p for p in candidates if p["id"] in json_valid_ids]
    except Exception:
        return []


def refine_image_matches(original_img_path: str, candidates: list) -> list:
    def upload_image_get_id(path: str):
        with open(path, "rb") as file:
            return client.files.create(file=file, purpose="vision").id

    try:
        original_file_id = upload_image_get_id(original_img_path)
    except Exception:
        return []

    vision_content = [
        {
            "type": "input_text",
            "text": (
                "The user provided this image and some candidate product images. "
                "Identify which products are visually similar or match the style. "
                'Please return only an array of product IDs (e.g., ["p001", "p002"]).'
            ),
        },
        {"type": "input_image", "file_id": original_file_id},
    ]
    for product in candidates:
        try:
            candidate_image_path = resolve_catalog_image_path(
                product.get("image_path") or product.get("image")
            )
            if not candidate_image_path:
                continue
            file_id = upload_image_get_id(candidate_image_path)
        except Exception:
            continue
        vision_content.append({"type": "input_image", "file_id": file_id})
        vision_content.append(
            {
                "type": "input_text",
                "text": (
                    f"{product['name']} (ID: {product['id']}) - {product['description']}. "
                    f"Features: {', '.join(product.get('features', []))}."
                ),
            }
        )
    response = client.responses.create(
        model=settings.chat_model,
        input=[{"role": "user", "content": vision_content}],
    )
    try:
        valid_ids = re.search(r"\[.*?\]", response.output_text.strip(), flags=re.DOTALL)
        json_valid_ids = json.loads(valid_ids.group(0))
        if not json_valid_ids:
            return []
        return [p for p in candidates if p["id"] in json_valid_ids]
    except Exception:
        return []


def get_agent_response(user_input: str, user_id: str = "default") -> dict:
    intent = classify_intent(user_input)
    add_to_memory(user_id, "user", user_input)
    if intent == "product_recommendation":
        clarified_query = clarify_user_query(user_input, user_id)
        user_profile = update_profile_from_query(user_id, clarified_query)
        candidates = retrieve_products(
            query=clarified_query,
            user_profile=user_profile,
            top_k=max(settings.text_top_k, 5),
        )
        refined_matches = refine_text_matches(clarified_query, candidates) or candidates[: settings.text_top_k]
        update_profile_from_products(user_id, refined_matches)
        explanation = explain_recommendation(clarified_query, refined_matches, user_id)
        return {
            "type": "recommendation",
            "response": explanation if refined_matches else "Sorry, no matching products were found.",
            "products": refined_matches,
        }

    messages = [{"role": "system", "content": get_system_prompt()}] + CONVERSATION_MEMORY[user_id]
    completion = client.chat.completions.create(model=settings.chat_model, messages=messages)
    assistant_reply = completion.choices[0].message.content.strip()
    add_to_memory(user_id, "assistant", assistant_reply)
    return {"type": "chat", "response": assistant_reply, "products": []}


def process_multimodal_query(user_input: str, image_path: str, user_id: str = "default") -> dict:
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return {"type": "multimodal-search", "response": "Could not process the image.", "products": []}

    add_to_memory(user_id, "user", user_input or "[image attachment]")
    clarified_query = clarify_user_query(user_input, user_id) if user_input.strip() else ""
    generic_visual_query = is_generic_visual_query(user_input or clarified_query)
    if generic_visual_query:
        user_profile = load_profile(user_id)
    else:
        user_profile = update_profile_from_query(user_id, clarified_query or user_input)
    candidates = retrieve_products(
        query=clarified_query or user_input,
        image=img,
        user_profile=user_profile,
        top_k=max(settings.text_top_k, settings.image_top_k, 5),
        mode="multimodal",
        generic_visual_query=generic_visual_query,
    )
    refined_matches = candidates[: max(settings.text_top_k, settings.image_top_k)]
    update_profile_from_products(user_id, refined_matches)
    explanation = explain_multimodal_search(user_input or clarified_query, refined_matches, user_id)
    return {
        "type": "multimodal-search",
        "response": explanation if refined_matches else "Sorry, no matching products were found.",
        "products": refined_matches,
    }


def process_image_query(image_path: str, user_id: str = "default") -> dict:
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return {"type": "image-search", "response": "Could not process the image.", "products": []}
    user_profile = load_profile(user_id)
    candidates = retrieve_products(
        image=img,
        user_profile=user_profile,
        top_k=max(settings.image_top_k, 5),
        mode="image",
    )
    refined_matches = refine_image_matches(image_path, candidates) or candidates[: settings.image_top_k]
    update_profile_from_products(user_id, refined_matches)
    explanation = explain_search(refined_matches, user_id)
    return {
        "type": "image-search",
        "response": explanation if refined_matches else "Sorry, no visually similar products were found.",
        "products": refined_matches,
    }


def process_catalog_image_query(product_id: str, user_id: str = "default") -> dict:
    product = get_product_by_id(product_id)
    if not product:
        return {"type": "image-search", "response": "Could not find that catalog product.", "products": []}

    image_path = resolve_catalog_image_path(product.get("image_path") or product.get("image"))
    if not image_path:
        return {"type": "image-search", "response": "Could not resolve that product image.", "products": []}

    add_to_memory(user_id, "user", f"[search similar to {product.get('name', product_id)}]")
    return process_image_query(image_path, user_id)
