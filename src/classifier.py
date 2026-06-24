import json
import os

import pandas as pd

from dotenv import load_dotenv
from ollama import Client

from data_preprocessing import build_listing_context
from prompts import (
    SYSTEM_PROMPT,
    build_classification_prompt,
)
from rules import determine_candidate_categories


load_dotenv()


MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "gemma4:12b",
)

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434",
)

VALID_CATEGORIES = {
    "Nursery",
    "SEN School",
    "Food Store",
    "None",
}

VALID_CONFIDENCE = {
    "High",
    "Medium",
    "Low",
}


client = Client(host=OLLAMA_HOST)


def query_model(
    prompt: str,
    system_prompt: str | None = None,
    think: bool = False,
) -> dict:
    """Query the configured Ollama model."""

    messages = []

    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    response = client.chat(
        model=MODEL_NAME,
        messages=messages,
        think=think,
        format="json",
    )

    return parse_response(
        response["message"]["content"]
    )


def parse_response(
    response: str,
) -> dict:
    """Parse, recover and validate model output."""

    try:
        result = json.loads(response)

    except json.JSONDecodeError:
        result = json.loads(
            extract_json(response)
        )

    validate_response(result)

    return result


def extract_json(
    text: str,
) -> str:
    """
    Recover the largest JSON object from a model response.
    Handles Markdown code fences and surrounding text.
    """

    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]

    if text.endswith("```"):
        text = text[:-3]

    text = text.strip()

    start = text.find("{")

    if start == -1:
        raise ValueError(
            "No JSON object found."
        )

    while start != -1:

        candidate = text[start:]

        while candidate:

            try:
                json.loads(candidate)
                return candidate

            except json.JSONDecodeError:

                end = candidate.rfind("}")

                if end == -1:
                    break

                candidate = candidate[:end]

        start = text.find("{", start + 1)

    raise ValueError(
        "Could not recover a valid JSON object."
    )


def validate_response(
    result: dict,
) -> None:
    """Validate the model response."""

    print(result)

    required_fields = {
        "category",
        "confidence",
        "reasoning",
    }

    missing = required_fields - result.keys()

    if missing:
        raise ValueError(
            f"Missing response fields: {missing}"
        )

    if result["category"] not in VALID_CATEGORIES:
        raise ValueError(
            f"Invalid category: {result['category']}"
        )

    if result["confidence"] not in VALID_CONFIDENCE:
        raise ValueError(
            f"Invalid confidence level: {result['confidence']}"
        )

    if not isinstance(
        result["reasoning"],
        str,
    ):
        raise ValueError(
            "Reasoning must be a string."
        )


def classify_listing(
    listing: pd.Series,
    think: bool = False,
) -> dict:
    """Classify a single property listing."""

    listing_context = build_listing_context(listing)

    candidate_categories = determine_candidate_categories(
        listing
    )

    prompt = build_classification_prompt(
        listing_context,
        candidate_categories,
    )

    return query_model(
        prompt=prompt,
        system_prompt=SYSTEM_PROMPT,
        think=think,
    )