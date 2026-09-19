import json
import logging
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

logger = logging.getLogger(__name__)

load_dotenv()


MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "gemma4:12b",
)

OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434",
)

# Retries apply only to unusable model output (invalid JSON / schema).
# Connection errors are not retried: if Ollama is down, fail fast.
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "3"))

# Deterministic decoding so repeated runs give comparable results.
MODEL_OPTIONS = {
    "temperature": 0,
    "seed": 42,
}

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
    candidates: list[str] | None = None
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
        options=MODEL_OPTIONS
    )

    return parse_response(
        response["message"]["content"],
        candidates,
    )


def parse_response(
    response: str,
    candidates: list[str] | None = None,
) -> dict:
    """Return the last top-level JSON object in the reply that passes validation."""
    valid = []
    last_error = "no JSON object found"

    for obj in extract_json_objects(response):
        try:
            validate_response(obj, candidates)
        except ValueError as error:
            last_error = str(error)
            continue
        valid.append(obj)

    if not valid:
        raise ValueError(f"No valid response object: {last_error}")

    if len(valid) > 1:
        logger.warning(
            "Model returned %d valid objects; using the last one.", len(valid)
        )

    return valid[-1]


def extract_json_objects(text: str) -> list[dict]:
    """Return every top-level JSON object in the text, in order of appearance."""
    decoder = json.JSONDecoder()
    objects = []
    position = text.find("{")

    while position != -1:
        try:
            obj, length = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            position = text.find("{", position + 1)   # not JSON, try the next "{"
            continue

        objects.append(obj)
        position = text.find("{", position + length)  # jump past it, skip nested objects

    return objects


def validate_response(
    result: dict,
    candidates: list[str] | None = None,
) -> bool | None:
    """Validate the model response.

        When ``candidates`` is given, the category must be one of them (or
        "None"), so the deterministic rules cannot be overridden by the model.
    """

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

    if (
        candidates is not None
        and result["category"] != "None"
        and result["category"] not in candidates
    ):
        raise ValueError(
            f"Category {result['category']!r} is not one of the "
            f"candidate categories: {candidates}"
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

    return True


def describe_size(listing: pd.Series) -> str:
    """Human-readable size summary used in rule-based reasoning."""

    parts = []

    if not pd.isna(listing["sizeFt"]):
        parts.append(f"{listing['sizeFt']:,.0f} sq ft")

    if not pd.isna(listing["sizeAc"]):
        parts.append(f"{listing['sizeAc']:g} acres")

    return ", ".join(parts)

def classify_listing(
    listing: pd.Series,
    think: bool = False,
) -> dict:
    """Classify a single property listing.
    
    Returns a dict with ``category``, ``confidence`` and ``reasoning``.
    """

    

    candidate_categories, size_filter_applied = (
        determine_candidate_categories(listing)
        )

    # The size rules already exclude every category, so the model is not
        # asked (it could only contradict the rules).
    if not candidate_categories:
        return {
            "category": "None",
            "confidence": "High",
            "reasoning": (
                "Excluded by size rules: the listed size "
                f"({describe_size(listing)}) does not meet the "
                "requirements of any category."
                ),
            }

    listing_context = build_listing_context(listing)

    prompt = build_classification_prompt(
    listing_context,
    candidate_categories,
    size_filter_applied,
    )

    last_error: Exception | None = None
    
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return query_model(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                think=think,
                candidates=candidate_categories,
            )
    
        except ValueError as error:
            # json.JSONDecodeError is a ValueError subclass
            last_error = error
            logger.warning(
                "Attempt %d/%d failed: %s",
                attempt,
                MAX_ATTEMPTS,
                error,
            )
    
    return {
        "category": "None",
        "confidence": "Low",
        "reasoning": (
            f"ERROR: no valid model response after {MAX_ATTEMPTS} "
            f"attempts ({last_error})."
            ),
        }