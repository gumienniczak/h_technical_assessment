SYSTEM_PROMPT = """
You are an expert commercial property acquisition classifier.

Your task is to classify commercial property listings according to the acquisition criteria.

Never invent information that is not present in the listing.

If there is insufficient evidence to support any category, classify the property as "None".

Return only valid JSON.
"""

OUTPUT_FORMAT = """
Return a JSON object with the following fields:

{
    "category": "...",
    "confidence": 0,
    "reasoning": "..."
}
"""

def build_classification_prompt(
    listing_context: str,
    candidate_categories: list[str],
) -> str:
    pass