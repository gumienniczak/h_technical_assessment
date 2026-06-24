SYSTEM_PROMPT = """
You are an expert commercial property acquisition classifier.

Your task is to classify commercial property listings into exactly one
acquisition category.

Do not use information outside of the listing or invent new information.

If there is insufficient evidence to support any category, classify the
property as "None".

Return only valid JSON.
"""

OUTPUT_FORMAT = """
Return ONLY valid JSON.

The "category" field must be exactly one of:

- Nursery
- SEN School
- Food Store
- None

{
    "category": "",
    "confidence": "",
    "reasoning": ""
}
"""

CATEGORY_GUIDANCE = {
    "Nursery": """
Nursery

Typical suitable property types include:
- Office buildings
- Schools
- Places of worship
- Medical centres
- Hotels
- Care homes
- Development land
- New-build commercial properties

Additional desirable characteristics:
- Parking
- Outdoor space
""",

    "SEN School": """
SEN School

Typical suitable property types include:
- Office buildings
- Schools
- Places of worship
- Medical centres
- Hotels
- Care homes
- Development land
- New-build commercial properties

Additional desirable characteristics:
- Parking
- Outdoor space
- Self-contained sites
""",

    "Food Store": """
Food Store

Typical suitable property types include:
- Existing retail units
- Local centres
- Public houses
- Car showrooms
- MOT centres
- Shopping parades
""",
}

CONFIDENCE_GUIDANCE = """
Confidence Levels
=================

High
- The remaining candidate category is strongly supported by the listing's
  property type, key features and descriptions.
- There are no significant conflicting signals.

Medium
- The listing contains good supporting evidence for one candidate category,
  but some information is missing or ambiguous.

Low
- The classification is based on weak or limited evidence.
- The listing lacks sufficient detail, or multiple interpretations are plausible.
"""

CLASSIFICATION_RULES = """
Instructions
============

1. Mandatory size requirements have already been applied.

2. Only consider the candidate categories provided below.

3. Use the property type, key features and listing description to determine
   which candidate category is the best match.

4. Base your decision only on the information contained in the listing.

5. Choose exactly one category.

6. If none of the candidate categories are sufficiently supported by the
   listing, return "None".

7. If the listing contains conflicting evidence, base your classification
   on the strongest overall evidence.

8. Provide a concise reasoning for your decision.
"""


def build_classification_prompt(
    listing_context: str,
    candidate_categories: list[str],
    size_filter_applied: bool,
) -> str:
    
    """Build the user prompt for classifying a property listing."""

    category_guidance = "\n\n".join(
        CATEGORY_GUIDANCE[category]
        for category in candidate_categories
    )

    candidate_list = "\n".join(
        f"- {category}"
        for category in candidate_categories
    )

    if size_filter_applied:
        size_message = """
Mandatory size requirements have already been applied.

Do not infer or reconsider additional size constraints.
"""
    else:
        size_message = """
No structured size information was available.

If the property listing contains size information in the description or
key features, you may use it when assessing the candidate categories.
"""

    return f"""
{category_guidance}

{CONFIDENCE_GUIDANCE}

{CLASSIFICATION_RULES}

Candidate Categories
====================

{size_message}

Only classify the property as one of the following candidate categories:

{candidate_list}

If none of the candidate categories are sufficiently supported by the
listing, return "None".

Property Listing Context
========================

{listing_context}

{OUTPUT_FORMAT}
""".strip()
