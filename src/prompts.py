SYSTEM_PROMPT = """
You are an expert commercial property acquisition classifier.

Your task is to classify commercial property listings into exactly one acquisition category.

Do not use information outside of the listing or invent new information.

If there is insufficient evidence to support any category, classify the property as "None".

Return only valid JSON.
"""

OUTPUT_FORMAT = """
Return ONLY valid JSON.

{
    "category": "Nursery | SEN School | Food Store | None",
    "confidence": 0,
    "reasoning": ""
}
"""

CATEGORY_GUIDANCE = """
Classification Categories
=========================

Nursery

Suitable property types include:
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

------------------------------------------------------------

SEN School

Suitable property types include:
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

------------------------------------------------------------

Food Store

Suitable property types include:
- Existing retail units
- Local centres
- Public houses
- Car showrooms
- MOT centres
- Shopping parades

------------------------------------------------------------

None

Select "None" if the listing does not provide sufficient evidence
for any candidate category.
"""

CLASSIFICATION_RULES = """
Instructions
============

1. Mandatory size constraints have already been applied.

2. Only consider the candidate categories provided below.

3. Compare the property against the suitable property types and desirable
   characteristics for each candidate category.

4. Base your decision only on the information contained in the listing.

5. Choose exactly one category.

6. If there is insufficient evidence for any candidate category,
   return "None".

7. Provide a concise reasoning for your decision.
"""

def build_classification_prompt(
    listing_context: str,
    candidate_categories: list[str],
) -> str:

    candidate_list = "\n".join(
        f"- {category}"
        for category in candidate_categories
    )

    return f"""
{CATEGORY_GUIDANCE}

{CLASSIFICATION_RULES}

Candidate Categories
====================

The following categories remain possible after applying the mandatory
size requirements:

{candidate_list}

Property Listing Context
========================

{listing_context}

{OUTPUT_FORMAT}
""".strip()
