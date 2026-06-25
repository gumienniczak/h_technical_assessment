SYSTEM_PROMPT = """
You are an expert commercial property acquisition classifier.

Your task is to determine which Harkalm acquisition category a commercial
property is most suitable for.

Harkalm acquires existing commercial properties for future conversion.
Assess the property's suitability for acquisition, not simply its current
use.

Base every decision only on the information contained within the listing.

Do not invent missing information or assumptions.

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
    "confidence": "High | Medium | Low",
    "reasoning": ""
}
"""


CATEGORY_GUIDANCE = {

    "Nursery": """
Nursery

Primary acquisition opportunities:

- Office buildings
- Schools
- Places of worship
- Medical centres
- Hotels
- Care homes
- Development land
- New-build commercial properties

The property does NOT need to currently operate as a nursery.

Positive evidence includes:

- the property type matches one of the acquisition opportunities;
- references to alternative uses or conversion;
- community or healthcare use;
- layouts suitable for childcare;
- parking;
- outdoor space.
""",

    "SEN School": """
SEN School

Primary acquisition opportunities:

- Office buildings
- Schools
- Places of worship
- Medical centres
- Hotels
- Care homes
- Development land
- New-build commercial properties

The property does NOT need to currently operate as a SEN school.

Positive evidence includes:

- the property type matches one of the acquisition opportunities;
- references to alternative uses or conversion;
- community or educational use;
- self-contained sites;
- parking;
- outdoor space.
""",

    "Food Store": """
Food Store

Primary acquisition opportunities:

- Existing retail units
- Local centres
- Public houses
- Car showrooms
- MOT centres
- Shopping parades

The property does NOT need to currently operate as a food store.

Positive evidence includes:

- retail use;
- prominent roadside locations;
- local shopping centres;
- references to retail conversion or alternative retail use.
""",
}


CONFIDENCE_GUIDANCE = """
Confidence Levels
=================

High

- Strong evidence supports one candidate category.
- There are no meaningful conflicting signals.

Medium

- Good evidence exists, but some information is incomplete or ambiguous.

Low

- The decision relies on limited evidence.
- Multiple interpretations remain plausible.
"""


CLASSIFICATION_RULES = """
Instructions
============

1. Mandatory size requirements have already been applied.

2. Only consider the candidate categories provided below.

3. Harkalm acquires commercial properties for future conversion.

4. Do NOT classify the property solely according to its current use.

5. Evaluate each candidate category independently.

For each category:

a) Determine whether the property's current type or previous use matches
one of the listed acquisition opportunities.

b) Identify evidence supporting conversion into that category.

Treat phrases such as:

- suitable for alternative uses
- redevelopment opportunity
- community use
- healthcare use
- educational use

as positive evidence where they align with the acquisition criteria.

c) Identify desirable characteristics.

d) Compare all candidate categories and select the one supported by the
strongest overall evidence.

6. Base every decision only on evidence contained within the listing.

7. Do not invent evidence that is not present.

8. Only return "None" if none of the candidate categories are sufficiently
supported by the listing.

9. Choose exactly one category.

10. Keep the reasoning concise and refer only to evidence from the listing.
"""


def build_classification_prompt(
    listing_context: str,
    candidate_categories: list[str],
    size_filter_applied: bool,
) -> str:
    """Build the prompt for classifying a property listing."""

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

Do not infer additional size constraints.
"""
    else:
        size_message = """
No structured size information was available.

If the listing contains size information within the description or key
features, you may use it when evaluating the candidate categories.
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

Evaluation Process
==================

Evaluate every candidate category using the following sequence.

Step 1
Determine whether the property's current type or previous use matches one
of the listed acquisition opportunities.

Step 2
Identify evidence supporting future conversion into that category.

Treat references to:

- alternative uses;
- conversion;
- redevelopment;
- community use;
- healthcare use;
- educational use;

as positive evidence where appropriate.

Step 3
Identify desirable characteristics that strengthen the case.

Step 4
Compare all candidate categories.

Choose the category supported by the strongest overall evidence.

Only return "None" if none of the candidate categories are sufficiently
supported.

Property Listing
================

{listing_context}

{OUTPUT_FORMAT}
""".strip()