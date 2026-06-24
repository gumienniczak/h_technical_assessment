import pandas as pd

CATEGORY_RULES = {
    "Nursery": {
        "min_sqft": 2_000,
        "max_sqft": 7_000,
    },
    "SEN School": {
        "min_sqft": 7_000,
        "max_sqft": 25_000,
    },
    "Food Store": {
        "min_sqft": 2_500,
        "max_sqft": 30_000,
        "min_acres": 0.25,
        "max_acres": 10,
    },
}

def determine_candidate_categories(
    listing: pd.Series,
) -> tuple[list[str], bool]:
    
    """
    Determine which acquisition categories remain feasible after applying
    deterministic size constraints.

    If no size information is available, all categories remain possible.
    """
    
    size_ft = listing["sizeFt"]
    size_ac = listing["sizeAc"]

    if pd.isna(size_ft) and pd.isna(size_ac):
        return list(CATEGORY_RULES.keys()), False
    
    candidates = []

    for category, rules in CATEGORY_RULES.items():

        # Check building size
        if not pd.isna(size_ft):

            min_sqft = rules.get("min_sqft")
            max_sqft = rules.get("max_sqft")

            if (
                min_sqft is not None
                and max_sqft is not None
                and min_sqft <= size_ft <= max_sqft
            ):
                candidates.append(category)
                continue

        # Check land size
        if not pd.isna(size_ac):

            min_acres = rules.get("min_acres")
            max_acres = rules.get("max_acres")

            if (
                min_acres is not None
                and max_acres is not None
                and min_acres <= size_ac <= max_acres
            ):
                candidates.append(category)

    return candidates, True