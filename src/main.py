import csv
import pandas as pd


def load_csv(filepath: str) -> pd.DataFrame:
    """Takes a filepath and returns a dataframe"""

    try:
        listing_data = pd.read_csv(filepath)
    
    except FileNotFoundError:
        raise ValueError(
            f"CSV file not found: {filepath}"
        )

    except pd.errors.EmptyDataError:
        raise ValueError(
            "CSV file is empty"
        )

    except Exception as e:
        raise RuntimeError(
            f"Failed to load CSV: {e}"
        ) from e

    return listing_data


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:

    empty_cols = df.columns[df.isna().all()].tolist()

    print(
    f"Dropping {len(empty_cols)} empty columns:"
)
    print(empty_cols)

    return df.dropna(axis = 1, how = "all")

CLASSIFICATION_COLUMNS = [
    "summary",
    "propertySubType",
    "description",
    "detailedDescription",
    "shareDescription",
    "keyFeatures",
    "pageTitle",
    "sizeFt",
    "sizeAc",
    "commercial",
    "residential",
]

SUPPORTING_COLUMNS = [
    "price",
    "pricePerUnit",
    "tenureType",
    "tenure",
    "tenureFull",
    "floorAreaUnits",
    "currency",
    "currencyCode",
    "priceFrequency",
    "status",
    "infoReelItems",
    "nearestStations",
    "pointsOfInterest",
    "name",
]

METADATA_COLUMNS = [
    "id",
    "displayAddress",
    "address",
    "postalCode",
    "Region",
    "region",
    "latitude",
    "longitude",
    "firstVisibleDate",
    "listingUpdateDate",
    "updateDate",
    "listingUpdateReason",
    "numberOfImages",
    "agentCompanyAddress",
    "agentCompanyName",
    "agentCompanyPhone",
    "agentCompanyPostcode",
    "listingHistory",
    "analyticsTaxonomy",
]

def select_classification_cols(df: pd.DataFrame, sel_cols: list[str]) -> pd.DataFrame:

    missing_cols = [col for col in sel_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    return df[sel_cols].copy()


def destringify_list(string_list: str) -> list[str]:

    if pd.isna(string_list):
        return []

    return [
            item.strip("'\"") for item in string_list.strip("[]").split(", ")
        ]

def construct_feature_description():
    pass

df = load_csv("../data/listings.csv")
df = process_dataframe(df)
df = select_classification_cols(df, CLASSIFICATION_COLUMNS)
df["destringifiedFeatures"] = df["keyFeatures"].apply(destringify_list)
df.info()
df.iloc[0]["destringifiedFeatures"]

STRUCTURED_ORDER = [
    "pageTitle",
    "propertySubType",
    "sizeFt",
    "sizeAc",
    "commercial",
    "residential",
    "destringifiedFeatures",
]

TEXT_ORDER = [
    "summary",
    "description",
    "detailedDescription",
    "shareDescription",
]

def build_listing_context(listing: pd.Series) -> str:

    description_parts = []

    for col_name in TEXT_ORDER:

        value = listing[col_name]

        if pd.isna(value):
            continue

        value = str(value).strip()

        if not value:
            continue

        description_parts.append(
            f"{col_name}\n{value}"
        )

    return "\n\n".join(description_parts)

# for col in TEXT_COLUMNS:
#     print(f"\n--- {col} ---")
#     print(df.iloc[2][col])

print(df.columns.tolist())

# print(df[
#     [
#         "summary",
#         "description",
#         "detailedDescription",
#         "keyFeatures",
#         "propertySubType"
#     ]
# ].head(3))

# print(df.iloc[0].summary)
# print(df.iloc[0].description)
# print(df.iloc[0].detailedDescription)
# print(df.iloc[0].keyFeatures)
# print(df.iloc[0].propertySubType)
# print(df.iloc[0].infoReelItems)
# print(df.iloc[0].analyticsTaxonomy)

# print(df.iloc[0])

# print(df["analyticsTaxonomy"].iloc[5])
# print(df["propertySubType"].head())
# print(df["commercial"].value_counts(dropna=False))
# print(df["residential"].value_counts(dropna=False))