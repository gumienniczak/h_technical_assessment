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

df = load_csv("../data/listings.csv")
df = process_dataframe(df)
df.info()

print(df.columns.tolist())

print(df[
    [
        "summary",
        "description",
        "detailedDescription",
        "keyFeatures",
        "propertySubType"
    ]
].head(3))

print(df.iloc[0].summary)
print(df.iloc[0].description)
print(df.iloc[0].detailedDescription)
print(df.iloc[0].keyFeatures)
print(df.iloc[0].propertySubType)
print(df.iloc[0].infoReelItems)