import pandas as pd


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
    "commercial",c
    "residential",
]

FIELD_LABELS = {
    "pageTitle": "Page Title",
    "propertySubType": "Property Sub Type",
    "sizeFt": "Size (sq ft)",
    "sizeAc": "Size (acres)",
    "summary": "Summary",
    "description": "Description",
    "detailedDescription": "Detailed Description",
    "shareDescription": "Share Description",
}

STRUCTURED_COLUMNS = [
    "pageTitle",
    "propertySubType",
    "sizeFt",
    "sizeAc",
]

TEXT_COLUMNS = [
    "summary",
    "description",
    "detailedDescription",
    "shareDescription",
]


def load_csv(filepath: str) -> pd.DataFrame:
    """Load CSV file into a DataFrame."""

    try:
        return pd.read_csv(filepath)

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


def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that contain only null values."""

    empty_cols = df.columns[
        df.isna().all()
    ].tolist()

    print(
        f"Dropping {len(empty_cols)} empty columns:"
    )
    print(empty_cols)

    return df.dropna(
        axis=1,
        how="all"
    )


def select_classification_cols(
    df: pd.DataFrame,
    selected_cols: list[str],
) -> pd.DataFrame:
    """Select columns required for classification."""

    missing_cols = [
        col
        for col in selected_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}"
        )

    return df[selected_cols].copy()


def destringify_list(value: str) -> list[str]:
    """Convert a stringified list into a Python list."""

    if pd.isna(value):
        return []

    cleaned = value.strip("[]").strip()

    if not cleaned:
        return []

    return [
        item.strip("'\"")
        for item in cleaned.split(", ")
    ]


def clean_value(value) -> str | None:
    """Convert a value into a clean string."""

    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    return value


def build_section(
    listing: pd.Series,
    columns: list[str],
) -> str:
    """Build a formatted text section."""

    parts = []

    for col_name in columns:

        value = clean_value(
            listing[col_name]
        )

        if value is None:
            continue

        parts.append(
            f"{FIELD_LABELS[col_name]}:\n{value}"
        )

    return "\n\n".join(parts)


def build_features_section(
    listing: pd.Series,
) -> str:
    """Build the key features section."""

    features = listing[
        "destringifiedFeatures"
    ]

    if not features:
        return ""

    return (
        "Key Features:\n"
        + "\n".join(
            f"- {feature}"
            for feature in features
        )
    )


def build_listing_context(
    listing: pd.Series,
) -> str:
    """Build the final context sent to the classifier."""

    sections = []

    structured = build_section(
        listing,
        STRUCTURED_COLUMNS,
    )

    if structured:
        sections.append(structured)

    features = build_features_section(
        listing
    )

    if features:
        sections.append(features)

    text = build_section(
        listing,
        TEXT_COLUMNS,
    )

    if text:
        sections.append(text)

    return "\n\n".join(sections)


def main():

    df = load_csv(
        "../data/listings.csv"
    )

    df = process_dataframe(df)

    df = select_classification_cols(
        df,
        CLASSIFICATION_COLUMNS,
    )

    df["destringifiedFeatures"] = (
        df["keyFeatures"]
        .apply(destringify_list)
    )

    print(
        build_listing_context(
            df.iloc[0]
        )
    )


if __name__ == "__main__":
    main()