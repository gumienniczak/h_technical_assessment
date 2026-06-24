import pandas as pd

from classifier import classify_listing
from data_preprocessing import (
    CLASSIFICATION_COLUMNS,
    destringify_list,
    load_csv,
    process_dataframe,
    select_classification_cols,
)


INPUT_FILE = "../data/listings.csv"
OUTPUT_FILE = "../output/classified_listings.csv"


def main() -> None:
    """Classify all property listings and save the results."""

    # Load the original dataset
    original_df = load_csv(INPUT_FILE)

    # Prepare the data required for classification
    working_df = process_dataframe(original_df)
    working_df = select_classification_cols(
        working_df,
        CLASSIFICATION_COLUMNS,
    )

    working_df["destringifiedFeatures"] = (
        working_df["keyFeatures"]
        .apply(destringify_list)
    )

    # Classify each listing
    results = working_df.apply(
        classify_listing,
        axis=1,
    )

    results_df = pd.DataFrame(results.tolist()).rename(
        columns={
            "category": "predictedCategory",
        }
    )

    # Append predictions to the original data
    output_df = pd.concat(
        [
            original_df.reset_index(drop=True),
            results_df,
        ],
        axis=1,
    )

    output_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Successfully classified {len(output_df)} listings."
    )
    print(
        f"Results saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()