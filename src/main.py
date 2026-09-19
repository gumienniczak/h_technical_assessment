import logging
from pathlib import Path

import pandas as pd

from classifier import classify_listing
from data_preprocessing import (
    CLASSIFICATION_COLUMNS,
    destringify_list,
    load_csv,
    process_dataframe,
    select_classification_cols,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "listings.csv"
OUTPUT_FILE = PROJECT_ROOT / "output" / "classified_listings.csv"

logger = logging.getLogger(__name__)


def main(
    input_file: Path = INPUT_FILE,
    output_file: Path = OUTPUT_FILE,
) -> pd.DataFrame:
    """Classify all property listings and save the results."""

    # Load the original dataset
    original_df = load_csv(str(input_file))

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

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(
        output_file,
        index=False,
    )

    failed = results_df["reasoning"].str.startswith("ERROR:").sum()

    print(
        f"Successfully classified {len(output_df)} listings "
        f"({failed} failed)."
    )
    print(
        results_df["predictedCategory"]
        .value_counts()
        .to_string()
    )
    print(
        f"Results saved to: {output_file}"
    )

    return output_df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    main()