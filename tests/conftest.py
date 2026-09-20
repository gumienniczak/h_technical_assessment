import numpy as np
import pandas as pd
import pytest

from data_preprocessing import CLASSIFICATION_COLUMNS


@pytest.fixture
def make_listing():
    """Build a listing row with every classification column present.

    Unspecified fields are NaN, mirroring the sparse real data.
    """

    def _make(**overrides) -> pd.Series:
        row = {column: np.nan for column in CLASSIFICATION_COLUMNS}
        row["destringifiedFeatures"] = []
        row.update(overrides)
        return pd.Series(row)

    return _make
