import numpy as np
import pandas as pd
import pytest

from rules import determine_candidate_categories


def size(ft=np.nan, ac=np.nan) -> pd.Series:
    return pd.Series({"sizeFt": ft, "sizeAc": ac})


def test_no_size_information_keeps_all_categories():
    candidates, filter_applied = determine_candidate_categories(size())

    assert candidates == ["Nursery", "SEN School", "Food Store"]
    assert filter_applied is False


@pytest.mark.parametrize(
    "sqft, expected",
    [
        (1_999, []),
        (2_000, ["Nursery"]),
        (2_500, ["Nursery", "Food Store"]),
        (7_000, ["Nursery", "SEN School", "Food Store"]),  # both bounds inclusive
        (7_001, ["SEN School", "Food Store"]),
        (25_001, ["Food Store"]),
        (30_001, []),
    ],
)
def test_building_size_boundaries(sqft, expected):
    candidates, filter_applied = determine_candidate_categories(size(ft=sqft))

    assert candidates == expected
    assert filter_applied is True


@pytest.mark.parametrize(
    "acres, expected",
    [
        (0.2, []),
        (0.25, ["Food Store"]),
        (10, ["Food Store"]),
        (10.1, []),
    ],
)
def test_land_size_only_applies_to_food_store(acres, expected):
    candidates, filter_applied = determine_candidate_categories(size(ac=acres))

    assert candidates == expected
    assert filter_applied is True


def test_land_size_can_qualify_when_building_size_does_not():
    candidates, _ = determine_candidate_categories(size(ft=1_000, ac=1))

    assert candidates == ["Food Store"]


def test_implausible_size_excludes_everything():
    """Documents current behaviour: the rules trust the structured size."""

    candidates, filter_applied = determine_candidate_categories(size(ft=1))

    assert candidates == []
    assert filter_applied is True
