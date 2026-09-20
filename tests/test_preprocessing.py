import numpy as np
import pandas as pd
import pytest

from data_preprocessing import (
    CLASSIFICATION_COLUMNS,
    build_listing_context,
    destringify_list,
    select_classification_cols,
)


class TestDestringifyList:
    def test_parses_simple_list(self):
        assert destringify_list("['Parking', 'Roadside']") == ["Parking", "Roadside"]

    def test_keeps_items_containing_commas_intact(self):
        value = "['Freehold, vacant possession', 'Parking']"

        assert destringify_list(value) == [
            "Freehold, vacant possession",
            "Parking",
        ]

    def test_handles_double_quoted_items_with_apostrophes(self):
        value = '["Grade II listed", "Owner\'s accommodation"]'

        assert destringify_list(value) == [
            "Grade II listed",
            "Owner's accommodation",
        ]

    @pytest.mark.parametrize("value", [np.nan, None, "", "   ", "[]"])
    def test_missing_or_empty_values_give_empty_list(self, value):
        assert destringify_list(value) == []

    def test_malformed_value_falls_back_instead_of_raising(self):
        assert destringify_list("['Parking', 'Roadside'") == ["Parking", "Roadside"]

    def test_drops_blank_items(self):
        assert destringify_list("['Parking', '  ']") == ["Parking"]


class TestBuildListingContext:
    def test_omits_missing_fields(self, make_listing):
        listing = make_listing(summary="Former nursery building")

        context = build_listing_context(listing)

        assert "Former nursery building" in context
        assert "Size (sq ft)" not in context
        assert "Key Features" not in context

    def test_includes_structured_features_and_text(self, make_listing):
        listing = make_listing(
            propertySubType="Office",
            sizeFt=3_200.0,
            summary="Suitable for a variety of alternative uses",
            destringifiedFeatures=["Car park", "Freehold"],
        )

        context = build_listing_context(listing)

        assert "Property Sub Type:\nOffice" in context
        assert "Size (sq ft):\n3200.0" in context
        assert "- Car park" in context
        assert "- Freehold" in context
        assert "alternative uses" in context

    def test_blank_strings_are_ignored(self, make_listing):
        listing = make_listing(summary="   ")

        assert build_listing_context(listing) == ""


class TestDataFrameHelpers:
    def test_select_classification_cols_reports_missing_columns(self):
        df = pd.DataFrame({"summary": ["x"]})

        with pytest.raises(ValueError, match="Missing required columns"):
            select_classification_cols(df, CLASSIFICATION_COLUMNS)
