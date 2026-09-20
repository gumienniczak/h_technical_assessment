import json
import re
from pathlib import Path

import pandas as pd

import classifier
import main as pipeline

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "listings.csv"


def first_candidate_chat(calls):
    """Fake model that picks the first candidate category in the prompt."""

    def chat(**kwargs):
        calls.append(kwargs)
        prompt = kwargs["messages"][-1]["content"]
        block = re.search(r"candidate categories:\n\n((?:- .+\n?)+)", prompt).group(1)
        first = block.splitlines()[0].removeprefix("- ").strip()
        return {
            "message": {
                "content": json.dumps(
                    {"category": first, "confidence": "Medium", "reasoning": "stub"}
                )
            }
        }

    return chat


def test_pipeline_end_to_end_with_stub_model(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(classifier.client, "chat", first_candidate_chat(calls))
    output_file = tmp_path / "out" / "classified.csv"

    output = pipeline.main(DATA_FILE, output_file)

    original = pd.read_csv(DATA_FILE)
    saved = pd.read_csv(output_file, keep_default_na=False)

    assert len(output) == len(original) == len(saved) == 23
    assert {"predictedCategory", "confidence", "reasoning"} <= set(saved.columns)
    assert set(saved["predictedCategory"]) <= {
        "Nursery",
        "SEN School",
        "Food Store",
        "None",
    }

    # 10 of the 23 listings are excluded by the size rules alone, so the
    # model is only consulted for the remaining 13.
    assert len(calls) == 13
    excluded = saved[saved["reasoning"].str.startswith("Excluded by size rules")]
    assert len(excluded) == 10
    assert set(excluded["predictedCategory"]) == {"None"}
