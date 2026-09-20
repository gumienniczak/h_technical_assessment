import json

import pytest

import classifier
from classifier import (
    classify_listing,
    extract_json_objects,
    parse_response,
    validate_response,
)


def valid_result(category="Nursery", confidence="High"):
    return {
        "category": category,
        "confidence": confidence,
        "reasoning": "Because.",
    }


class FakeChat:
    """Stands in for ``ollama.Client.chat`` and records every call."""

    def __init__(self, *contents):
        self.contents = list(contents)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        content = self.contents.pop(0) if len(self.contents) > 1 else self.contents[0]
        return {"message": {"content": content}}


@pytest.fixture
def fake_chat(monkeypatch):
    def install(*contents):
        chat = FakeChat(*contents)
        monkeypatch.setattr(classifier.client, "chat", chat)
        return chat

    return install


class TestExtractJsonObjects:
    def test_finds_json_inside_markdown_fence(self):
        text = '```json\n{"category": "Nursery"}\n```'
        assert extract_json_objects(text) == [{"category": "Nursery"}]

    def test_finds_json_surrounded_by_text(self):
        text = 'Here you go: {"category": "None"} hope it helps'
        assert extract_json_objects(text) == [{"category": "None"}]

    def test_returns_empty_list_when_no_json_present(self):
        assert extract_json_objects("no json here") == []

    def test_returns_objects_in_order_of_appearance(self):
        assert extract_json_objects('{"a": 1} then {"b": 2}') == [{"a": 1}, {"b": 2}]

    def test_nested_object_is_not_returned_separately(self):
        assert extract_json_objects('{"a": {"b": 1}}') == [{"a": {"b": 1}}]

    def test_skips_braces_that_are_not_json(self):
        text = 'Use {this} format: {"category": "Nursery"}'
        assert extract_json_objects(text) == [{"category": "Nursery"}]


class TestParseResponse:
    def test_accepts_fenced_json(self):
        text = "```json\n" + json.dumps(valid_result()) + "\n```"
        assert parse_response(text)["category"] == "Nursery"

    def test_skips_echoed_template_and_uses_real_answer(self):
        template = '{"category": "", "confidence": "High | Medium | Low", "reasoning": ""}'
        text = template + "\n" + json.dumps(valid_result("Nursery"))
        assert parse_response(text)["category"] == "Nursery"

    def test_uses_last_of_several_valid_objects(self):
        text = json.dumps(valid_result("Nursery")) + " " + json.dumps(valid_result("None"))
        assert parse_response(text)["category"] == "None"

    def test_raises_when_no_valid_object(self):
        with pytest.raises(ValueError):
            parse_response("no json here")

    def test_wrapped_answer_is_rejected(self):
        with pytest.raises(ValueError):
            parse_response(json.dumps({"result": valid_result()}))


class TestValidation:
    def test_accepts_valid_result(self):
        validate_response(valid_result())

    def test_rejects_missing_fields(self):
        with pytest.raises(ValueError, match="Missing"):
            validate_response({"category": "Nursery"})

    def test_rejects_unknown_category(self):
        with pytest.raises(ValueError, match="Invalid category"):
            validate_response(valid_result(category="Warehouse"))

    def test_rejects_unknown_confidence(self):
        with pytest.raises(ValueError, match="Invalid confidence"):
            validate_response(valid_result(confidence="Certain"))

    def test_rejects_non_string_reasoning(self):
        result = valid_result()
        result["reasoning"] = ["not", "a", "string"]

        with pytest.raises(ValueError, match="Reasoning"):
            validate_response(result)

    def test_rejects_category_outside_candidates(self):
        with pytest.raises(ValueError, match="candidate"):
            validate_response(
                valid_result(category="SEN School"),
                candidates=["Nursery"],
            )

    def test_none_is_always_allowed(self):
        validate_response(valid_result(category="None"), candidates=["Nursery"])


class TestClassifyListing:
    def test_excluded_by_size_rules_never_calls_the_model(
        self, make_listing, fake_chat
    ):
        chat = fake_chat(json.dumps(valid_result()))
        listing = make_listing(sizeFt=500.0)

        result = classify_listing(listing)

        assert chat.calls == []
        assert result["category"] == "None"
        assert result["confidence"] == "High"
        assert "500 sq ft" in result["reasoning"]

    def test_returns_valid_model_result(self, make_listing, fake_chat):
        chat = fake_chat(json.dumps(valid_result("Nursery")))
        listing = make_listing(sizeFt=3_000.0, summary="Former day nursery")

        result = classify_listing(listing)

        assert result == valid_result("Nursery")
        assert len(chat.calls) == 1

    def test_uses_deterministic_decoding(self, make_listing, fake_chat):
        chat = fake_chat(json.dumps(valid_result("Nursery")))

        classify_listing(make_listing(sizeFt=3_000.0))

        assert chat.calls[0]["options"]["temperature"] == 0

    def test_retries_after_unusable_output(self, make_listing, fake_chat):
        chat = fake_chat("not json at all", json.dumps(valid_result("Nursery")))

        result = classify_listing(make_listing(sizeFt=3_000.0))

        assert result["category"] == "Nursery"
        assert len(chat.calls) == 2

    def test_gives_up_with_error_result_after_max_attempts(
        self, make_listing, fake_chat
    ):
        chat = fake_chat("not json at all")

        result = classify_listing(make_listing(sizeFt=3_000.0))

        assert len(chat.calls) == classifier.MAX_ATTEMPTS
        assert result["category"] == "None"
        assert result["confidence"] == "Low"
        assert result["reasoning"].startswith("ERROR:")

    def test_model_cannot_override_size_rules(self, make_listing, fake_chat):
        # 2,100 sq ft leaves only Nursery as a candidate.
        chat = fake_chat(json.dumps(valid_result("SEN School")))

        result = classify_listing(make_listing(sizeFt=2_100.0))

        assert result["reasoning"].startswith("ERROR:")
        assert len(chat.calls) == classifier.MAX_ATTEMPTS
