# Commercial Property Acquisition Classifier

## Overview

This project classifies commercial property listings into one of four acquisition categories:

- Nursery
- SEN School
- Food Store
- None

The solution combines deterministic business rules with an LLM-based semantic classifier. Objective constraints, such as mandatory size requirements, are evaluated programmatically before the listing is passed to the language model. The LLM is then responsible for interpreting the remaining textual and structured evidence to determine the most appropriate category.

The output is a CSV file containing the original listing data together with the predicted category, confidence level and supporting reasoning.

## Background

This project began as a take-home technical assessment for an AI developer role at another company (anonymised here). The brief was a simplified version of a real classification problem the company works on, and it permitted AI tools. The sample data in `data/listings.csv` is 23 publicly available property listings supplied with the brief.

`ASSESSMENT_NOTES.md` is the write-up from the original submission. After submitting, I reviewed the solution critically and made further changes; see [Development notes](#development-notes).

## Project Structure

```
src/
├── classifier.py           # LLM interaction, response parsing and validation
├── data_preprocessing.py   # Data loading and context construction
├── prompts.py              # Prompt templates
├── rules.py                # Deterministic business rules
└── main.py                 # End-to-end pipeline

tests/                      # pytest suite (model calls are stubbed)
data/                       # Sample listings
output/                     # Classified listings
ASSESSMENT_NOTES.md         # Notes from the original submission
```

## Approach

The classification pipeline consists of four stages:

1. **Data preparation**
   - Load the input CSV and extract only the fields relevant for classification.
   - Convert feature lists into structured Python lists.
   - Build a structured listing context from property type, key features and textual descriptions.

2. **Deterministic filtering**
   - Apply mandatory size requirements in Python.
   - Eliminate categories that cannot satisfy the acquisition criteria.
   - If structured size information is unavailable, all categories remain candidates and the LLM may use any size information contained within the listing text.
   - If size information is available but no category can satisfy it, the listing is classified as `None` without calling the model.

3. **LLM classification**
   - Generate a prompt containing:
     - the prepared listing context,
     - the remaining candidate categories,
     - category-specific guidance,
     - confidence guidelines,
     - output schema.

   - Query a locally hosted Ollama model with deterministic decoding (temperature 0, fixed seed).

4. **Validation**
   - Recover JSON responses if necessary. If a reply contains several JSON objects, the last valid one is used.
   - Validate the response schema and allowed values. The category must be one of the remaining candidates, so the model cannot override the size rules.
   - Unusable responses are retried (3 attempts by default). If they keep failing, the listing is recorded as `None` / `Low` with an `ERROR:` reasoning, so one bad response does not abort the run.
   - Append the prediction to the original dataset.

## Installation

Create and activate a virtual environment, then install the dependencies.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install the required Ollama model.

```bash
ollama pull gemma4:12b
```

Start the Ollama server (skip this if the Ollama app is already running).

```bash
ollama serve
```

## Configuration

Settings are read from environment variables, and every one has a default. To override them, create a `.env` file:

```text
MODEL_NAME=gemma4:12b
OLLAMA_HOST=http://localhost:11434
MAX_ATTEMPTS=3
```

## Running

```bash
python src/main.py
```

Paths are resolved relative to the project, so the command works from any directory.

The classifier writes the enriched dataset to:

```
output/classified_listings.csv
```

Note: `None` is a valid category, but pandas treats the string "None" as a
missing value by default. When reading the output, use
`pd.read_csv(path, keep_default_na=False)`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests cover the size rules, `keyFeatures` parsing, context building,
JSON recovery, response validation, retry behaviour, and an end-to-end run over
the sample data with a stubbed model (no Ollama needed). They check the
pipeline's logic, not the quality of the model's classifications.

## Known limitations

- **No labelled evaluation.** Classification accuracy has not been measured.
- **Confidence is self-reported.** The model's High / Medium / Low labels are not calibrated against labelled data.
- **Size data is trusted.** The rules take the structured size fields as given, so an implausible value (for example, 1 sq ft) excludes a listing. Listings without a structured size, 10 of the 23 samples, rely on the model to read the size from the text, and nothing checks it.
- **Retries repeat the same request.** At temperature 0 an identical prompt normally gives an identical answer, so retries mainly help with transient failures.
- **Sequential processing.** Each listing that passes the size rules is one model call, made one after another.

## Development notes

The original solution was written by me for the assessment. After submission I used an AI assistant (Claude) to review the code and to draft fixes and the test suite. I reviewed each change, ran the pipeline and the tests, and checked the effect: in the regenerated output only three labels changed, all listings the size rules should already have excluded.

Changes made after the original submission:

- `keyFeatures` parsing now uses `ast.literal_eval`, so items containing commas are no longer split.
- The size rules are enforced in code: the model is skipped when no category qualifies, and its category is checked against the candidates.
- Decoding is deterministic, and unusable output is retried and recorded per listing instead of aborting the run.
- JSON recovery finds every JSON object in a reply and uses the last valid one.
- A test suite was added.
