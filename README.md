# Commercial Property Acquisition Classifier

## Overview

This project classifies commercial property listings into one of four acquisition categories:

- Nursery
- SEN School
- Food Store
- None

The solution combines deterministic business rules with an LLM-based semantic classifier. Objective constraints, such as mandatory size requirements, are evaluated programmatically before the listing is passed to the language model. The LLM is then responsible for interpreting the remaining textual and structured evidence to determine the most appropriate category.

The output is a CSV file containing the original listing data together with the predicted category, confidence level and supporting reasoning.

## Project Structure

```
src/
├── classifier.py          # LLM interaction and response validation
├── data_preprocessing.py    # Data loading and context construction
├── prompts.py             # Prompt templates
├── rules.py               # Deterministic business rules
└── main.py                # End-to-end pipeline

tests/                     # pytest suite (model calls are stubbed)
data/
output/
```

## Approach

The classification pipeline consists of four stages:

1. **Data preparation**
   - Load and clean the input CSV.
   - Extract only the fields relevant for classification.
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
   - Recover JSON responses if necessary.
   - Validate the response schema and allowed values. The category must be one of the remaining candidates, so the model cannot override the size rules.
   - Unusable responses are retried (3 attempts by default). If they keep failing, the listing is recorded as `None` / `Low` with an `ERROR:` reasoning, so one bad response does not abort the run.
   - Append the prediction to the original dataset.

## Installation

Create and activate a virtual environment.

```bash
python -m venv .venv
pip install -r requirements.txt
```

Install the required Ollama model.

```bash
ollama pull gemma4:12b
```

Start the Ollama server.

```bash
ollama serve
```

## Configuration

Create a `.env` file containing:

```text
MODEL_NAME=gemma4:12b
OLLAMA_HOST=http://localhost:11434
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
response validation and recovery, retry behaviour, and an end-to-end run over
the sample data with a stubbed model (no Ollama needed).

## Development notes

The original solution was written for a technical assessment, which permitted
AI tools. After submission I used an AI assistant (Claude) to review the code
and to draft the fixes and the test suite. I reviewed each change, ran the
pipeline and the tests, and checked the effect: after the fixes only the three
expected labels changed in the regenerated output.
