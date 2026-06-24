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
├── data_preparation.py    # Data loading and context construction
├── prompts.py             # Prompt templates
├── rules.py               # Deterministic business rules
└── main.py                # End-to-end pipeline

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

3. **LLM classification**
   - Generate a prompt containing:
     - the prepared listing context,
     - the remaining candidate categories,
     - category-specific guidance,
     - confidence guidelines,
     - output schema.

   - Query a locally hosted Ollama model.

4. **Validation**
   - Recover JSON responses if necessary.
   - Validate the response schema and allowed values.
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

The classifier writes the enriched dataset to:

```
output/classified_listings.csv
```
