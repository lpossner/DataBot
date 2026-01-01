# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a data bot project focused on machine learning and LLM-powered data analysis. The codebase includes examples for AutoML (using AutoGluon), SQL query generation using LLMs, and agentic planning with LangGraph.

## Key Technologies

- **AutoGluon**: Used for automated machine learning on tabular data (classification/regression)
- **LangChain & LangGraph**: Framework for building LLM-powered agents and workflows
- **llama-cpp-python**: Local LLM inference using GGUF quantized models
- **Streamlit**: Web interface (app.py - currently a basic chat UI template)
- **SQLite**: Database for storing and querying test result data

## Project Structure

- `app.py` - Streamlit-based chat interface (currently a skeleton with placeholder responses)
- `models.py` - Empty file, likely intended for model definitions
- `scratchpad/` - Experimental notebooks demonstrating key patterns:
  - `automl_example.ipynb` - AutoGluon tabular prediction workflow
  - `sql_bot_example.ipynb` - SQL query generation using local LLMs with LangGraph
  - `planning_example.ipynb` - Tool-calling agent example with LangGraph
  - `prepare_dummy_data.ipynb` - Data preparation utilities
- `scratchpad/AutogluonModels/` - Trained AutoGluon model artifacts (gitignored)
- `scratchpad/LlamaCppModels/` - Local GGUF models for inference (gitignored)
- `Data/` - SQLite databases (gitignored)

## Common Development Commands

The project uses Jupyter notebooks for experimentation. There are no build/test commands defined yet.

To run the Streamlit app:
```bash
streamlit run app.py
```

## Architecture Patterns

### AutoML Workflow (automl_example.ipynb)

The notebook demonstrates a complete supervised learning pipeline for predicting test failures:

1. **Data Loading**: Reads from SQLite database with structure `(igef, test, test_result)` where:
   - `igef` = vehicle identifier
   - `test` = test name (e.g., "test_0", "test_1", ... "test_80")
   - `test_result` = "OK" or "NOK"

2. **Feature Engineering**:
   - Pivots data so each vehicle becomes a row with test results as binary features
   - Target variable: specific test result (e.g., "test_80")
   - Features: all other test results for that vehicle
   - Uses vehicle-level splitting to prevent data leakage

3. **Training**:
   - Uses `TabularPredictor` with F1 metric (handles class imbalance)
   - Configuration: `medium_quality_faster_train` preset, 5-fold bagging, 1-level stacking
   - Trains ensemble of models (CatBoost, XGBoost, LightGBM, Neural Networks, Random Forest, etc.)

4. **Evaluation**: Generates leaderboard, classification report, confusion matrix, and feature importances

### SQL Bot Architecture (sql_bot_example.ipynb)

Three-node LangGraph pipeline for safe SQL query generation:

1. **plan_sql**: LLM generates SQL from natural language using schema description
   - System prompt enforces read-only SELECT queries
   - Schema inspector produces compact table/column/FK descriptions
   - Returns single SQL statement without explanation

2. **execute_sql**: Guardrail + execution layer
   - `ensure_readonly_select()`: Validates only SELECT/WITH queries, auto-adds LIMIT
   - `run_select()`: Executes against SQLite with `Row` factory for dict results

3. **respond**: Formats SQL and result preview as markdown
   - Shows exact SQL used
   - Displays first 10 rows as markdown table

**Key Safety Pattern**: Schema inspection → LLM planning → Guardrail validation → Execution → Response formatting

### Tool-Calling Agent (planning_example.ipynb)

Demonstrates LangGraph agent with tool binding:

- Uses local LLM server (LM Studio or similar at `localhost:1234`)
- Tools defined with `@tool` decorator (e.g., `get_current_time`, `execute_code`)
- `call_tools()` function matches tool calls from AIMessage to actual tool implementations
- Simple single-node graph that invokes LLM and executes tools in response

## Data Model

The primary data structure is a normalized test results table:
```
dummy_data(igef TEXT, test TEXT, test_result TEXT)
```

Where:
- One row per test per vehicle
- `test_result` values: "OK" or "NOK"
- Used for both AutoML feature engineering and SQL querying

## Model Storage

- **AutoGluon models**: Saved to `scratchpad/AutogluonModels/ag-<timestamp>/`
  - Contains metadata.json, version.txt, and model artifacts
  - Use `TabularPredictor.load(path)` to reload

- **GGUF models**: Stored in `scratchpad/LlamaCppModels/<model-name>/`
  - Example: `Llama-3.2-1B-Instruct-Q6_K_L.gguf`
  - Loaded via `ChatLlamaCpp` with custom model_path

## Important Patterns

### Vehicle-Level Splitting
When working with the test data, always split by unique `igef` values (not individual rows) to prevent data leakage, since one vehicle has multiple test results.

### Read-Only SQL Guardrail
The `ensure_readonly_select()` function is critical for security - it validates that queries:
- Start with SELECT or WITH
- Don't contain dangerous statements (INSERT, UPDATE, DELETE, DROP)
- Have a LIMIT clause (adds default if missing)

### LangGraph State Management
State is passed through the graph as a TypedDict with fields like:
- `messages`: Chat history
- `sql`: Generated query
- `rows`: Query results
- `error`: Error messages

Each node returns updated state dict, which is automatically merged by LangGraph.

## Development Notes

- The Streamlit app (app.py) is currently a skeleton - the OpenAI integration is commented out and returns placeholder responses
- `models.py` is empty and likely intended for shared model classes/utilities
- Notebooks in `scratchpad/` are experimental and demonstrate patterns rather than production code
- All model artifacts, databases, and GGUF files are gitignored
