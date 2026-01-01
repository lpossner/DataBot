# DataBot

An intelligent agent for automated data querying and model building.

## Overview

DataBot is an LLM-powered system that automates:
- **Data querying**: Natural language to SQL translation with safety guardrails
- **AutoML**: Automated machine learning for tabular data using AutoGluon
- **Agentic workflows**: Tool-calling agents built with LangGraph

## Status

⚠️ **Under Development** - This project is currently in active development and not ready for production use.

## Key Features (In Progress)

- Natural language SQL query generation with read-only safety enforcement
- Automated machine learning pipelines for classification and regression tasks
- Streamlit-based chat interface

## Tech Stack

- **AutoGluon** - Automated machine learning
- **LangChain & LangGraph** - LLM orchestration and agent workflows
- **Streamlit** - Web interface
- **SQLite** - Data storage

## Getting Started

1. Install dependencies:
```bash
pip install uv
uv sync
```

2. Run streamlit frontent:
```bash
# Run the Streamlit app
streamlit run app.py
```

Explore the `scratchpad/` directory for experimental notebooks demonstrating core patterns.

## Documentation

See [CLAUDE.md](CLAUDE.md) for detailed architecture and development guidance.
