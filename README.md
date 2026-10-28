# Natural Language SQL Agent

An Agentic AI application that lets users query a relational database
using natural language.

## Features

- Database schema inspection through SQLAlchemy
- SQL validation using SQLGlot
- Read-only SQL execution with SQLite
- LLM tool calling using Groq and LangChain
- Stateful workflow orchestration using LangGraph
- Bounded error recovery
- FastAPI backend
- Automated testing with Pytest

## Tech Stack

Python, LangGraph, LangChain, Groq, FastAPI, SQLAlchemy, SQLite,
SQLGlot, and Pytest.

## Setup

Create and activate a virtual environment, then install dependencies:

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install -r requirements.txt

Copy `.env.example` to `.env` and configure your Groq API key.

Create the database and seed the sample data:

    python -m app.database.seed

Run the API:

    uvicorn app.main:app --reload

Open http://127.0.0.1:8000/docs for the API documentation.

Run tests:

    python -m pytest -v
