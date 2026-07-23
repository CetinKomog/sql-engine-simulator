# Hybrid SQL Engine & AI Translator Simulator

This project is a Python-based simulator designed to parse SQL queries, perform dialect translations (e.g., PostgreSQL, MSSQL, Oracle), execute queries against an in-memory database with synthetic data, and translate natural language inputs into SQL using a local machine learning model.

## Features

- **SQL-to-SQL Translation:** Parses and translates queries across different veritabanı lehçeleri using `sqlglot`.
- **Natural Language to SQL (Text-to-SQL):** Integrates a local Hugging Face Seq2Seq model (`transformers`) to generate SQL queries from plain text inputs.
- **In-Memory Database Execution:** Generates mock datasets using `Faker` and executes queries dynamically in memory via `sqlite3`.
- **Data Analytics & Export:** Processes query execution results with `pandas` to generate performance metrics and automatic CSV reports.
- **CLI Interface:** Provides a unified command-line entry point to run both standard dialect translations and AI-assisted query generation.

## Project Architecture

- `parser_engine.py`: Core SQL parsing and dialect conversion logic.
- `database_executor.py`: Mock data generation and in-memory execution pipeline.
- `data_analyzer.py`: Analytical processing, metrics logging, and report exporting.
- `ai_translator.py`: Local Hugging Face pipeline for natural language to SQL translation.
- `main.py`: Command-line orchestrator connecting all modules.

## Requirements

Ensure all required dependencies are installed:

```bash
pip install torch transformers sqlglot pandas faker

Usage
Standard Mode (SQL-to-SQL)
Convert an existing SQL query to a target dialect and execute it against mock data:

python main.py --query "SELECT * FROM users WHERE age > 25" --target-dialect mssql

AI Mode (Text-to-SQL)
Generate an SQL query from plain text and execute it:

python main.py --query "show all employees with salary over 50000" --ai

