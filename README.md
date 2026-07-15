# Advanced SQL Syntax Translation, Virtual Execution and Analytical Profiling Engine

A modular, high-performance SQL Engine Simulator designed to demonstrate relational database theory, compiler design principles, analytical profiling, and experimental localized NLP model integration. 

This project is built using Object-Oriented Programming (OOP) paradigms in Python to serve as a comprehensive software engineering portfolio.

---

## Architecture and Core Modules

The system is decoupled into five distinct functional modules to ensure strict adherence to the Single Responsibility Principle:

1. **`parser_engine.py` (The Parser Layer)**
   * Syntactic validation of incoming raw terminal inputs.
   * Leverages `sqlglot` to parse valid SQL statements into an Abstract Syntax Tree (AST).
   * Translates queries seamlessly across different SQL dialects (PostgreSQL, T-SQL, MySQL, Oracle).
   * Renders the hierarchical syntax tree to the terminal via the `--ast` parameter.

2. **`database_executor.py` (The Database & Simulation Layer)**
   * Initializes a temporary, lightweight relational database completely in memory (`sqlite3`).
   * Automatically provisions schemas and seed data for `Personel` and `Yemekler` tables.
   * Executes valid DDL and DML queries in a transaction-safe environment.
   * Features a scalable data generator (`--scale N`) powered by `Faker` to populate the tables with realistic Turkish test data.

3. **`data_analyzer.py` (The Analytical & Logging Layer)**
   * Converts SQL result matrices into `pandas` DataFrames for statistical profiling.
   * Extracts quantitative summaries (`--profile`) including mean, standard deviation, min, max, and missing value counts for numeric columns.
   * Measures execution time at microsecond precision and logs transaction history to `sql_engine_history.log`.
   * Exports analytical query results directly into `.csv` files using the `--export` parameter.

4. **`ai_translator.py` (The Experimental Machine Learning Layer)**
   * Integrates a localized Sequence-to-Sequence (Seq2Seq) Transformer model (`t5-small`) via Hugging Face `transformers`.
   * Triggered optionally with the `--ai` flag to perform syntax translations, highlighting deep learning integration in data workflows without impacting core execution speeds.

5. **`main.py` (The User Interface Layer)**
   * Consolidates all modules into a unified Command Line Interface (CLI).
   * Employs the `rich` library to render color-coded, tabular, and highly readable execution logs directly on the terminal.

---

## Technical Specifications & Features

* **Error Handling:** Invalid terminal inputs are captured gracefully via a custom `NotASQLError` exception, ensuring system stability.
* **Deterministic vs. Stochastic Translation:** Combines high-speed rule-based AST translation (default) with deep learning-based translation (optional AI).
* **High-Volume Data Testing:** Enables simulated performance testing by populating thousands of rows instantly in memory.

---

## Installation

Ensure Python 3.10 or higher is installed. Install the necessary packages using the terminal:

```bash
pip install sqlglot rich faker pandas transformers torch