# End-to-End RAG System with SQL, pgvector, Supabase & LangChain

This project is an **end-to-end Retrieval-Augmented Generation (RAG) system** built to answer **natural language business questions over a SQL database** with **correctness, transparency, and control**.

Unlike typical LLM demos, this system explicitly separates:
- **business context understanding (RAG)**
- **query generation (SQL agent)**
- **data execution (PostgreSQL)**

The result is a reliable, debuggable architecture suitable for real-world analytics use cases.

## 🧠 What This Project Does

Users can ask questions like:

> *“If we sell all Levi’s T-shirts today with discounts applied, how much revenue will we generate?”*

The system:
1. Retrieves **relevant business rules** using a vector database (pgvector)
2. Injects that context into the LLM prompt (RAG)
3. Generates **correct, read-only SQL** using a constrained SQL agent
4. Ensures discounts, joins, and aggregations are handled correctly
5. Produces structured output that can be safely executed and explained

## 🏗️ Architecture Overview

```
User Question
      ↓
Embedding Generation (sentence-transformers)
      ↓
Vector Search (pgvector in Supabase)
      ↓
Retrieved Business Context
      ↓
Prompt Injection (RAG)
      ↓
SQL Agent (LangChain)
      ↓
Structured SQL Output (JSON)
      ↓
PostgreSQL Execution
      ↓
Final Answer
```

## Embeddings

Model: sentence-transformers/all-MiniLM-L6-v2

Dimension: 384

Used for semantic similarity search in pgvector

## SQL Agent (LangChain)

Generates SQL only (no execution)

Enforced constraints:

No DML statements

ENUM-aware filtering

Correct JOIN logic

Case-sensitive matching

Output is strictly JSON-locked
```
{
  "sql": "...",
  "metric": "...",
  "assumptions": [...]
}
```
## Retrieval-Augmented Generation (RAG)

RAG is used only for interpretation, not computation.

Examples of stored RAG documents:

> *“Discounts are optional; revenue must include non-discounted items.”*

> *“Post-discount revenue applies percentage discounts per product.”*

> *“Revenue = price × stock quantity.”*

This ensures:

- **SQL logic is consistent**

- **business definitions are respected**

- **data changes do not break reasoning**

## Tech Stack

- **Python**

- **LangChain (Agents + Tools)**

- **Supabase (PostgreSQL)**

- **pgvector**

- **sentence-transformers**

- **psycopg2**

- **GroqCloud**
