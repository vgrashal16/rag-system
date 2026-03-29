import os
import json
import psycopg2
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# 1.  Database connection
# ─────────────────────────────────────────────

def get_connection():
    return psycopg2.connect(os.environ["DATABASE_URL"])


# ─────────────────────────────────────────────
# 2.  Business rules corpus
#     (covers all 3 complexity tiers explicitly)
# ─────────────────────────────────────────────

BUSINESS_RULES = [
    # ── Tier 1 : Discounted Revenue ──────────────────────────────────────
    {
        "id": "rule_discount_001",
        "tier": "tier1_discounted_revenue",
        "rule": (
            "Discounts are optional per product. "
            "Revenue must include both discounted and non-discounted items. "
            "Never filter out products with NULL discount."
        ),
        "example_sql": (
            "SELECT SUM(price * stock_quantity * (1 - COALESCE(discount, 0))) "
            "AS total_revenue FROM products;"
        ),
    },
    {
        "id": "rule_discount_002",
        "tier": "tier1_discounted_revenue",
        "rule": (
            "Post-discount revenue = price × quantity × (1 - discount_rate). "
            "discount_rate is stored as a decimal between 0 and 1 (e.g., 0.20 for 20%)."
        ),
        "example_sql": (
            "SELECT brand, "
            "SUM(price * stock_quantity * (1 - COALESCE(discount, 0))) AS discounted_revenue "
            "FROM products WHERE brand = 'Levi''s' GROUP BY brand;"
        ),
    },
    {
        "id": "rule_discount_003",
        "tier": "tier1_discounted_revenue",
        "rule": (
            "Revenue without discounts = price × stock_quantity. "
            "Use this when the question does not mention discounts."
        ),
        "example_sql": (
            "SELECT SUM(price * stock_quantity) AS revenue FROM products;"
        ),
    },

    # ── Tier 2 : Optional Joins ───────────────────────────────────────────
    {
        "id": "rule_join_001",
        "tier": "tier2_optional_joins",
        "rule": (
            "Always use LEFT JOIN when combining products with orders or reviews "
            "so that products with zero orders are not excluded."
        ),
        "example_sql": (
            "SELECT p.product_name, COUNT(o.order_id) AS order_count "
            "FROM products p "
            "LEFT JOIN orders o ON p.product_id = o.product_id "
            "GROUP BY p.product_name;"
        ),
    },
    {
        "id": "rule_join_002",
        "tier": "tier2_optional_joins",
        "rule": (
            "Category names in the categories table use title-case (e.g., 'T-Shirts'). "
            "Always join products → categories on category_id for category filtering."
        ),
        "example_sql": (
            "SELECT p.product_name, c.category_name "
            "FROM products p "
            "LEFT JOIN categories c ON p.category_id = c.category_id "
            "WHERE c.category_name = 'T-Shirts';"
        ),
    },
    {
        "id": "rule_join_003",
        "tier": "tier2_optional_joins",
        "rule": (
            "Brand names are case-sensitive strings stored in the products table. "
            "Use exact match (=) not ILIKE unless the user specifies fuzzy search."
        ),
        "example_sql": (
            "SELECT * FROM products WHERE brand = 'Nike';"
        ),
    },

    # ── Tier 3 : Aggregations ─────────────────────────────────────────────
    {
        "id": "rule_agg_001",
        "tier": "tier3_aggregations",
        "rule": (
            "When ranking or comparing brands/categories, always use GROUP BY "
            "and aggregate functions (SUM, COUNT, AVG). "
            "Never return raw rows for analytical questions."
        ),
        "example_sql": (
            "SELECT brand, SUM(price * stock_quantity) AS total_inventory_value "
            "FROM products GROUP BY brand ORDER BY total_inventory_value DESC;"
        ),
    },
    {
        "id": "rule_agg_002",
        "tier": "tier3_aggregations",
        "rule": (
            "Average price queries should use AVG(price) grouped by the requested dimension "
            "(brand, category, size, etc.)."
        ),
        "example_sql": (
            "SELECT category_id, AVG(price) AS avg_price "
            "FROM products GROUP BY category_id;"
        ),
    },
    {
        "id": "rule_agg_003",
        "tier": "tier3_aggregations",
        "rule": (
            "Stock count queries use SUM(stock_quantity) or COUNT(*). "
            "Use COUNT(DISTINCT product_id) for unique product counts."
        ),
        "example_sql": (
            "SELECT brand, SUM(stock_quantity) AS total_stock, "
            "COUNT(DISTINCT product_id) AS unique_products "
            "FROM products GROUP BY brand;"
        ),
    },
    {
        "id": "rule_agg_004",
        "tier": "tier3_aggregations",
        "rule": (
            "For questions asking 'how many orders' per product or brand, "
            "join with the orders table and use COUNT(order_id)."
        ),
        "example_sql": (
            "SELECT p.brand, COUNT(o.order_id) AS total_orders "
            "FROM products p "
            "LEFT JOIN orders o ON p.product_id = o.product_id "
            "GROUP BY p.brand ORDER BY total_orders DESC;"
        ),
    },
]


# ─────────────────────────────────────────────
# 3.  Embedding model  (384-dim)
# ─────────────────────────────────────────────

_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def embed(text: str) -> list[float]:
    """Return a 384-dim embedding as a plain Python list."""
    vec = get_model().encode(text, normalize_embeddings=True)
    return vec.tolist()


# ─────────────────────────────────────────────
# 4.  One-time setup: create table + insert rules
# ─────────────────────────────────────────────

SETUP_SQL = """
-- Enable pgvector extension (run once per database)
CREATE EXTENSION IF NOT EXISTS vector;

-- Business rules table
CREATE TABLE IF NOT EXISTS business_rules (
    id          TEXT PRIMARY KEY,
    tier        TEXT NOT NULL,
    rule        TEXT NOT NULL,
    example_sql TEXT,
    embedding   vector(384)         -- pgvector column
);

-- IVFFlat index for fast approximate nearest-neighbour search
CREATE INDEX IF NOT EXISTS business_rules_embedding_idx
    ON business_rules
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 5);
"""


def setup_pgvector_table():
    """Create the pgvector table and index if they don't exist."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(SETUP_SQL)
    conn.commit()
    cur.close()
    conn.close()
    print("✅ pgvector table + index created (or already exists).")


def seed_business_rules(force: bool = False):
    """
    Embed each business rule and upsert it into the pgvector table.
    Pass force=True to re-embed and overwrite existing rows.
    """
    conn = get_connection()
    cur = conn.cursor()

    for rule_doc in BUSINESS_RULES:
        if not force:
            cur.execute("SELECT 1 FROM business_rules WHERE id = %s", (rule_doc["id"],))
            if cur.fetchone():
                print(f"  ⏭  Skipping {rule_doc['id']} (already seeded)")
                continue

        # Embed the rule text (the part the retriever searches over)
        vector = embed(rule_doc["rule"])

        cur.execute(
            """
            INSERT INTO business_rules (id, tier, rule, example_sql, embedding)
            VALUES (%s, %s, %s, %s, %s::vector)
            ON CONFLICT (id) DO UPDATE
                SET rule        = EXCLUDED.rule,
                    example_sql = EXCLUDED.example_sql,
                    embedding   = EXCLUDED.embedding
            """,
            (
                rule_doc["id"],
                rule_doc["tier"],
                rule_doc["rule"],
                rule_doc.get("example_sql", ""),
                json.dumps(vector),         # psycopg2 passes as string; pgvector parses it
            ),
        )
        print(f"  ✅ Seeded {rule_doc['id']} ({rule_doc['tier']})")

    conn.commit()
    cur.close()
    conn.close()
    print("Done seeding business rules into pgvector.")


# ─────────────────────────────────────────────
# 5.  Retrieval  (semantic search)
# ─────────────────────────────────────────────

def retrieve_business_rules(
    query: str,
    top_k: int = 3,
    min_similarity: float = 0.30,
) -> list[dict]:
    """
    Embed the user query and return the top-k most similar business rules
    from pgvector using cosine similarity.

    Returns a list of dicts with keys: id, tier, rule, example_sql, similarity
    """
    query_vec = embed(query)
    query_vec_str = json.dumps(query_vec)

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT
            id,
            tier,
            rule,
            example_sql,
            1 - (embedding <=> %s::vector) AS similarity
        FROM business_rules
        WHERE 1 - (embedding <=> %s::vector) >= %s
        ORDER BY embedding <=> %s::vector          -- cosine distance (ascending)
        LIMIT %s;
        """,
        (query_vec_str, query_vec_str, min_similarity, query_vec_str, top_k),
    )

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [
        {
            "id": r[0],
            "tier": r[1],
            "rule": r[2],
            "example_sql": r[3],
            "similarity": round(float(r[4]), 4),
        }
        for r in rows
    ]


# ─────────────────────────────────────────────
# 6.  Prompt builder  (RAG injection)
# ─────────────────────────────────────────────

def build_rag_prompt(user_question: str, schema_info: str, top_k: int = 3) -> str:
    """
    Retrieve relevant business rules and inject them into the SQL agent prompt.

    Args:
        user_question : natural language question from the user
        schema_info   : string describing the DB schema (table names, columns)
        top_k         : number of rules to retrieve

    Returns:
        A fully-formed prompt string ready to pass to the LLM agent.
    """
    retrieved = retrieve_business_rules(user_question, top_k=top_k)

    if not retrieved:
        context_block = "No specific business rules retrieved. Use general SQL best practices."
    else:
        lines = []
        for i, doc in enumerate(retrieved, 1):
            lines.append(f"Rule {i} [{doc['tier']}] (similarity={doc['similarity']}):")
            lines.append(f"  {doc['rule']}")
            if doc["example_sql"]:
                lines.append(f"  Example SQL: {doc['example_sql']}")
        context_block = "\n".join(lines)

    prompt = f"""You are a precise SQL generation agent. Your ONLY output must be a JSON object.

## Database Schema
{schema_info}

## Retrieved Business Rules (via pgvector semantic search)
{context_block}

## Constraints
- Generate SELECT-only SQL. No INSERT, UPDATE, DELETE, DROP, or DDL.
- Use COALESCE(discount, 0) when discount may be NULL.
- Use LEFT JOIN when combining tables unless an INNER JOIN is explicitly justified.
- Match string values exactly (case-sensitive) unless told otherwise.
- Always GROUP BY when using aggregate functions.

## User Question
{user_question}

## Required Output Format (JSON only, no markdown, no explanation)
{{
  "sql": "<your SQL query>",
  "metric": "<what this query computes>",
  "assumptions": ["<assumption 1>", "<assumption 2>"]
}}
"""
    return prompt


# ─────────────────────────────────────────────
# 7.  Quick smoke-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    SCHEMA = """
    products(product_id, product_name, brand, category_id, size, price, stock_quantity, discount)
    categories(category_id, category_name)
    orders(order_id, product_id, customer_id, quantity, order_date)
    """

    test_questions = [
        # Tier 1
        "If we sell all Levi's T-shirts with discounts applied, how much revenue will we generate?",
        # Tier 2
        "Show me all brands and how many orders each brand received, including brands with zero orders.",
        # Tier 3
        "Which brand has the highest total inventory value across all products?",
    ]

    print("\n" + "=" * 60)
    print("pgvector RAG — Retrieval Smoke Test")
    print("=" * 60)

    for q in test_questions:
        print(f"\nQuestion: {q}")
        results = retrieve_business_rules(q, top_k=2)
        for r in results:
            print(f"  [{r['similarity']:.3f}] {r['tier']}: {r['rule'][:80]}...")

    print("\n" + "=" * 60)
    print("Sample RAG Prompt (Tier 1 question):")
    print("=" * 60)
    print(build_rag_prompt(test_questions[0], SCHEMA))