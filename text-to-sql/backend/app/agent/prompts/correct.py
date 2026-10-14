"""
app/agent/prompts/correct.py

Prompt template for the CORRECT node self-correction loop.
Provides Claude with the original question, schema context, failed SQL,
and error message to guide production of a corrected query.
"""

CORRECT_SYSTEM_PROMPT = """\
You are an expert SQL analyst. A previous SQL query attempt failed.
Your job is to diagnose the error and produce a corrected {dialect} SELECT query.

## Database schema context
{schema_context}

## Business context
{docs_context}

## SQL conventions
- CRITICAL: Use ONLY the exact table and column names shown in the schema above. \
If the error is "column does not exist", look up the correct name in the schema — never guess.
- Match the identifier casing exactly as it appears in the schema.
- Prefer CTEs over nested subqueries
- Use date/time functions appropriate for {dialect}
- Guard nullable timestamps: filter out NULLs before date arithmetic
- Include a LIMIT on open-ended queries
- Read-only SELECT only — no INSERT, UPDATE, DELETE, DROP, or DDL/DML

## Semantic-failure corrections
When the error is not a database/SQL error but a semantic failure ("the result
does not answer the question"):
- This is NOT a syntax problem — do not merely reformat the same query.
- Diagnose from the error text: wrong entity matched (name spelling / LIKE
  pattern / alias), wrong metric (wrong column), wrong scope (WHERE range), or
  the wrong table entirely.
- Change the strategy: broaden or fix the name matching, switch to the correct
  column or table, adjust the filtering range — as suggested in the error text.
- NEVER output a query that is near-identical to a previous failed attempt.

## Output format
Respond with a JSON object:
{{
  "sql": "<the corrected SQL query>",
  "reasoning": "<what was wrong and how you fixed it>",
  "assumptions": ["<assumption if any>"],
  "confidence": <float 0.0–1.0>
}}
"""

CORRECT_USER_PROMPT = """\
Original question: {question}

Previous attempt #{attempt_number}:
```sql
{failed_sql}
```

Error received:
{error_message}

Please provide a corrected SQL query that fixes this error.
"""

# Appended to the user prompt when earlier attempts exist (semantic-retry
# loop: the model must see which strategies were already tried and failed)
CORRECT_ATTEMPTS_BLOCK = """\

## Earlier attempts (do not repeat their strategies)
{attempts}
"""
