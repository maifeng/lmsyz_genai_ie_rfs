# Change the prompt safely

## Problem

You edited your prompt. Will the library reuse your old results from the cache, or will
it re-run all the rows?

## Solution

By default, no reuse. Changing the prompt invalidates the cache for every row. The
library computes a hash of the prompt at the start of each run and only skips rows whose
cached entry was produced by the same hash. Change one character and every row
re-executes.

```python
import pandas as pd
from lmsyz_genai_ie_rfs import extract_df

df = pd.DataFrame({
    "id": [f"doc_{i}" for i in range(5)],
    "text": [
        "The team delivered the product on time.",
        "Management values transparency and ethics.",
        "Rapid iteration drives our innovation culture.",
        "Customer feedback shapes every roadmap decision.",
        "Performance targets are reviewed quarterly.",
    ],
})

CACHE = "runs/prompt_demo.sqlite"

prompt_v1 = """
For each row, copy input_id and classify as "positive", "neutral", or "negative".
Return {"all_results": [...]}.
"""

# Run 1: fills the cache with prompt_v1's hash.
out1 = extract_df(df, prompt=prompt_v1, cache_path=CACHE, backend="openai", model="gpt-4.1-mini")
print("Run 1:", len(out1), "rows")

prompt_v2 = """
For each row, copy input_id and classify as "positive", "neutral", or "negative".
Be conservative: prefer "neutral" when uncertain.
Return {"all_results": [...]}.
"""

# Run 2: different prompt hash -> all 5 rows re-execute.
out2 = extract_df(df, prompt=prompt_v2, cache_path=CACHE, backend="openai", model="gpt-4.1-mini")
print("Run 2:", len(out2), "rows (re-executed, new prompt)")
```

After both runs the SQLite file holds rows from both prompts. You can inspect them:

```python
import sqlite3

with sqlite3.connect(CACHE) as con:
    rows = con.execute(
        "SELECT prompt_hash, COUNT(*) FROM results GROUP BY prompt_hash"
    ).fetchall()

for hash_val, count in rows:
    print(f"  prompt_hash={hash_val}  rows={count}")
# prompt_hash=3c7a9f0d12e4b816  rows=5
# prompt_hash=9e2d0a5f71c3b48a  rows=5
```

Each prompt's results live at its own hash. They do not overwrite each other.

### The escape hatch: `ignore_prompt_hash=True`

Sometimes a prompt edit is non-semantic: you fixed a typo, adjusted whitespace, or
renamed a heading without changing the task. In that case you want to reuse the prior
rows without spending tokens. Pass `ignore_prompt_hash=True`:

```python
prompt_v2_typo_fixed = """
For each row, copy input_id and classify as "positive", "neutral", or "negative".
Be conservative: prefer "neutral" when uncertain.
Return {"all_results": [...]}.
"""

out3 = extract_df(
    df,
    prompt=prompt_v2_typo_fixed,
    cache_path=CACHE,
    model="gpt-4.1-mini",
    ignore_prompt_hash=True,   # reuse cached rows regardless of which prompt produced them
)
print("Run 3:", len(out3), "rows (served from cache)")
```

When `ignore_prompt_hash=True`, `all_ids()` is called without a hash filter, so any
cached row is eligible for reuse. New results produced in this run are still written with
the current prompt's hash, so future runs without the flag will see the new hash correctly.

Use this flag deliberately. It is intended for the case where you are confident the
semantic meaning of the prompt did not change.

## Explanation

The prompt hash is computed once per `extract_df` call using SHA-256 truncated to 16 hex
characters. It is stable across Python versions and machines because it operates on the
UTF-8 byte representation of the prompt string:

```python
import hashlib
hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
```

On a cache lookup, the library calls `SqliteCache.all_ids(prompt_hash=current_hash)`.
Only IDs whose stored `prompt_hash` matches are skipped. All other IDs are added to the
working set and sent to the API.

This means the same `cache_path` can hold results from multiple prompt versions without
any of them interfering. The SQLite file is a log, not a single-version store. You can
query by `prompt_hash` to compare results across prompt versions (see the SQL snippet
above).

### Legacy caches

If you have a SQLite file that was created before the `prompt_hash` column existed, the
library migrates it automatically with `ALTER TABLE ... ADD COLUMN prompt_hash TEXT`. The
migrated rows have `NULL` for the hash and are treated as not matching any specific
prompt, so they will be re-processed on the next run. This is the safe default.

### Comparing two prompt versions side by side

Use separate `cache_path` files per prompt for clean separation:

```python
out_v1 = extract_df(df, prompt=prompt_v1, cache_path="runs/v1.sqlite", backend="openai", model="gpt-4.1-mini")
out_v2 = extract_df(df, prompt=prompt_v2, cache_path="runs/v2.sqlite", backend="openai", model="gpt-4.1-mini")

merged = out_v1.merge(out_v2, on="input_id", suffixes=("_v1", "_v2"))
print(merged[["input_id", "sentiment_v1", "sentiment_v2"]])
```

Or keep them in one file and query by hash:

```sql
SELECT
    a.row_id,
    json_extract(a.json_result, '$.sentiment') AS sentiment_v1,
    json_extract(b.json_result, '$.sentiment') AS sentiment_v2
FROM results a
JOIN results b ON a.row_id = b.row_id
WHERE a.prompt_hash = '3c7a9f0d12e4b816'
  AND b.prompt_hash = '9e2d0a5f71c3b48a';
```

## Related

- [Resume after a crash](resume-after-crash.md): how the same cache gating enables
  crash recovery.
- [Inspect the results database](inspect-results-db.md): querying the SQLite file
  directly, including grouping by `prompt_hash`.
- [Reference: `extract_df`](../reference/extract_df.md)
- [Concepts: results database](../concepts/results-db.md)
