# Inspect the results database

## Problem

You want to look at what is in the SQLite results database: how many rows completed, what
the raw JSON looks like, and whether any rows from your input DataFrame are missing.

## Solution

### From the command line

```bash
# Count completed rows.
sqlite3 runs/my_results.sqlite "SELECT COUNT(*) FROM results;"

# See the schema.
sqlite3 runs/my_results.sqlite ".schema results"

# Peek at the first five rows.
sqlite3 runs/my_results.sqlite "SELECT row_id, prompt_hash, json_result FROM results LIMIT 5;"
```

### From Python

```python
import sqlite3
import json
import pandas as pd

DB = "runs/my_results.sqlite"

with sqlite3.connect(DB) as con:
    # Count rows.
    n = con.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    print(f"Rows in DB: {n}")

    # Peek at one row.
    row = con.execute(
        "SELECT row_id, prompt_hash, json_result FROM results LIMIT 1"
    ).fetchone()
    if row:
        print("row_id     :", row[0])
        print("prompt_hash:", row[1])
        print("json_result:", json.loads(row[2]))
```

## Explanation

### Schema

See [the schema in concepts/results-db.md](../concepts/results-db.md#schema) for the
full DDL and column descriptions.

`row_id` is the string form of the value in `id_col`. `json_result` is the row dict
returned by the model, serialised as JSON. `prompt_hash` is a 16-character SHA-256
digest of the prompt that produced this row. Rows migrated from a pre-hash cache have
`NULL` for the hash.

### Useful SQL queries

```sql
-- Count rows.
SELECT COUNT(*) FROM results;

-- Group by prompt hash to see results from multiple prompt versions.
SELECT prompt_hash, COUNT(*) AS n_rows
FROM results
GROUP BY prompt_hash
ORDER BY n_rows DESC;

-- Parse a JSON field inline (SQLite json_extract).
SELECT row_id,
       json_extract(json_result, '$.culture_type') AS culture_type,
       json_extract(json_result, '$.tone')         AS tone,
       json_extract(json_result, '$.confidence')   AS confidence
FROM results
LIMIT 10;

-- Find rows with a specific value.
SELECT row_id, json_result
FROM results
WHERE json_extract(json_result, '$.tone') = 'negative';
```

### Loading the cache into a DataFrame

**Option A: via `SqliteCache`**

```python
import pandas as pd
from pathlib import Path
from lmsyz_genai_ie_rfs import SqliteCache

cache = SqliteCache(Path("runs/my_results.sqlite"))

# Get all IDs stored under a specific prompt hash (or all IDs if hash is None).
all_ids = cache.all_ids(prompt_hash=None)

rows = []
for rid in all_ids:
    result = cache.get(rid, prompt_hash=None)
    if result is not None:
        rows.append(result)

out = pd.DataFrame(rows)
print(out.head())
```

**Option B: via `pandas.read_sql` (faster for large caches)**

```python
import sqlite3
import json
import pandas as pd

with sqlite3.connect("runs/my_results.sqlite") as con:
    raw = pd.read_sql("SELECT row_id, json_result FROM results", con)

# Expand the JSON column into full DataFrame columns.
out = pd.json_normalize(raw["json_result"].apply(json.loads))
print(out.head())
```

### Finding missing rows

Use this pattern to identify which input IDs did not make it into the cache, which
indicates a chunk failure:

```python
import sqlite3
import pandas as pd

input_df = pd.read_csv("my_corpus.csv")   # your original input DataFrame
DB = "runs/my_results.sqlite"

with sqlite3.connect(DB) as con:
    cached_ids = {
        r[0] for r in con.execute("SELECT row_id FROM results").fetchall()
    }

input_ids = set(input_df["id"].astype(str))
missing = input_ids - cached_ids

print(f"Input rows  : {len(input_ids)}")
print(f"Cached rows : {len(cached_ids)}")
print(f"Missing rows: {len(missing)}")

if missing:
    print("Sample missing IDs:", list(missing)[:10])
    missing_df = input_df[input_df["id"].astype(str).isin(missing)]
    missing_df.to_csv("runs/missing_rows.csv", index=False)
```

### Computing the success rate

```python
import sqlite3
import pandas as pd

input_df = pd.read_csv("my_corpus.csv")

with sqlite3.connect("runs/my_results.sqlite") as con:
    cached = con.execute("SELECT COUNT(*) FROM results").fetchone()[0]

total = len(input_df)
print(f"Success rate: {cached}/{total} = {100 * cached / total:.1f}%")
```

### Joining results back to the input DataFrame

```python
import sqlite3
import json
import pandas as pd

input_df = pd.read_csv("my_corpus.csv")

with sqlite3.connect("runs/my_results.sqlite") as con:
    raw = pd.read_sql("SELECT row_id, json_result FROM results", con)

results_df = pd.json_normalize(raw["json_result"].apply(json.loads))

# Join on the shared identifier.
# (The model copies input_id into the result; row_id is the cache key.)
joined = input_df.merge(
    results_df,
    left_on="id",
    right_on="input_id",
    how="left",
)
print(joined.head())

n_matched = joined["input_id"].notna().sum()
print(f"Matched {n_matched} / {len(input_df)} rows")
```

### Chunk-level vs row-level failures

All rows in a chunk share the same fate: if a chunk fails, all its rows are absent from
the cache. The log line looks like:

```
extract_df: chunk failed; results for this chunk skipped.
```

followed by the exception traceback. A chunk typically contains `chunk_size` rows
(default 5). If you see a cluster of consecutive missing IDs of size 5, a single chunk
failed. Retrying the full job will re-send only the missing chunks.

If a chunk fails repeatedly (it will be retried up to 5 times with exponential backoff),
the rows stay absent. Narrow the problem by running those specific rows in isolation:

```python
retry_df = input_df[input_df["id"].astype(str).isin(missing)]

out_retry = extract_df(
    retry_df,
    prompt=prompt,
    cache_path="runs/my_results.sqlite",   # same cache: successful retries fill the gap
    model="gpt-4.1-mini",
)
```

## Related

- [Resume after a crash](resume-after-crash.md): the cache gating mechanism that makes
  partial results resumable.
- [Change the prompt safely](change-prompt-safely.md): how `prompt_hash` is used to gate
  cache lookups.
- [Reference: `SqliteCache`](../reference/sqlite-cache.md)
- [Concepts: results database](../concepts/results-db.md)
