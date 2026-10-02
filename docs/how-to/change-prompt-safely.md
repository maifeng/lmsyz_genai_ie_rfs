# Change the prompt safely

Changing the prompt causes a cache miss for rows stored under a different prompt hash.
A successful new result overwrites the earlier result for the same row ID. The database
stores one current result per ID, so use separate cache files to preserve experiments.

## Reprocess after a prompt edit

```python
from pathlib import Path
import pandas as pd
from lmsyz_genai_ie_rfs import extract_df

Path("runs").mkdir(parents=True, exist_ok=True)
df = pd.DataFrame({
    "id": ["doc_1", "doc_2"],
    "text": ["The team delivered on time.", "Management ignored feedback."],
})

prompt_v1 = '''
For each row, copy input_id verbatim and return sentiment as
"positive", "neutral", or "negative".
Return a JSON object with key "all_results" containing one object per input row.
'''
prompt_v2 = prompt_v1 + '\nPrefer "neutral" when the evidence is ambiguous.'

CACHE = "runs/prompt_demo.sqlite"
out1 = extract_df(df, prompt=prompt_v1, cache_path=CACHE, model="gpt-4.1-mini")
out2 = extract_df(df, prompt=prompt_v2, cache_path=CACHE, model="gpt-4.1-mini")
```

After both runs succeed, the cache contains two rows with `prompt_v2`'s hash. It does
not retain the earlier versions. If part of the second run fails, the database can
contain a mixture: updated IDs under the new hash and older IDs under the old hash.
Lookups for `prompt_v2` skip only rows whose stored hash matches it.

```python
import sqlite3

with sqlite3.connect(CACHE) as con:
    counts = con.execute(
        "SELECT prompt_hash, COUNT(*) FROM results GROUP BY prompt_hash"
    ).fetchall()
print(counts)
```

The hash is `sha256(prompt.encode("utf-8")).hexdigest()[:16]`. Whitespace changes
normally change it. The cache does not automatically invalidate when the text, model,
provider, endpoint, or schema changes while IDs and prompt remain the same.

## Compare experiments

Use one cache per prompt/model/schema/corpus combination:

```python
out_v1 = extract_df(df, prompt=prompt_v1, cache_path="runs/v1.sqlite", model="gpt-4.1-mini")
out_v2 = extract_df(df, prompt=prompt_v2, cache_path="runs/v2.sqlite", model="gpt-4.1-mini")

merged = out_v1.merge(out_v2, on="input_id", suffixes=("_v1", "_v2"))
print(merged[["input_id", "sentiment_v1", "sentiment_v2"]])
```

This preserves both results for each observation. A self-join within one cache cannot
recover overwritten versions. Back up existing files before reusing them if needed.

## Reuse after a non-semantic edit

If you intentionally want to keep results after a typo or formatting correction,
pass `ignore_prompt_hash=True`:

```python
out3 = extract_df(
    df, prompt=prompt_v2, cache_path=CACHE, model="gpt-4.1-mini",
    ignore_prompt_hash=True,
)
```

Existing rows keep their original hashes. Only newly processed rows are stamped with
the current prompt hash. A later call without `ignore_prompt_hash=True` resumes normal
hash filtering.

## Force a refresh

`fresh=True` bypasses cache reads for that call and processes all input rows.
Successful rows overwrite earlier results. Failed rows leave previous entries intact,
and a later ordinary call can reuse those entries when their hashes match. To start
with an empty database, choose a new path or explicitly delete the old file.

## Legacy caches

Caches without a `prompt_hash` column are migrated automatically. Their existing rows
have a null hash and are reprocessed on a hash-filtered lookup. Passing
`ignore_prompt_hash=True` deliberately permits their reuse.

## Related

- [Resume after a crash](resume-after-crash.md)
- [Inspect the results database](inspect-results-db.md)
- [Results database](../concepts/results-db.md)
