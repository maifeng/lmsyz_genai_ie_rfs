# genai_batch_ie_rfs

Concurrent and batch LLM calls on a pandas DataFrame with Pydantic-typed outputs.

Originally developed for Li, Mai, Shen, Yang & Zhang (2026),
"Dissecting Corporate Culture Using Generative AI,"
*Review of Financial Studies* 39(1):253-296.
https://doi.org/10.1093/rfs/hhaf081

---

## Install

```python
# In a Colab cell:
!pip install genai_batch_ie_rfs
```

## Quick example

```python
import pandas as pd
from genai_batch_ie_rfs import LLMClient
from genai_batch_ie_rfs.schema import CultureRow

df = pd.DataFrame({"id": [1, 2], "text": ["They emphasize transparency.", "Speed wins here."]})
results = LLMClient(backend="openai").classify_df(df, schema=CultureRow, prompt="Extract culture type and tone.")
print(results)
```

## License

MIT. See [LICENSE](LICENSE).
