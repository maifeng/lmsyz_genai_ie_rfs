# Batch extractors

The batch extractors handle the asynchronous batch API path. Use these when you have
thousands to millions of rows and can tolerate up to 24 hours of turnaround in exchange
for roughly 50% lower per-token cost. Both classes follow the same four-step lifecycle:
build requests, submit, poll status, retrieve results as a DataFrame.

`OpenAIBatchExtractor` uses the OpenAI Batch API (JSONL file upload). `AnthropicBatchExtractor`
uses Anthropic Message Batches (JSON body POST, no file upload). All intermediate files
are written to disk under `batch_root_dir/job_id/` so you can inspect what was sent and
what came back.

## OpenAIBatchExtractor

::: lmsyz_genai_ie_rfs.OpenAIBatchExtractor

## AnthropicBatchExtractor

::: lmsyz_genai_ie_rfs.AnthropicBatchExtractor

---

## See also

- [Run a batch job](../how-to/batch-jobs.md): end-to-end walkthrough for both providers with on-disk artifact inspection.
- [Architecture](../concepts/architecture.md): where the batch path fits in the library.
