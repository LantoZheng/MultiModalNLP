# MultiModalNLP

An analysis scaffold for exploring performance gaps between language-only (LLM),
vision-language (VLM), and audio-language (ALM) model families. The project
compares benchmark predictions; it does not run or download models.

## Compare benchmark results

Create a UTF-8 JSON Lines file with one result per line. Each record must include
the benchmark task identifier, model family, expected answer, and model
prediction:

```json
{"task_id":"image_qa","model_family":"LLM","expected":"cat","prediction":"cat"}
{"task_id":"image_qa","model_family":"VLM","expected":"cat","prediction":"dog"}
{"task_id":"audio_qa","model_family":"LLM","expected":"yes","prediction":"no"}
{"task_id":"audio_qa","model_family":"ALM","expected":"yes","prediction":"yes"}
```

Run the analyzer with Python 3:

```bash
python multimodal_gap.py results.jsonl
```

The JSON report includes pooled exact-match accuracy by family, per-task
accuracy, and each multimodal family's mean accuracy difference from LLMs on
tasks represented by both families. Differences are fractions (for example,
`0.1` is 10 percentage points); task-level differences are averaged equally so
that a larger task does not dominate the gap calculation. Answers are compared
after trimming surrounding whitespace and ignoring case.

Run the tests with:

```bash
python -m unittest discover -s tests
```
