# Retrieval Results

This project includes a lightweight offline benchmark for text, image, and multimodal shopping queries.

Evaluation entrypoints:

- [eval/queries.jsonl](../eval/queries.jsonl)
- [eval/eval_retrieval.py](../eval/eval_retrieval.py)

## Summary

The current retrieval stack performs strongly across all three modes, with the best gains showing up in multimodal intent-shift queries such as starting from a formal shoe image and asking for something better suited to running.

The following results come from `make eval` on the current 18-query benchmark set.

| Metric | Score |
| --- | ---: |
| Strict Hit@3 | 0.944 |
| Strict Hit@5 | 0.944 |
| Strict MRR | 0.889 |
| Semantic Category Hit@3 | 1.000 |
| Semantic Subcategory Hit@3 | 1.000 |
| Semantic Keyword Hit@3 | 0.722 |
| Average Latency (ms) | 3925.93 |

## By Modality

| Mode | Queries | Strict Hit@3 | Strict MRR | Semantic Subcategory Hit@3 | Avg Latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Text | 8 | 0.875 | 0.750 | 1.000 | 696.04 |
| Image | 5 | 1.000 | 1.000 | 1.000 | 11447.26 |
| Multimodal | 5 | 1.000 | 1.000 | 1.000 | 1572.44 |

## What Improved

- First-stage candidate generation now uses stronger stopword filtering and intent-aware seed candidates, which makes it much easier for the system to recover from misleading source images.
- Second-stage reranking now combines hybrid scores with query-intent alignment, lightweight attribute matching, and optional LLM reranking for high-value cases.
- Multimodal retrieval is now stable on hard shifts such as `better for running` and `better for basketball`, where the source image alone would otherwise bias the system toward the wrong footwear type.

## Reading The Metrics

- `Strict` metrics use a hand-authored set of positive product IDs for each query.
- `Semantic` metrics check whether the returned results match the expected category, subcategory, or keyword intent even if the exact IDs differ.
- Image latency includes cold-start CLIP loading, so the first image query is much slower than warmed-up runs.

## Interpretation

- The benchmark is intentionally small and human-authored, so it functions best as a regression harness and portfolio artifact rather than a final large-scale evaluation suite.
- The remaining strict miss is a broad sandals query where the retrieval system still returns the correct subcategory, but the labeled positives are narrower than the semantic target.
- In practice, the current benchmark shows that the system is strongest on image and multimodal retrieval, while text retrieval is solid and still has room for finer-grained ranking improvements on broader shopping queries.
