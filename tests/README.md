# Testing & Debugging Tools

This folder provides standalone scripts to inspect and test the **Extraction**, **Chunking**, and **Azure Hybrid Retrieval** without needing to open the Streamlit UI or run chat completions.

---

## 1. Test Text Extraction & Chunks

Inspect raw text extracted by `extractor.py` and the chunks created by `chunker.py`.

### Test a single file:
```bash
python tests/test_extraction.py data/Abdelrahman_Karawia_Flutter_CV.pdf
```

### Save full output to `tests/output/`:
```bash
python tests/test_extraction.py data/Abdelrahman_Karawia_Flutter_CV.pdf --save
```
*(Saves `{name}_extracted.txt` and `{name}_chunks.txt` for easy reading)*

### Test all CVs in `data/`:
```bash
python tests/test_extraction.py --all
```

---

## 2. Test Retrieval & View Chunks

Run hybrid search (BM25 + Dense Vector HNSW) directly against Azure AI Search to see exactly which chunks match your queries.

### Run a specific query:
```bash
python tests/test_retrieval.py "Who has experience with Flutter and mobile apps?" -k 5
```

### Save retrieved chunks to file:
```bash
python tests/test_retrieval.py "Python backend Django" -k 5 --save
```

### Interactive query mode:
```bash
python tests/test_retrieval.py
```
*(Prompts for queries interactively so you can test multiple queries quickly)*

---

## 3. Test FlashRank Re-ranker

Benchmark and verify FlashRank's cross-encoder re-ranking scores and ordering:

```bash
python tests/test_reranker.py
```

---

## 4. Test Redis Semantic Cache & Vector Similarity

Verify that Redis cache operations, cosine similarity calculations, and threshold evaluations work correctly:

```bash
python tests/test_semantic_cache.py
```

### Quick Semantic Cache Hit Demo:
```bash
python tests/demo_semantic_hit.py
```

---

## 5. Test Hallucination & Faithfulness Auditor

Verify the claim-level grounding auditor on grounded vs fabricated candidate statements:

```bash
python tests/test_hallucination.py
```

