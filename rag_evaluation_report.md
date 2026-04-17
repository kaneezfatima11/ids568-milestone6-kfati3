# RAG Evaluation Report
## IDS568 Milestone 6 - Part 1
**Author:** Kaneez Fatima | **NetID:** kfati3

---

## Experimental Setup

| Parameter | Value |
|---|---|
| Embedding Model | all-MiniLM-L6-v2 |
| Vector Store | FAISS IndexFlatL2 |
| Chunk Size | 100 words |
| Chunk Overlap | 20 words |
| Top-k Retrieval | k=3 |
| LLM | Qwen2.5-7B-Instruct |
| Serving Method | HuggingFace Transformers |
| Hardware | Tesla T4 GPU (15GB VRAM) |
| Corpus Size | 10 documents, 10 chunks |

---

## Retrieval Accuracy on 10 Queries

| Q1 | What is RAG and how does it work? | doc1, doc8 | doc8, doc1, doc5 | 0.67 | 1.00 | 1.00 |
| Q2 | What are the different types of FAISS indexes? | doc2 | doc2, doc8, doc1 | 0.33 | 1.00 | 1.00 |
| Q3 | How do you choose chunk size and overlap? | doc3 | doc3, doc1, doc8 | 0.33 | 1.00 | 1.00 |
| Q4 | What embedding models are available? | doc4 | doc4, doc10, doc2 | 0.33 | 1.00 | 1.00 |
| Q5 | What causes hallucination in LLMs? | doc5, doc1 | doc5, doc8, doc1 | 0.67 | 1.00 | 1.00 |
| Q6 | What is the ReAct pattern in agentic AI? | doc6 | doc6, doc1, doc8 | 0.33 | 1.00 | 1.00 |
| Q7 | What is MLOps and its key components? | doc7 | doc7, doc10, doc6 | 0.33 | 1.00 | 1.00 |
| Q8 | How is retrieval quality measured in RAG? | doc8 | doc8, doc1, doc8 | 0.33 | 1.00 | 1.00 |
| Q9 | What prompt engineering techniques work for RAG? | doc9 | doc9, doc5, doc1 | 0.33 | 1.00 | 1.00 |
| Q10 | Differences between vLLM and HF Transformers? | doc10 | doc10, doc7, doc5 | 0.33 | 1.00 | 1.00 |

---

## Latency Measurements

| Stage | Measurement |
|---|---|
| Embedding model load time | 3.60s |
| Embedding generation (10 chunks) | 0.993s |
| FAISS index build time | 0.005s |
| Retrieval latency per query | 10-25ms |
| Generation latency per query | 30-130s |
| End-to-end latency per query | 30-130s |

---

## Qualitative Grounding Analysis

### Grounding Observations
The Qwen2.5-7B-Instruct model demonstrates strong grounding behavior when provided with relevant context. In the test query "What is RAG and how does it work?", the model produced a detailed response that accurately reflected the content of the retrieved chunks without introducing fabricated information.

### Hallucination Cases
- **Query Q1:** Response was well grounded in retrieved context with no hallucinations detected
- **Edge cases:** When retrieved chunks are not directly relevant to the query, the model occasionally supplements with parametric knowledge rather than stating insufficient context

### Retrieval Failures vs Generation Failures
- **Retrieval failures:** Occur when the query terms do not semantically match the relevant document embeddings. For example, paraphrased queries retrieve different chunks than expected.
- **Generation failures:** Occur when the model ignores the instruction to only use provided context and draws on parametric knowledge instead.

---

## Chunking and Indexing Design Decisions

### Chunk Size (100 words)
A chunk size of 100 words was chosen to balance semantic coherence and retrieval precision. Smaller chunks (50 words) were too fragmented to contain complete concepts. Larger chunks (200+ words) diluted the embedding signal by mixing multiple topics.

### Chunk Overlap (20 words)
20-word overlap (20% of chunk size) ensures that concepts spanning chunk boundaries are not lost. This is particularly important for technical definitions that span multiple sentences.

### Embedding Model (all-MiniLM-L6-v2)
Selected for its balance of speed and quality on the T4 GPU. Produces 384-dimensional embeddings which are compact enough for fast FAISS search while capturing sufficient semantic meaning.

### FAISS IndexFlatL2
Chosen for exact nearest neighbor search on a small corpus of 10 chunks. For larger corpora (>10,000 chunks), IVF or HNSW indexes would be more appropriate for sub-linear search time.

### Top-k=3
Retrieving 3 chunks provides sufficient context for most queries without exceeding the LLM context window or introducing irrelevant information.

---

## Error Attribution

### Retrieval Errors
- Semantic mismatch between query phrasing and document content
- Small corpus limits diversity of retrievable context
- Single embedding per chunk may miss nuanced query aspects

### Generation Errors
- Model occasionally draws on parametric knowledge despite instructions
- Long generation times (30-130s) due to T4 GPU memory constraints
- Some parameters offloaded to CPU causing inference slowdown

---

## Summary Statistics
*(To be updated after evaluation queries complete)*

| Metric | Value |
|---|---|
| Average P@3 | 0.400 |
| Average R@3 | 1.000 |
| Average MRR | 1.000 |
| Average Retrieval Latency | 10.7ms |
| Average Generation Latency | 46.0s |
| Average End-to-End Latency | 46.0s |