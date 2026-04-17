# RAG Pipeline Architecture Diagram

## Data Flow
┌─────────────────────────────────────────────────────────┐
│                    INDEXING PHASE (Offline)              │
│                                                          │
│  Documents → Chunker → Embedder → FAISS Vector Store    │
│  (10 docs)   (100w      (all-        (384-dim            │
│              chunks)    MiniLM)      IndexFlatL2)        │
└─────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────┐
│                    QUERY PHASE (Online)                  │
│                                                          │
│  User Query                                             │
│      │                                                   │
│      ▼                                                   │
│  Query Embedder (all-MiniLM-L6-v2)                      │
│      │                                                   │
│      ▼                                                   │
│  FAISS Similarity Search (top-k=3)                      │
│      │                                                   │
│      ▼                                                   │
│  Retrieved Chunks + Context Assembly                     │
│      │                                                   │
│      ▼                                                   │
│  Grounded Prompt Construction                           │
│      │                                                   │
│      ▼                                                   │
│  Qwen2.5-7B-Instruct (HF Transformers)                  │
│      │                                                   │
│      ▼                                                   │
│  Grounded Response with Sources                         │
└─────────────────────────────────────────────────────────┘
## Key Components

| Component | Implementation | Purpose |
|---|---|---|
| Document Loader | Python dict corpus | Ingests 10 domain documents |
| Text Chunker | Custom word-based splitter | Splits docs into 100-word chunks with 20-word overlap |
| Embedding Model | all-MiniLM-L6-v2 (384-dim) | Converts text to semantic vectors |
| Vector Store | FAISS IndexFlatL2 | Stores and searches embeddings |
| Retriever | Top-k similarity search | Finds k=3 most relevant chunks |
| LLM Generator | Qwen2.5-7B-Instruct | Generates grounded responses |

## Design Decisions

- **Chunk size 100 words** with 20-word overlap preserves context across boundaries
- **all-MiniLM-L6-v2** chosen for speed and quality balance on T4 GPU
- **FAISS IndexFlatL2** chosen for exact search on small corpus
- **k=3** retrieval balances context richness vs prompt length
- **Qwen2.5-7B-Instruct** chosen as approved 7B model that fits in 15GB VRAM

## Latency Profile

| Stage | Typical Latency |
|---|---|
| Embedding generation | ~1ms per query |
| FAISS search | ~0.1ms per query |
| Total retrieval | ~10-25ms |
| LLM generation | ~30-130s |
| End-to-end | ~30-130s |