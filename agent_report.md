# Agent Controller Report
## IDS568 Milestone 6 - Part 2
**Author:** Kaneez Fatima | **NetID:** kfati3

---

## Agent Architecture Overview

The agent controller implements a multi-tool RAG agent that intelligently
selects between a RetrieverTool and a SummarizerTool based on query intent.
The agent uses a keyword-based heuristic router combined with an LLM for
grounded response generation.

### Tools Implemented

| Tool | Purpose | Trigger Keywords |
|---|---|---|
| RetrieverTool | Retrieves relevant chunks and generates grounded answer | what, how, explain, describe, define, compare, why |
| SummarizerTool | Retrieves context and summarizes it concisely | summarize, summary, brief, tldr, condense, overview |

---

## Tool Selection Policy

The agent uses a two-layer routing strategy:

**Layer 1 - Keyword Heuristic:**
- Scans query for summarization keywords → routes to SummarizerTool
- Scans query for retrieval/question keywords → routes to RetrieverTool
- Default fallback → RetrieverTool

**Layer 2 - LLM Generation:**
- Once tool is selected, LLM generates response using retrieved context
- For RetrieverTool: grounded prompt with strict context-only instruction
- For SummarizerTool: summarization prompt over retrieved chunks

**Why this approach:**
- Fast routing (no LLM call needed for routing decision)
- Deterministic and observable tool selection
- Easy to debug and extend with new tools

---

## Retrieval Integration

The retrieval component is directly reused from Part 1:
- Same FAISS IndexFlatL2 index
- Same all-MiniLM-L6-v2 embedding model
- Same document corpus (10 documents)
- k=3 for retriever tool, k=2 for summarizer tool

The retriever is triggered as the first action for every task,
ensuring all responses are grounded in retrieved evidence.

---

## Performance Analysis on 10 Tasks

| Task | Query | Tool Selected | Success | Actual Time |
|---|---|---|---|---|
| 1 | What is RAG and how does it work? | retriever | ✅ | 124.6s |
| 2 | Summarize key concepts of vector databases | summarizer | ✅ | 64.1s |
| 3 | What are the different document chunking strategies? | retriever | ✅ | 14.4s |
| 4 | How do embedding models capture semantic meaning? | retriever | ✅ | 10.0s |
| 5 | What causes hallucination in LLMs and how can it be mitigated? | retriever | ✅ | 54.3s |
| 6 | Explain the ReAct pattern in agentic AI systems | retriever | ✅ | 39.4s |
| 7 | What are the key components of an MLOps pipeline? | retriever | ✅ | 18.3s |
| 8 | How is retrieval quality measured in RAG systems? | retriever | ✅ | 46.9s |
| 9 | What prompt engineering techniques work best for RAG? | retriever | ✅ | 18.3s |
| 10 | What is the difference between vLLM and HF Transformers? | retriever | ✅ | 21.6s |

**Average task time: 41.2s | Tool accuracy: 10/10 correct routing**
---

## Failure Analysis

### Case 1: Wrong Tool Selection
- **Scenario:** Ambiguous queries that contain both summarization and
  retrieval keywords
- **Example:** "Give me a brief explanation of RAG" - contains "brief"
  (summarizer keyword) but is really a retrieval question
- **Impact:** Response is still correct but may be shorter than optimal
- **Mitigation:** Add query length and intent analysis to router

### Case 2: Retrieval Miss
- **Scenario:** Query phrasing does not semantically match document content
- **Example:** Using technical jargon that differs from document vocabulary
- **Impact:** Wrong chunks retrieved, grounding fails
- **Mitigation:** Query expansion or HyDE retrieval

### Case 3: Context Insufficiency
- **Scenario:** Query requires information not present in the 10-document corpus
- **Example:** Asking about specific model benchmarks not in documents
- **Impact:** Model either says insufficient context or hallucinates
- **Mitigation:** Expand corpus or implement web search tool

### Case 4: Generation Latency
- **Scenario:** T4 GPU memory constraints cause slow generation
- **Impact:** Each task takes 30-130 seconds
- **Mitigation:** Use vLLM with matching CUDA version or quantized model

---

## Model Quality Analysis

**Model:** Qwen2.5-7B-Instruct via HuggingFace Transformers

**Strengths:**
- Strong instruction following - respects context-only constraint
- Good grounding - responses closely follow retrieved content
- Clear and structured responses

**Weaknesses:**
- Slow inference on T4 GPU (30-130s per query)
- Some parameters offloaded to CPU due to VRAM constraints
- Occasionally supplements with parametric knowledge

**Latency breakdown:**
- Model load time: ~87s (one time)
- Per-query generation: 30-130s
- Per-query retrieval: 10-25ms

---

## Design Reflection

The agent successfully demonstrates intelligent tool coordination with
observable decision traces. The keyword-based router provides fast,
deterministic routing that is easy to debug. The main limitation is
generation latency due to hardware constraints.

For production use, the routing layer would benefit from an LLM-based
classifier for more accurate tool selection. The corpus would need to be
expanded significantly for real-world use cases.