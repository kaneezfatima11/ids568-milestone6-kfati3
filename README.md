# IDS 568 Milestone 6: RAG & Agentic Pipeline
**Author:** Kaneez Fatima | **NetID:** kfati3

---

## Overview
This repository implements a complete Retrieval-Augmented Generation (RAG) pipeline and a multi-tool agent controller as required by Milestone 6 of the MLOps course.

---

## Model Information
| Detail | Value |
|---|---|
| Model Name | Qwen/Qwen2.5-7B-Instruct |
| Size Class | 7B parameters |
| Serving Method | Hugging Face Transformers |
| Hardware | Tesla T4 GPU (15GB VRAM) |
| Precision | float16 |
| Typical Generation Latency | 30-130 seconds per query |

---

## Repository Structure

```
ids568-milestone6-kfati3/
├── rag_pipeline.ipynb          # Part 1: RAG implementation
├── agent_controller.py         # Part 2: Agent implementation
├── rag_evaluation_report.md    # Part 1: Evaluation report
├── rag_pipeline_diagram.md     # Part 1: Pipeline diagram
├── agent_report.md             # Part 2: Agent analysis
├── agent_traces/               # Part 2: 10 task traces
├── requirements.txt            # Pinned dependencies
└── README.md                   # This file
```
---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/kaneezfatima11/ids568-milestone6-kfati3.git
cd ids568-milestone6-kfati3
```

### 2. Create and activate virtual environment
```bash
python3 -m venv milestone6_env
source milestone6_env/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set HuggingFace token (required for model access)
```bash
export HF_TOKEN="your_huggingface_token"
```

### 5. Start Jupyter and open notebook
```bash
jupyter notebook rag_pipeline.ipynb
```

---

## Part 1: RAG Pipeline Usage

### Running the pipeline
Open `rag_pipeline.ipynb` and run all cells in order:
- Cell 1: Imports and setup
- Cell 2: Load document corpus
- Cell 3: Chunk documents
- Cell 4: Generate embeddings and build FAISS index
- Cell 5: Retrieval function
- Cell 6: Load LLM (takes 2-3 minutes)
- Cell 7: Full RAG pipeline
- Cell 8: Run 10 evaluation queries
- Cell 9: Latency summary
- Cell 10: Save results

### Example usage
```python
result = rag_query("What is RAG and how does it work?")
print(result["answer"])
```

---

## Part 2: Agent Controller Usage

### Running the agent
```bash
python3 agent_controller.py
```

### Example usage
```python
from agent_controller import RAGAgent
agent = RAGAgent()
result = agent.run("What are the key components of a RAG system?")
print(result["answer"])
```

---

## Architecture Overview

### RAG Pipeline
- **Embedding Model:** all-MiniLM-L6-v2 (384 dimensions)
- **Vector Store:** FAISS IndexFlatL2
- **Chunk Size:** 100 words with 20-word overlap
- **Top-k retrieval:** k=3
- **LLM:** Qwen2.5-7B-Instruct via HuggingFace Transformers

### Agent Controller
- **Tools:** RetrieverTool, SummarizerTool
- **Routing:** Keyword-based heuristic + LLM decision
- **Tracing:** Full JSON traces saved per task

---

## Known Limitations
- Generation latency is high (30-130s) due to T4 GPU constraints
- vLLM was attempted but failed due to CUDA 12.2 vs 12.1/12.4 incompatibility
- Corpus is limited to 10 documents for demonstration purposes
- Some model parameters offloaded to CPU due to VRAM constraints
- Llama-3.1-8B-Instruct access was requested but pending Meta approval at submission time

---

## Hardware & Runtime
- **GPU:** Tesla T4 (15GB VRAM)
- **CUDA:** 12.2
- **Python:** 3.11.2
- **PyTorch:** 2.4.0+cu121
- **Transformers:** 5.5.0
- **OS:** Ubuntu (GCP VM)