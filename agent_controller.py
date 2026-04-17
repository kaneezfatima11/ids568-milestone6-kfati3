import os
import json
import time
import torch
import numpy as np
import faiss
from dataclasses import dataclass, field
from typing import Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Configuration ────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
TRACES_DIR = "agent_traces"
os.makedirs(TRACES_DIR, exist_ok=True)

# ── Document Corpus (shared with RAG pipeline) ───────────────────
DOCUMENTS = [
    {"id": "doc1",  "title": "RAG Architecture",     "content": "Retrieval-Augmented Generation (RAG) is an AI architecture that enhances large language model outputs by retrieving relevant information from external knowledge bases. RAG consists of two main phases: indexing and querying. During indexing, documents are chunked, embedded into vectors, and stored in a vector database. During querying, the user query is embedded and used to find similar chunks via similarity search. The retrieved chunks are then passed as context to the LLM for grounded response generation."},
    {"id": "doc2",  "title": "Vector Databases",     "content": "Vector databases store high-dimensional embeddings and enable efficient similarity search. Popular vector databases include FAISS, Chroma, Weaviate, and Pinecone. FAISS (Facebook AI Similarity Search) is an open-source library optimized for fast nearest neighbor search. It supports multiple index types including Flat, IVF, and HNSW. Flat indexes provide exact search while IVF and HNSW provide approximate nearest neighbor search for better scalability."},
    {"id": "doc3",  "title": "Document Chunking",    "content": "Document chunking is the process of splitting large documents into smaller pieces for embedding and retrieval. Common chunking strategies include fixed-size chunking, recursive character splitting, and semantic chunking. Fixed-size chunking splits documents into equal sized chunks. Recursive splitting tries to split on natural boundaries like paragraphs and sentences. Chunk size and overlap are critical parameters - typical chunk sizes range from 256 to 1024 tokens with 10-20% overlap to preserve context across chunk boundaries."},
    {"id": "doc4",  "title": "Embedding Models",     "content": "Embedding models convert text into dense numerical vectors that capture semantic meaning. Popular embedding models include all-MiniLM-L6-v2, all-mpnet-base-v2, and bge-large-en-v1.5. The all-MiniLM-L6-v2 model produces 384-dimensional embeddings and is optimized for speed. Semantic similarity between texts is measured using cosine similarity or dot product between their embedding vectors. Models trained with contrastive learning produce embeddings where semantically similar texts are close together in vector space."},
    {"id": "doc5",  "title": "LLM Hallucination",    "content": "Hallucination in large language models refers to the generation of factually incorrect or fabricated information. RAG systems help reduce hallucinations by grounding responses in retrieved context. Hallucinations occur when models rely on parametric knowledge rather than provided context. Evaluation of hallucinations involves checking if response claims are supported by retrieved documents. Common mitigation strategies include strict prompting to only use provided context, citation requirements, and faithfulness scoring."},
    {"id": "doc6",  "title": "Agentic AI Systems",   "content": "Agentic AI systems are LLM-powered applications that can reason, plan, and execute multi-step tasks autonomously. Key components include planning modules, tool use capabilities, memory management, and self-correction mechanisms. The ReAct pattern combines reasoning and acting by alternating between thought steps and action steps. Agents can use tools like web search, code execution, and retrieval systems. Multi-agent systems coordinate multiple specialized agents to solve complex tasks."},
    {"id": "doc7",  "title": "MLOps Pipeline",       "content": "MLOps refers to the practice of applying DevOps principles to machine learning systems. Key components of MLOps include data versioning, model training pipelines, model registry, serving infrastructure, and monitoring. Continuous integration and deployment for ML models involves automated testing of data quality, model performance, and serving endpoints. Model drift detection monitors for changes in input data distribution or model output quality over time."},
    {"id": "doc8",  "title": "Retrieval Evaluation", "content": "Retrieval evaluation measures the quality of document retrieval in RAG systems. Key metrics include Precision@K which measures the fraction of retrieved documents that are relevant, Recall@K which measures the fraction of relevant documents that are retrieved, and Mean Reciprocal Rank (MRR) which measures the rank of the first relevant document. A good retrieval system balances precision and recall. End-to-end RAG evaluation also includes generation metrics like faithfulness, answer relevance, and context utilization."},
    {"id": "doc9",  "title": "Prompt Engineering",   "content": "Prompt engineering is the practice of designing effective inputs to language models to achieve desired outputs. For RAG systems, prompts typically include a system instruction, retrieved context, and the user question. Key techniques include chain-of-thought prompting which encourages step-by-step reasoning, few-shot prompting which provides examples, and structured output prompting which requests responses in specific formats. Context ordering affects model performance - placing most relevant context first or last tends to work better than middle placement."},
    {"id": "doc10", "title": "Model Serving",        "content": "Model serving refers to deploying trained machine learning models to serve inference requests in production. Common serving frameworks include Hugging Face Transformers, vLLM, Ollama, and TorchServe. Hugging Face Transformers provides a simple interface for loading and running models locally. vLLM is optimized for high throughput serving using PagedAttention memory management. Key serving metrics include latency (time to first token), throughput (tokens per second), and hardware utilization."},
]

# ── Data Classes ─────────────────────────────────────────────────
@dataclass
class TraceStep:
    step_type: str
    content: dict
    timestamp: float = field(default_factory=time.time)

@dataclass
class AgentTrace:
    task_id: int
    query: str
    steps: list = field(default_factory=list)
    final_answer: str = ""
    tool_used: str = ""
    success: bool = False
    total_time_s: float = 0.0

# ── Index Builder ─────────────────────────────────────────────────
def build_index(documents):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    chunks = []
    for doc in documents:
        chunks.append({
            "chunk_id": f"{doc['id']}_chunk0",
            "doc_id": doc["id"],
            "title": doc["title"],
            "text": doc["content"]
        })
    embeddings = embedder.encode(
        [c["text"] for c in chunks],
        convert_to_numpy=True
    )
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype(np.float32))
    return embedder, index, chunks

# ── LLM Loader ────────────────────────────────────────────────────
def load_llm(model_name):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto"
    )
    print("LLM loaded!")
    return tokenizer, model

def generate(tokenizer, model, prompt, max_new_tokens=300):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt")
    # Move inputs to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "assistant" in response.lower():
        response = response.split("assistant")[-1].strip()
    return response

# ── Tools ─────────────────────────────────────────────────────────
def retriever_tool(query, embedder, index, chunks, k=3):
    """Retrieve top-k relevant chunks for a query."""
    query_emb = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb.astype(np.float32), k)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        results.append({
            "title": chunks[idx]["title"],
            "doc_id": chunks[idx]["doc_id"],
            "text": chunks[idx]["text"],
            "distance": float(dist)
        })
    return results

def summarizer_tool(text, tokenizer, model):
    """Summarize a given text."""
    prompt = f"Summarize the following text in 2-3 concise sentences:\n\n{text}"
    return generate(tokenizer, model, prompt, max_new_tokens=150)

# ── Router ────────────────────────────────────────────────────────
def route_query(query):
    """Keyword-based router to select tool."""
    query_lower = query.lower()
    summarize_keywords = ["summarize", "summary", "brief", "tldr", "condense", "overview"]
    retrieve_keywords  = ["what", "how", "explain", "describe", "define", "difference",
                          "compare", "why", "when", "which", "tell me"]
    
    for kw in summarize_keywords:
        if kw in query_lower:
            return "summarizer"
    for kw in retrieve_keywords:
        if kw in query_lower:
            return "retriever"
    return "retriever"  # default

# ── Agent ─────────────────────────────────────────────────────────
class RAGAgent:
    def __init__(self):
        print("Initializing RAG Agent...")
        self.embedder, self.index, self.chunks = build_index(DOCUMENTS)
        self.tokenizer, self.model = load_llm(MODEL_NAME)
        print("Agent ready!")

    def run(self, query, task_id=0):
        start_time = time.time()
        trace = AgentTrace(task_id=task_id, query=query)

        # Step 1: Thought
        thought = f"Analyzing query: '{query}'. Determining best tool to use."
        trace.steps.append(TraceStep("thought", {"content": thought}))
        print(f"\n[THOUGHT] {thought}")

        # Step 2: Route
        tool_name = route_query(query)
        trace.steps.append(TraceStep("routing", {
            "decision": tool_name,
            "reason": f"Query matched {tool_name} keywords"
        }))
        print(f"[ROUTING] Selected tool: {tool_name}")

        # Step 3: Execute tool
        if tool_name == "retriever":
            retrieved = retriever_tool(
                query, self.embedder, self.index, self.chunks, k=3
            )
            context = "\n\n".join([
                f"[{r['title']}]: {r['text']}" for r in retrieved
            ])
            trace.steps.append(TraceStep("action", {
                "tool": "retriever",
                "input": query,
                "retrieved_docs": [r["title"] for r in retrieved]
            }))
            print(f"[ACTION] Retrieved: {[r['title'] for r in retrieved]}")

            # Step 4: Generate grounded response
            prompt = f"""Answer the question using ONLY the provided context.
If the context does not contain enough information, say so.

Context:
{context}

Question: {query}

Answer:"""
            answer = generate(self.tokenizer, self.model, prompt, max_new_tokens=300)
            if "Answer:" in answer:
                answer = answer.split("Answer:")[-1].strip()

        elif tool_name == "summarizer":
            retrieved = retriever_tool(
                query, self.embedder, self.index, self.chunks, k=2
            )
            combined_text = " ".join([r["text"] for r in retrieved])
            trace.steps.append(TraceStep("action", {
                "tool": "summarizer",
                "input": query,
                "retrieved_docs": [r["title"] for r in retrieved]
            }))
            print(f"[ACTION] Summarizing: {[r['title'] for r in retrieved]}")
            answer = summarizer_tool(combined_text, self.tokenizer, self.model)

        # Step 5: Observation + Answer
        trace.steps.append(TraceStep("observation", {"output_preview": answer[:200]}))
        trace.steps.append(TraceStep("answer", {"final_answer": answer}))
        print(f"[ANSWER] {answer[:200]}...")

        trace.final_answer = answer
        trace.tool_used = tool_name
        trace.success = True
        trace.total_time_s = time.time() - start_time

        # Save trace
        self._save_trace(trace)
        return {
            "answer": answer,
            "tool_used": tool_name,
            "trace": trace,
            "total_time_s": trace.total_time_s
        }

    def _save_trace(self, trace):
        trace_data = {
            "task_id": trace.task_id,
            "query": trace.query,
            "tool_used": trace.tool_used,
            "final_answer": trace.final_answer,
            "success": trace.success,
            "total_time_s": trace.total_time_s,
            "steps": [
                {
                    "step_type": s.step_type,
                    "content": s.content,
                    "timestamp": s.timestamp
                }
                for s in trace.steps
            ]
        }
        path = os.path.join(TRACES_DIR, f"task_{trace.task_id:02d}.json")
        with open(path, "w") as f:
            json.dump(trace_data, f, indent=2)
        print(f"[TRACE] Saved to {path}")


# ── Evaluation Tasks ──────────────────────────────────────────────
EVAL_TASKS = [
    "What is RAG and how does it work?",
    "Summarize the key concepts of vector databases",
    "What are the different document chunking strategies?",
    "How do embedding models capture semantic meaning?",
    "What causes hallucination in LLMs and how can it be mitigated?",
    "Explain the ReAct pattern in agentic AI systems",
    "What are the key components of an MLOps pipeline?",
    "How is retrieval quality measured in RAG systems?",
    "What prompt engineering techniques work best for RAG?",
    "What is the difference between vLLM and Hugging Face Transformers?"
]


if __name__ == "__main__":
    agent = RAGAgent()
    
    print("\n" + "="*60)
    print("Running 10 evaluation tasks...")
    print("="*60)
    
    for i, task in enumerate(EVAL_TASKS):
        print(f"\n{'='*60}")
        print(f"Task {i+1}/10: {task}")
        print("="*60)
        result = agent.run(task, task_id=i+1)
        print(f"Tool used: {result['tool_used']}")
        print(f"Time: {result['total_time_s']:.1f}s")
    
    print("\nAll 10 tasks complete!")
    print(f"Traces saved to {TRACES_DIR}/")