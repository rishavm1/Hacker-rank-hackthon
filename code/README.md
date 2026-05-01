# HackerRank Orchestrate Support Triage Agent

## 🏗️ Architecture & Approach

This project implements a multi-domain support triage agent using a lightweight, deterministic **Retrieve-and-Route** architecture. The core philosophy of this implementation is safety, determinism, and speed, avoiding common LLM pitfalls like hallucination, prompt injection, and excessive latency.

### 1. Classification & Domain Inference (Rule-Based + BM25)
When a ticket is received, the system first normalizes the input. 
- If the company is missing, the system uses strict keyword matching combined with **BM25 semantic search** to infer the domain (HackerRank, Claude, or Visa). 
- Request Types (`bug`, `feature_request`, `product_issue`, `invalid`) are assigned deterministically via Regex patterns, eliminating LLM hallucinations in classification.

### 2. Retrieval System (BM25Okapi)
**Decision: Why BM25 over Dense Embeddings (Vector DBs)?**
We chose **BM25** via `rank_bm25` instead of dense embeddings (e.g., FAISS + OpenAI text-embedding) for several critical reasons:
1. **Zero Latency / Offline**: BM25 runs entirely locally with zero API calls.
2. **Exact Term Matching**: Support domains rely heavily on exact nomenclature (e.g., "3D Secure", "LTI Connector", "GCAS"). Dense embeddings often retrieve conceptually similar but factually irrelevant documents. BM25 perfectly matches exact support terminology.
3. **Reproducibility**: BM25 requires no stochastic vector generation, guaranteeing deterministic results across environments.

### 3. Safety & Escalation Gate
Before generating a response, the ticket passes through a rigid safety filter (`safety.py`).
- **High-Risk Triggers**: Any queries involving fraud, account hacking, system outages, billing disputes requiring manual action, or legal matters are flagged for mandatory human escalation.
- **Confidence Gate**: If the top retrieved BM25 score is below `5.0` (indicating the corpus lacks relevant documentation), the pipeline **hard-skips the LLM entirely**, escalating the ticket to prevent hallucinations.
- **Prompt Injection**: Known adversarial injection patterns automatically short-circuit the pipeline, resulting in an immediate `invalid` categorization.

### 4. Output Generation (Strict Prompting & Seeded LLM)
If the ticket passes the safety gate, the relevant corpus chunks are passed to the `tencent/hy3-preview:free` model (via OpenRouter API).
- **Seeded Sampling**: The LLM is called with `temperature=0.0` and `seed=42` to maximize determinism.
- **Format Constraints**: The system prompt strictly prohibits bullet points, numbering, and newlines, forcing a concise 2-4 sentence JSON response.
- **Sanitization**: Before writing to `output.csv`, Python performs a final sanitization pass, aggressively stripping any rogue `\n` or `\r` characters to guarantee zero CSV parsing errors.

## 🚀 Execution

To run the pipeline and generate the output:

```bash
python code/main.py
```

### Dependencies
Dependencies are strictly pinned in `requirements.txt` for total reproducibility:
- `openai==2.33.0`
- `scikit-learn==1.8.0`
- `rank_bm25==0.2.2`
- `rich==14.3.3`
- `python-dotenv==1.2.1`
