# Design Document

## System Architecture

The Aptino AI Health Insurance Claims Decision Engine is built on a modern Multi-Agent Retrieval-Augmented Generation (RAG) architecture using LangGraph, FastAPI, and Streamlit.

### 1. LangGraph Workflow
The core logic relies on a state machine with strict type enforcement via Pydantic models. We implemented a 5-node graph:
1. **Case Analysis Agent**: Extracts claim facts, flags missing required fields (like documents or hospital name), explicitly ignores irrelevant attributes, and creates a tailored investigation plan based on the claim type (e.g. domiciliary vs day care).
2. **Policy Evidence Agent**: Takes the investigation plan and queries the vector store using Hybrid Search.
3. **Coverage Agent**: Reads the evidence and produces structured outputs detailing coverage findings and applicable sub-limits.
4. **Decision Agent**: Synthesizes the coverage findings into a final `decision` status, calculates a `confidence` score (0.0-1.0), and formats citations mapping each key finding back to a specific `chunk_id`.
5. **Validation Agent**: Acts as an LLM-as-a-judge critic. It strictly verifies that the Decision Agent's claims are grounded in the cited text. If it detects hallucinated or unsupported claims, it fails the validation, triggering a loop back to the Decision Agent to revise its output. 

### 2. Hybrid Retrieval Pipeline
Instead of relying solely on dense embeddings, we implemented a robust Hybrid Search combining:
- **Dense Retrieval**: `sentence-transformers/all-MiniLM-L6-v2` captures semantic meaning.
- **Sparse Retrieval**: `BM25Okapi` handles exact keyword matching.
Results are fused using Reciprocal Rank Fusion (RRF), ensuring high recall across both dimensions. 

## Reliability Scenarios Handled
- **Missing Information**: `Case Analysis Agent` checks for required fields. If evidence is missing, the `Decision Agent` safely defaults to `NEEDS_REVIEW`.
- **Irrelevant Attributes**: `Case Analysis Agent` explicitly logs ignored attributes, preventing the LLM from hallucinating policy rules about hair color or cafeteria ratings.
- **Hallucination Prevention**: The `Validation Agent` creates a self-correcting retry loop. To prevent infinite loops, a max of 3 iterations is enforced before failing gracefully to `NEEDS_REVIEW`.
- **Buried Clauses**: The Hybrid Search + RRF combination ensures that small, buried clauses (like specific disease waiting periods) are surfaced accurately.

## Implementation Trade-Offs

- **Same-Model Grading Limitation**: Both the generator (Decision Agent) and the critic (Validation Agent) utilize Groq's LLaMA-3 models. While this ensures high speed, using the same model to grade its own output risks confirmation bias. In a production setting, a diverse model architecture (e.g., using GPT-4o for validation) is recommended.
- **Offline Indexing on Render**: Render's free tier provides an ephemeral filesystem, meaning data written to disk is lost on restarts. To mitigate this without relying on a paid external database (like Pinecone), we pre-build the ChromaDB index and commit the binary files to the repository. This guarantees extremely fast cold starts and no indexing delay, but it means policy updates require a new deployment build.
- **Simple Chunking vs Vision Extraction**: The current `document_parser.py` uses heuristic regex and character splitting. For highly unstructured or table-heavy documents, a Vision-based LLM extractor or tools like Unstructured.io would provide better results, though at a higher latency and cost.
