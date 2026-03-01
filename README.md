# 🤖 Practical LLM Engineering

> A hands-on course for building **production-grade LLM systems** — from API basics to multi-agent orchestration.

**7 modules · 39 notebooks · 2 mini-capstones · 1 capstone project**

Every lesson includes working code, `[EXERCISE]` cells with auto-check assertions, and a readiness checklist before moving on.

---

## 🏗️ What You Build

By the end of the course, you'll have a fully functioning **Banking CRM Intelligence System** that:

- Ingests and classifies customer complaints
- Predicts churn using an ML + LLM hybrid pipeline
- Retrieves policy context via RAG
- Generates personalized retention offers
- Enforces guardrails and logs everything
- Runs **partly local at $0** and partly cloud

**Intermediate milestones:**

| Milestone | After | Description |
|-----------|-------|-------------|
| 🔬 Mini-Capstone 1 | Module 3 | Scientific Paper Assistant — multi-query RAG, metadata extraction, cost tracking |
| 🏥 Mini-Capstone 2 | Module 5 | Clinical Triage Pipeline — XGBoost + LLM hybrid, SHAP explanations, LLM-as-judge eval |
| 🏦 Final Capstone | Module 7 | Banking CRM Intelligence System — full production pipeline |

---

## 📁 Repository Structure

```
practical-llm-engineering/
│
├── modules/
│   ├── module_01_fundamentals/          # LLM Fundamentals & API Mastery
│   ├── module_02_structured_output/     # Structured Output & Data Extraction
│   ├── module_03_rag/                   # Retrieval Augmented Generation
│   ├── module_04_agents/                # AI Agents & Tool Use
│   ├── module_05_hybrid_ml/             # Classic ML + LLM Hybrid Systems
│   ├── module_06_evaluation/            # Evaluation, Evals & LLMOps
│   └── module_07_advanced/              # Advanced 2026: Reasoning & Multi-Agents
│
├── capstone/
│   ├── mini_capstone_1/                 # Scientific Paper Assistant
│   ├── mini_capstone_2/                 # Clinical Triage Pipeline
│   └── final_capstone/                  # Banking CRM Intelligence System
│
├── utils/
│   ├── llm_checker.py                   # Auto-check assertion helpers
│   └── setup_datasets.py                # Dataset download & preparation
│
├── data/                                # Local data cache (gitignored)
├── assets/                              # Diagrams and images
├── requirements.txt
└── README.md
```

---

## 📚 Modules

| # | Title | Domain | Key Skills |
|---|-------|--------|------------|
| 1 | LLM Fundamentals & API Mastery | E-commerce / Reviews | SDK, streaming, async, prompt engineering, context window |
| 2 | Structured Output & Data Extraction | Legal / Contracts | Pydantic, Instructor, NER pipelines, Text-to-SQL |
| 3 | Retrieval Augmented Generation | Scientific Literature | ChromaDB, LangChain, HyDE, Multi-Query, contextual retrieval |
| 4 | AI Agents & Tool Use | IT / DevOps | ReAct, LangGraph, MCP, parallel subgraphs, HITL |
| 5 | Classic ML + LLM Hybrid Systems | Healthcare / Clinical | XGBoost, SHAP, embedding features, confidence-based routing |
| 6 | Evaluation, Evals & LLMOps | Multi-domain | RAGAS, LLM-as-judge, Langfuse tracing, prompt A/B testing |
| 7 | Advanced 2026: Reasoning & Multi-Agents | Banking CRM | Extended thinking, vision extraction, production orchestration |

---

## 📓 Notebooks

### Module 1 — LLM Fundamentals & API Mastery
| Notebook | Title |
|----------|-------|
| `lesson_01a_how_llms_work.ipynb` | How LLMs Work — Transformer architecture, tokens, temperature |
| `lesson_01b_api_streaming.ipynb` | API Integration & Streaming — SDK, async, retry patterns |
| `lesson_01c_prompt_engineering.ipynb` | Prompt Engineering — Few-shot, CoT, XML tags, refinement loops |
| `lesson_01d_context_window.ipynb` | Context Window Management — tiktoken, trimming, summarization |

### Module 2 — Structured Output & Data Extraction
| Notebook | Title |
|----------|-------|
| `lesson_02a_pydantic_instructor.ipynb` | Pydantic & Instructor — Validated structured outputs |
| `lesson_02b_ner_pipelines.ipynb` | NER Pipelines — Named entity extraction from legal documents |
| `lesson_02c_text_to_sql.ipynb` | Text-to-SQL — Natural language to SQL with validation |

### Module 3 — Retrieval Augmented Generation
| Notebook | Title |
|----------|-------|
| `lesson_03a_embeddings_chunking.ipynb` | Embeddings & Chunking — Strategies and tradeoffs |
| `lesson_03b_chromadb_rag.ipynb` | ChromaDB RAG — Basic pipeline with similarity search |
| `lesson_03c_advanced_rag.ipynb` | Advanced RAG — HyDE, Multi-Query, contextual retrieval |
| `lesson_03d_local_rag.ipynb` | Local RAG — $0 local inference with LM Studio |

### Module 4 — AI Agents & Tool Use
| Notebook | Title |
|----------|-------|
| `lesson_04a_react_agent.ipynb` | ReAct Agent — Reasoning + acting with tool calls |
| `lesson_04b_langgraph.ipynb` | LangGraph — Stateful agent graphs |
| `lesson_04c_mcp.ipynb` | MCP (Model Context Protocol) — FastMCP server integration |
| `lesson_04d_langgraph_advanced.ipynb` | LangGraph Advanced — Parallel subgraphs, HITL |

### Module 5 — Classic ML + LLM Hybrid Systems
| Notebook | Title |
|----------|-------|
| `lesson_05a_feature_extractor.ipynb` | Feature Extractor — LLM-powered ML feature engineering |
| `lesson_05b_shap_explainer.ipynb` | SHAP Explainer — Interpretable ML with LLM summaries |
| `lesson_05c_hybrid_routing.ipynb` | Hybrid Routing — Confidence-based ML/LLM routing |

### Module 6 — Evaluation, Evals & LLMOps
| Notebook | Title |
|----------|-------|
| `lesson_06a_ragas_eval.ipynb` | RAGAS Evaluation — RAG pipeline quality metrics |
| `lesson_06b_llm_as_judge.ipynb` | LLM-as-Judge — Automated evaluation with structured rubrics |
| `lesson_06c_langfuse.ipynb` | Langfuse Tracing — LLMOps observability and A/B testing |

### Module 7 — Advanced 2026: Reasoning & Multi-Agents
| Notebook | Title |
|----------|-------|
| `lesson_07_reasoning_vision_multiagent.ipynb` | Overview & Integration |
| `lesson_07a_reasoning_extended_thinking.ipynb` | Extended Thinking — Deep reasoning with Claude |
| `lesson_07b_multimodal_vision.ipynb` | Multimodal Vision — GPT-4o vision API, document extraction |
| `lesson_07c_production_multiagent.ipynb` | Production Multi-Agent — Orchestration patterns |

### Capstone: Mini-Capstone 1 (after Module 3)
| Notebook | Title |
|----------|-------|
| `lesson_03h_mini_capstone_1.ipynb` | Scientific Paper Assistant — End-to-end RAG system |

### Capstone: Mini-Capstone 2 (after Module 5)
| Notebook | Title |
|----------|-------|
| `lesson_06_d_minicapstone2.ipynb` | Clinical Triage Pipeline — ML + LLM hybrid system |

### Capstone: Final (after Module 7)
| Notebook | Title |
|----------|-------|
| `lesson_08_capstone_preparation.ipynb` | Capstone Prep — Architecture and planning |
| `lesson_08a_ingestion_context_window.ipynb` | Ingestion Layer |
| `lesson_08b_intelligence_core_rag_ml.ipynb` | Intelligence Core — RAG + ML |
| `lesson_08c_evaluation_harness_cost.ipynb` | Evaluation Harness & Cost Analysis |
| `lesson_09_final_capstone.ipynb` | Final Capstone — Full system integration |
| `lesson_09a_ingestion_layer.ipynb` | Ingestion Layer — Complaint processing |
| `lesson_09b_intelligence_core.ipynb` | Intelligence Core — Churn prediction + RAG |
| `lesson_09c_agent_orchestrator.ipynb` | Agent Orchestrator — Retention offer generation |
| `lesson_09d_structured_output_guardrails.ipynb` | Guardrails — Output validation and safety |
| `lesson_09e_eval_monitoring.ipynb` | Eval & Monitoring — RAGAS + Langfuse |
| `lesson_09f_architecture_cost_summary.ipynb` | Architecture Diagram & Cost Summary |

---

## ⚙️ Setup

### Prerequisites

- Python 3.10+
- Basic Python fluency
- Familiarity with REST APIs
- **No prior LLM experience required**

### Install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/practical-llm-engineering.git
cd practical-llm-engineering
pip install -r requirements.txt
```

### Set API keys

```bash
export OPENAI_API_KEY=your_key_here
export ANTHROPIC_API_KEY=your_key_here
```

### Local inference (optional, $0)

Install [LM Studio](https://lmstudio.ai) and start the server at `http://localhost:1234/v1`.

Recommended models:
- **General tasks:** Mistral-7B-Instruct or Phi-3-mini
- **Polish language:** Bielik-11B

### Langfuse (Module 6)

```bash
git clone https://github.com/langfuse/langfuse && cd langfuse
docker compose up
```

UI at `http://localhost:3000`.

---

## 📦 Stack

| Category | Libraries |
|----------|-----------|
| **LLMs** | OpenAI API, Anthropic API, LM Studio (local, OpenAI-compatible) |
| **Structured Output** | Pydantic v2, Instructor |
| **RAG** | ChromaDB, LangChain, sentence-transformers (`all-MiniLM-L6-v2`) |
| **Agents** | LangGraph, MCP (FastMCP) |
| **ML** | XGBoost, scikit-learn, SHAP |
| **Eval** | RAGAS, LLM-as-judge, Langfuse (self-hosted Docker) |
| **Vision** | GPT-4o vision API, ReportLab |

---

## 🗃️ Datasets

All datasets are public or synthetically generated. No real patient records or financial documents.

| Module | Dataset |
|--------|---------|
| 1 | McAuley-Lab/Amazon-Reviews-2023, Yelp Open Dataset |
| 2 | SEC EDGAR 10-K filings, EUR-Lex (rcadas/EUR-Lex) |
| 3 | gfissore/arxiv-abstracts-2021, pubmed_qa, Wikipedia |
| 4 | giganticode/github-issues-small, synthetic runbooks |
| 5 | Synthetic clinical notes (LLM-generated), Heart Disease UCI |
| 6 | Golden sets built from M1–M5 work |
| 7 | cfpb/us-consumer-finance-complaints, Telco Customer Churn, synthetic bank statements |

Run `python utils/setup_datasets.py` to download and cache all datasets locally.

---

## 🎓 Notebook Structure

Each notebook follows the naming convention `lesson_NNx_title.ipynb` where `NN` is the module number and `x` is the sub-lesson letter.

**Exercise types:**

| Tag | Description |
|-----|-------------|
| `[EXAMPLE]` | Follow-along guided code |
| `[EXERCISE]` | Implement yourself |
| `[CHALLENGE]` | Design and extend |

Every notebook ends with a **readiness checklist** before the next lesson.

---

## 📊 Capstone Output

Running the full capstone on 100 customers produces:

- `complaints.jsonl` — classified and structured complaint records
- `hybrid_routing_log.jsonl` — ML vs LLM routing decisions with cost per record
- `triage_cards.jsonl` — structured risk cards with SHAP-grounded explanations
- `capstone_eval_report.md` — RAGAS scores, LLM-as-judge results, cost breakdown
- Architecture diagram (Mermaid) and executive summary

Includes cloud API vs local LM Studio cost comparison.

---

## 📄 License

MIT
