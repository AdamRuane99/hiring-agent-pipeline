---
title: AI Hiring Agent Pipeline
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# AI Hiring Agent Pipeline

A production-ready multi-agent pipeline that screens, scores, and ranks job candidates from their CVs — fully LLM-agnostic. Swap between Ollama, Google Gemini, OpenAI, or local HuggingFace models with a single environment variable.

Built with **CrewAI** agents, **FAISS** vector search, and **Streamlit**.

---

## What It Does

Upload a job description and a set of CVs. The pipeline runs three specialised agents in sequence:

| Agent | Role |
|---|---|
| **Screener** | Hard PASS / FAIL against required skills in the JD |
| **Scorer** | Rates each passing candidate 1–10 on Technical Skills, Experience, and Role Alignment — with cited evidence from the CV |
| **Reporter** | Produces a ranked markdown shortlist, ready to send to the hiring manager |

Candidate CVs are embedded into a FAISS vector index using `all-MiniLM-L6-v2`. Agents query this index with natural-language tool calls — answers are always grounded in the actual CV text.

---

## Architecture

```
[Job Description + CVs]
        │
        ▼
[Ingestion]  ─── cv_parser.py
  └── Embed chunks → FAISS index (all-MiniLM-L6-v2)
        │
        ▼
[Screener Agent]  ─── CrewAI + search_candidates tool
  └── PASS / FAIL per candidate
        │
        ▼
[Scorer Agent]  ─── CrewAI + search_candidates tool
  └── 1-10 scores with cited evidence
        │
        ▼
[Reporter Agent]  ─── CrewAI
  └── Ranked markdown shortlist
        │
        ▼
[Streamlit UI]  ─── app.py
```

---

## LLM Providers

Set `LLM_PROVIDER` in `.env` (or the Streamlit sidebar) — no other code changes needed.

| Provider | Env var value | Cost | Notes |
|---|---|---|---|
| **HuggingFace API** (default) | `hf_api` | Free tier | Free token at huggingface.co/settings/tokens |
| Ollama | `ollama` | Free | Local inference, no API key. Install Ollama + pull a model first |
| Google Gemini | `gemini` | Free tier | Free API key at aistudio.google.com. Not available in EU |
| OpenAI | `openai` | Pay-per-use | `gpt-4o-mini` default. Best quality |
| Local HuggingFace | `local_hf` | Free | Downloads model once. Requires ~2GB disk + CPU inference is slow |

Embeddings always run locally via `all-MiniLM-L6-v2` — no API key ever needed for retrieval.

---

## Quick Start (Ollama — recommended)

### 1. Install Ollama
Download from [ollama.com](https://ollama.com) and install. Ollama runs a local server automatically.

### 2. Pull a model
```bash
ollama pull llama3.2:3b
```

### 3. Clone and install
```bash
git clone https://github.com/AdamRuane99/hiring-agent-pipeline
cd hiring-agent-pipeline

python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 4. Configure
```bash
cp .env.example .env
```
The default `.env` is already configured for Ollama + `llama3.2:3b` — no edits needed.

### 5. Run
```bash
streamlit run app.py
```

Open `http://localhost:8501`, select **Ollama** in the sidebar, and click **Run Hiring Pipeline**.

---

## Quick Start (Other Providers)

**Google Gemini (free, non-EU only)**
```bash
# In .env:
LLM_PROVIDER=gemini
GEMINI_API_KEY=AIza_xxx
GEMINI_MODEL=gemini/gemini-2.0-flash-lite
```
Get a free key at [aistudio.google.com](https://aistudio.google.com).

**OpenAI**
```bash
# In .env:
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxx
OPENAI_MODEL=gpt-4o-mini
```

---

## Docker

```bash
# Ollama (default)
docker build -t hiring-agent .
docker run -p 8501:8501 \
  -e LLM_PROVIDER=ollama \
  -e OLLAMA_MODEL=llama3.2:3b \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  hiring-agent

# OpenAI
docker run -p 8501:8501 \
  -e LLM_PROVIDER=openai \
  -e OPENAI_API_KEY=sk-xxx \
  hiring-agent
```

> **Note:** Docker requires Ollama to be running on the host. `host.docker.internal` routes to the host machine on Windows/Mac.

---

## Project Structure

```
├── app.py                  Streamlit UI
├── core/
│   ├── llm_factory.py      Provider abstraction (ollama | gemini | openai | hf_api | local_hf)
│   ├── rag.py              FAISS embedding + retrieval
│   └── cv_parser.py        PDF / TXT / CSV parsing
├── pipeline/
│   ├── tools.py            CrewAI tool: search_candidates (FAISS-backed)
│   ├── agents.py           Screener, Scorer, Reporter agent definitions
│   ├── tasks.py            Task definitions with context chaining
│   └── crew.py             Crew orchestration entry point
├── config/
│   └── job_template.yaml   Config-driven JD template
├── sample_data/            Four sample candidate CVs for demo
├── .env.example
├── requirements.txt
└── Dockerfile
```

---

## Sample Data

Four fictional candidates covering a range of suitability for a Senior Marketing Manager role:

- **cv_aoife_murphy** — Strong match, senior B2C marketing experience
- **cv_ciaran_osullivan** — Good match, digital marketing background
- **cv_niamh_gallagher** — Partial match, relevant experience but gaps
- **cv_sean_byrne** — Weaker match, junior-level experience

---

## Extending

**Add a new LLM provider:** Add a branch in `core/llm_factory.py` returning a `crewai.LLM` instance or a `BaseLLM` subclass.

**Change the job role:** Edit `config/job_template.yaml` and load it in `app.py`.

**Add an agent:** Define it in `pipeline/agents.py`, add a task in `pipeline/tasks.py`, add it to the crew in `pipeline/crew.py`.

---

## Tech Stack

| Component | Library |
|---|---|
| Agent orchestration | CrewAI 1.x |
| LLM abstraction | crewai.LLM (LiteLLM) + custom BaseLLM subclasses |
| Local inference | Ollama |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | FAISS |
| UI | Streamlit |
| Deployment | Docker |
