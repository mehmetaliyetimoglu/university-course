# Demo Video Guide

**Recommended length:** 2-3 minutes  
**Language:** English  
**Goal:** Show that the system satisfies LangGraph, LangChain, RAG, and Web Search requirements.

---

## Recording Plan

### 1. Opening - 15 seconds

Say:

> Hello, we are Batuhan Akbasak, Mehmet Ali Yetimoglu, and Onur Kaan Sen. Our project is a University Course Selection Assistant. It is an agentic AI system that helps students choose courses, check prerequisites, and combine local course data with live web information.

Show:

- Project folder in VS Code
- `agent.py`, `main.py`, `corpus/`, and `TECHNICAL_REPORT.md`

---

### 2. Architecture - 25 seconds

Open `agent.py` and show:

- `AgentState`
- `build_graph()`
- Nodes:
  - `analyze`
  - `retrieve`
  - `web_search`
  - `respond`
- Conditional edges

Say:

> We use LangGraph to build the agent as a state machine. The first node analyzes the user query and decides whether the agent needs RAG, web search, both, or neither.

---

### 3. RAG Corpus - 20 seconds

Show:

- `corpus/` folder
- The 17 `.txt` documents
- `ingest.py`

Say:

> Our RAG pipeline uses 17 local documents. These documents contain course codes, prerequisites, credits, electives, graduation requirements, and career information. The documents are embedded with HuggingFace embeddings and stored in ChromaDB.

---

### 4. Run RAG Demo - 30 seconds

Run:

```powershell
python main.py --query "What are the prerequisites for the Machine Learning course?"
```

Point out:

- `Needs RAG: True`
- `Needs web: False`
- Retrieved document chunks
- English final answer

Say:

> This query only needs local university data, so the agent uses RAG and does not call web search.

---

### 5. Run Web Search Demo - 30 seconds

Run:

```powershell
python main.py --query "What was the average salary of AI engineers in Turkey in 2024?"
```

Point out:

- `Needs RAG: False`
- `Needs web: True`
- Web search step

Say:

> This question asks for current salary information, so the agent uses live web search instead of the local course database.

---

### 6. Run Hybrid Demo - 45 seconds

Run:

```powershell
python main.py --query "Which math courses should I complete before taking the Deep Learning elective? Also, what is the latest PyTorch version?"
```

Point out:

- `Needs RAG: True`
- `Needs web: True`
- RAG retrieves course prerequisites
- Web search gets current PyTorch information
- Final answer combines both

Say:

> This is the most important demo because it visibly uses both RAG and web search in the same run. Course prerequisites come from the local knowledge base, while the PyTorch version comes from web search.

---

### 7. Closing - 15 seconds

Say:

> In summary, our assistant uses LangGraph for branching agent control, LangChain for LLM and retrieval abstractions, ChromaDB and HuggingFace embeddings for RAG, and DuckDuckGo search for current web information. This satisfies the required agentic AI project components.

Show:

- `TECHNICAL_REPORT.md`
- `AI_USAGE.md`
- `.env.example`

---

## What Must Be Visible in the Video

- The project runs from the terminal.
- The agent prints its decision:
  - `Needs RAG`
  - `Needs web`
- At least one query uses RAG only.
- At least one query uses web search only.
- At least one query uses both RAG and web search.
- The final answer is in English.
- The repository contains:
  - `README.md`
  - `.env.example`
  - `AI_USAGE.md`
  - `TECHNICAL_REPORT.md`
  - `corpus/`

---

## Short Version If You Need a Faster Video

Run only:

```powershell
python main.py --demo
```

Then briefly explain the three demos:

1. Demo 1 uses RAG.
2. Demo 2 uses web search.
3. Demo 3 uses both RAG and web search.

This is enough for a 2-3 minute recording if you speak quickly and keep the focus on the terminal output.

