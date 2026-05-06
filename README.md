# University Course Selection Assistant

This repository contains an agentic AI course advising assistant built with LangGraph, LangChain, RAG, and live web search.

The application code and project documents are in:

```text
university-course/
```

Start here:

- [Technical Report](university-course/TECHNICAL_REPORT.md)
- [Demo Video Guide](university-course/VIDEO_GUIDE.md)
- [AI Usage Disclosure](university-course/AI_USAGE.md)
- [Application README](university-course/README.md)

## Quick Run

```powershell
cd university-course
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python ingest.py
python main.py --demo
```

Before running the demo, add your GitHub Models token to `.env`. Do not commit `.env`.
