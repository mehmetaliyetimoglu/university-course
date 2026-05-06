"""
agent.py
--------
Üniversite Ders Seçim Asistanı - LangGraph Ajanı

Mimari:
    START
      │
      ▼
  [analyze]  ←── Sorguyu analiz et, hangi araçlar gerekli?
      │
      ├─(needs_rag=True)──► [retrieve]  Yerel vektör deposundan bilgi al
      │                          │
      │                          └─(needs_web=True)──► [web_search]
      │                          └─(needs_web=False)─► [respond]
      │
      ├─(needs_rag=False, needs_web=True)──► [web_search]
      │                                           │
      │                                           └──► [respond]
      │
      └─(neither)──► [respond]
                         │
                        END

Gereksinimler:
  - LangGraph: StateGraph, 4 node, 2 conditional edge ✓
  - LangChain: ChatOpenAI, PromptTemplate, tool tanımları ✓
  - RAG: ChromaDB + HuggingFace embeddings (Türkçe model) ✓
  - Web Search: DDGS (DuckDuckGo) aracı ✓
"""

import os
import json
import operator
import contextlib
import io
from typing import TypedDict, Annotated

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

from dotenv import load_dotenv

# LangGraph
from langgraph.graph import StateGraph, END

# LangChain
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Web Search
from ddgs import DDGS
from openai import AuthenticationError

load_dotenv()

# ─────────────────────────────────────────────
# Sabitler
# ─────────────────────────────────────────────
CHROMA_DIR = "./chroma_db"

# Türkçe destekli çok dilli embedding modeli
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Groq (varsayılan — ücretsiz, anında kayıt: console.groq.com)
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"

# GitHub Models (alternatif)
GITHUB_BASE_URL = "https://models.github.ai/inference"
GITHUB_DEFAULT_MODEL = "openai/gpt-4.1"

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GITHUB_TOKEN  = os.getenv("GITHUB_TOKEN", "")

TOP_K_DOCS       = 5
MAX_WEB_RESULTS  = 3


# ─────────────────────────────────────────────
# Ajan Durumu (State)
# ─────────────────────────────────────────────
class AgentState(TypedDict):
    query:          str
    messages:       Annotated[list, operator.add]
    retrieved_docs: str
    web_results:    str
    needs_rag:      bool
    needs_web:      bool
    rag_done:       bool
    web_done:       bool
    response:       str


# ─────────────────────────────────────────────
# LLM & Araç Başlatma
# ─────────────────────────────────────────────
def get_llm() -> ChatOpenAI:
    if GROQ_API_KEY:
        model = os.getenv("GROQ_MODEL", GROQ_DEFAULT_MODEL)
        return ChatOpenAI(
            model=model,
            base_url=GROQ_BASE_URL,
            api_key=GROQ_API_KEY,
            temperature=0.3,
        )
    elif GITHUB_TOKEN:
        model = os.getenv("GITHUB_MODEL", GITHUB_DEFAULT_MODEL)
        return ChatOpenAI(
            model=model,
            base_url=GITHUB_BASE_URL,
            api_key=GITHUB_TOKEN,
            temperature=0.3,
        )
    else:
        raise ValueError(
            "\n❌ API anahtarı bulunamadı!\n\n"
            "SEÇENEK 1 — Groq (Önerilen, ücretsiz ve anında):\n"
            "  1. https://console.groq.com adresine gidin ve kayıt olun\n"
            "  2. 'API Keys' → 'Create API Key'\n"
            "  3. .env dosyasına ekleyin: GROQ_API_KEY=gsk_xxxx\n\n"
            "SEÇENEK 2 — GitHub Models (Student Pack gerekli):\n"
            "  1. https://education.github.com/pack başvurun\n"
            "  2. .env dosyasına ekleyin: GITHUB_TOKEN=ghp_xxxx\n"
        )


def explain_auth_error(provider: str) -> RuntimeError:
    if provider == "GitHub Models":
        return RuntimeError(
            "\nGitHub Models authentication failed (Unauthorized).\n\n"
            "Fix:\n"
            "  1. Create a new GitHub Personal Access Token.\n"
            "  2. Give it GitHub Models read permission (models: read / read:models).\n"
            "  3. Replace GITHUB_TOKEN in .env with the new token.\n"
            "  4. Keep GITHUB_MODEL=openai/gpt-4.1 and try again.\n\n"
            "Note: Do not put your token in .env.example; keep it only in .env."
        )
    return RuntimeError(
        "\nLLM provider authentication failed.\n"
        "Check the API key in your .env file."
    )


def invoke_llm(messages):
    try:
        return get_llm().invoke(messages)
    except AuthenticationError:
        provider = "Groq" if GROQ_API_KEY else "GitHub Models"
        raise explain_auth_error(provider) from None


def get_vectorstore() -> Chroma:
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            f"Vektör deposu bulunamadı: '{CHROMA_DIR}'\n"
            "Önce 'python ingest.py' komutunu çalıştırın!"
        )
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu", "local_files_only": True},
            encode_kwargs={"normalize_embeddings": True},
            show_progress=False,
        )
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="university_courses",
    )


def web_search_tool(query: str) -> str:
    """Run a DuckDuckGo web search."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=MAX_WEB_RESULTS, region="wt-wt"))
        if not results:
            return "No web search results were found."
        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"[{i}] {r.get('title', 'Untitled')}\n"
                f"    {r.get('href', '')}\n"
                f"    {r.get('body', '')[:400]}"
            )
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Web search failed: {str(e)}"


# ─────────────────────────────────────────────
# Node 1: Sorgu Analizi
# ─────────────────────────────────────────────
def analyze_query(state: AgentState) -> dict:
    """Decide which tools are needed for the query."""
    print("\n[AGENT] 🔍 Analyzing query...")

    messages = [
        SystemMessage(content="""You are a university course advising assistant.
Analyze the user's question and decide which tools are needed.

Tools:
1. RAG (Local Knowledge Base): Use it for course codes, course content, credits,
   prerequisites, grading, graduation requirements, academic calendar, scholarships,
   clubs, and university-specific advising information.

2. Web Search: Use it for current news, salaries, latest technology versions,
   job postings, and industry trends.

Return ONLY this JSON format and nothing else:
{"use_rag": true, "use_web": false, "reason": "short English explanation"}"""),
        HumanMessage(content=f"User question: {state['query']}")
    ]

    response = invoke_llm(messages)

    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        decision = json.loads(content.strip())
        needs_rag = bool(decision.get("use_rag", True))
        needs_web = bool(decision.get("use_web", False))
        reason    = decision.get("reason", "")
    except Exception:
        needs_rag = True
        needs_web = False
        reason    = "Could not parse analysis; defaulting to RAG."

    print(f"         Needs RAG: {needs_rag} | Needs web: {needs_web}")
    print(f"         Reason: {reason}")

    return {
        "needs_rag":      needs_rag,
        "needs_web":      needs_web,
        "rag_done":       False,
        "web_done":       False,
        "retrieved_docs": "",
        "web_results":    "",
        "response":       "",
        "messages":       [HumanMessage(content=state["query"])],
    }


# ─────────────────────────────────────────────
# Node 2: RAG Erişimi
# ─────────────────────────────────────────────
def retrieve_docs(state: AgentState) -> dict:
    """ChromaDB'den ilgili belgeleri getirir."""
    print("\n[AGENT] 📚 Retrieving from local knowledge base (RAG)...")

    vectorstore = get_vectorstore()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        docs = vectorstore.similarity_search(state["query"], k=TOP_K_DOCS)

    if not docs:
        retrieved_text = "No relevant information was found in the local knowledge base."
    else:
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[Source {i} — {source}]\n{doc.page_content}")
        retrieved_text = "\n\n---\n\n".join(parts)

    print(f"         Found {len(docs)} document chunks.")

    return {
        "retrieved_docs": retrieved_text,
        "rag_done":       True,
    }


# ─────────────────────────────────────────────
# Node 3: Web Araması
# ─────────────────────────────────────────────
def web_search(state: AgentState) -> dict:
    """DDGS ile güncel web araması yapar."""
    print("\n[AGENT] 🌐 Running web search...")

    search_query = f"{state['query']} Turkey university"
    results = web_search_tool(search_query)
    print("         Web search completed.")

    return {
        "web_results": results,
        "web_done":    True,
    }


# ─────────────────────────────────────────────
# Node 4: Yanıt Üretimi
# ─────────────────────────────────────────────
def generate_response(state: AgentState) -> dict:
    """Toplanan bilgilerle kapsamlı bir yanıt üretir."""
    print("\n[AGENT] ✍️  Generating response...")

    context_parts  = []
    sources_used   = []

    if state.get("retrieved_docs"):
        context_parts.append(
            "=== LOCAL KNOWLEDGE BASE (RAG) ===\n" + state["retrieved_docs"]
        )
        sources_used.append("📚 Local knowledge base (RAG)")

    if state.get("web_results"):
        context_parts.append(
            "=== WEB SEARCH RESULTS ===\n" + state["web_results"]
        )
        sources_used.append("🌐 Web search")

    context     = "\n\n".join(context_parts) if context_parts else "No context found."
    sources_str = " + ".join(sources_used)  if sources_used  else "General knowledge"

    messages = [
        SystemMessage(content=f"""You are a helpful university course advising assistant.
Use the CONTEXT below to answer the user's question.

IMPORTANT RULES:
- Answer in the same language as the user's question.
- If the user asks in English, answer fully in English.
- Use the context whenever it contains relevant information, and give concrete details
  such as course code, credits, prerequisites, and recommended order.
- If a Turkish course title appears in the context, you may translate it to English,
  but keep the official course code and optionally include the original title in parentheses.
- If web search results are available, include them in the answer.
- Make the answer clear, student-friendly, and structured with bullet points when useful.

Sources used: {sources_str}

CONTEXT:
{context}"""),
        HumanMessage(content=state["query"])
    ]

    response       = invoke_llm(messages)
    final_response = response.content

    print("         Response ready.")

    return {
        "response": final_response,
        "messages": [AIMessage(content=final_response)],
    }


# ─────────────────────────────────────────────
# Conditional Edge Fonksiyonları
# ─────────────────────────────────────────────
def route_after_analyze(state: AgentState) -> str:
    """Conditional edge #1: analiz → retrieve | web_search | respond"""
    if state.get("needs_rag") and not state.get("rag_done"):
        return "retrieve"
    elif state.get("needs_web") and not state.get("web_done"):
        return "web_search"
    return "respond"


def route_after_retrieve(state: AgentState) -> str:
    """Conditional edge #2: retrieve → web_search | respond"""
    if state.get("needs_web") and not state.get("web_done"):
        return "web_search"
    return "respond"


# ─────────────────────────────────────────────
# LangGraph State Machine
# ─────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("analyze",    analyze_query)
    graph.add_node("retrieve",   retrieve_docs)
    graph.add_node("web_search", web_search)
    graph.add_node("respond",    generate_response)

    graph.set_entry_point("analyze")

    graph.add_conditional_edges(
        "analyze", route_after_analyze,
        {"retrieve": "retrieve", "web_search": "web_search", "respond": "respond"}
    )
    graph.add_conditional_edges(
        "retrieve", route_after_retrieve,
        {"web_search": "web_search", "respond": "respond"}
    )

    graph.add_edge("web_search", "respond")
    graph.add_edge("respond", END)

    return graph.compile()


# ─────────────────────────────────────────────
# Ana Çalıştırma Fonksiyonu
# ─────────────────────────────────────────────
def run_agent(query: str) -> str:
    app = build_graph()

    initial_state: AgentState = {
        "query":          query,
        "messages":       [],
        "retrieved_docs": "",
        "web_results":    "",
        "needs_rag":      False,
        "needs_web":      False,
        "rag_done":       False,
        "web_done":       False,
        "response":       "",
    }

    result = app.invoke(initial_state)
    return result["response"]


if __name__ == "__main__":
    test_query = "What are the prerequisites for the Machine Learning course?"
    print(f"Test: {test_query}")
    print(run_agent(test_query))
