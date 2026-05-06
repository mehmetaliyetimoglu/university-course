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
from typing import TypedDict, Annotated

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
GITHUB_DEFAULT_MODEL = "openai/gpt-4o-mini"

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


def get_vectorstore() -> Chroma:
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            f"Vektör deposu bulunamadı: '{CHROMA_DIR}'\n"
            "Önce 'python ingest.py' komutunu çalıştırın!"
        )
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="university_courses",
    )


def web_search_tool(query: str) -> str:
    """DDGS ile web araması yapar."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=MAX_WEB_RESULTS, region="tr-tr"))
        if not results:
            return "Web aramasında sonuç bulunamadı."
        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"[{i}] {r.get('title', 'Başlıksız')}\n"
                f"    {r.get('href', '')}\n"
                f"    {r.get('body', '')[:400]}"
            )
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Web araması sırasında hata oluştu: {str(e)}"


# ─────────────────────────────────────────────
# Node 1: Sorgu Analizi
# ─────────────────────────────────────────────
def analyze_query(state: AgentState) -> dict:
    """Hangi araçların kullanılacağına karar verir."""
    print("\n[AJAN] 🔍 Sorgu analiz ediliyor...")

    llm = get_llm()

    messages = [
        SystemMessage(content="""Sen bir üniversite ders danışmanı asistanısın.
Kullanıcının sorusunu analiz et ve hangi araçları kullanman gerektiğine karar ver.

Araçların:
1. RAG (Yerel Veritabanı): Ders kodu, içerik, kredi, ön koşul, not sistemi,
   mezuniyet gereksinimleri, akademik takvim, burs, kulüp bilgileri için kullan.

2. Web Araması: Güncel haberler, maaş bilgileri, yeni teknoloji sürümleri,
   iş ilanları, sektör trendleri için kullan.

SADECE şu JSON formatında yanıt ver, başka hiçbir şey yazma:
{"use_rag": true, "use_web": false, "reason": "açıklama"}"""),
        HumanMessage(content=f"Kullanıcı sorusu: {state['query']}")
    ]

    response = llm.invoke(messages)

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
        reason    = "Analiz ayrıştırılamadı, RAG varsayılan."

    print(f"         RAG gerekli: {needs_rag} | Web gerekli: {needs_web}")
    print(f"         Neden: {reason}")

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
    print("\n[AJAN] 📚 Yerel veritabanından bilgi alınıyor (RAG)...")

    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(state["query"], k=TOP_K_DOCS)

    if not docs:
        retrieved_text = "Yerel veritabanında ilgili bilgi bulunamadı."
    else:
        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "bilinmeyen")
            parts.append(f"[Kaynak {i} — {source}]\n{doc.page_content}")
        retrieved_text = "\n\n---\n\n".join(parts)

    print(f"         {len(docs)} belge parçası bulundu.")

    return {
        "retrieved_docs": retrieved_text,
        "rag_done":       True,
    }


# ─────────────────────────────────────────────
# Node 3: Web Araması
# ─────────────────────────────────────────────
def web_search(state: AgentState) -> dict:
    """DDGS ile güncel web araması yapar."""
    print("\n[AJAN] 🌐 Web araması yapılıyor...")

    search_query = f"{state['query']} Türkiye üniversite"
    results = web_search_tool(search_query)
    print("         Web araması tamamlandı.")

    return {
        "web_results": results,
        "web_done":    True,
    }


# ─────────────────────────────────────────────
# Node 4: Yanıt Üretimi
# ─────────────────────────────────────────────
def generate_response(state: AgentState) -> dict:
    """Toplanan bilgilerle kapsamlı bir yanıt üretir."""
    print("\n[AJAN] ✍️  Yanıt oluşturuluyor...")

    llm = get_llm()

    context_parts  = []
    sources_used   = []

    if state.get("retrieved_docs"):
        context_parts.append(
            "=== YEREL VERİTABANI (RAG) BİLGİLERİ ===\n" + state["retrieved_docs"]
        )
        sources_used.append("📚 Yerel veritabanı (RAG)")

    if state.get("web_results"):
        context_parts.append(
            "=== WEB ARAMASI SONUÇLARI ===\n" + state["web_results"]
        )
        sources_used.append("🌐 Web araması")

    context     = "\n\n".join(context_parts) if context_parts else "Bağlam bulunamadı."
    sources_str = " + ".join(sources_used)  if sources_used  else "Genel bilgi"

    messages = [
        SystemMessage(content=f"""Sen yardımsever bir üniversite ders danışmanı asistanısın.
Aşağıdaki BAĞLAM bilgilerini kullanarak kullanıcının sorusunu Türkçe olarak yanıtla.

ÖNEMLİ KURALLAR:
- Bağlamdaki bilgileri MUTLAKA kullan ve somut detaylar ver (ders kodu, kredi, ön koşul vb.).
- Bağlamda bilgi varsa "bilgi yok" deme — varsa kullan.
- Web araması sonucu varsa onu da yanıta dahil et.
- Yanıt açık, öğrenciye yardımcı, madde madde olsun.

Kullanılan Kaynaklar: {sources_str}

BAĞLAM:
{context}"""),
        HumanMessage(content=state["query"])
    ]

    response       = llm.invoke(messages)
    final_response = response.content

    print("         Yanıt hazır.")

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
    test_query = "Makine öğrenmesi dersinin ön koşulları nelerdir?"
    print(f"Test: {test_query}")
    print(run_agent(test_query))
