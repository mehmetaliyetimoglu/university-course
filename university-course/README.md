# 🎓 Üniversite Ders Seçim Asistanı

LangGraph + LangChain + RAG + Web Araması kullanılarak geliştirilmiş ajansal yapay zeka uygulaması.

---

## Mimari Diyagram

```
                    ┌─────────────┐
    Kullanıcı ───► │   main.py   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   agent.py  │  ← LangGraph StateGraph
                    └──────┬──────┘
                           │
              ┌────────────▼────────────┐
              │     [analyze] Node      │  Node 1: Sorgu Analizi
              │   LLM ile planlama      │  (LangChain ChatOpenAI)
              └────────────┬────────────┘
                           │
          ┌────────────────┼────────────────┐
          │ (needs_rag)    │ (needs_web)    │ (neither)
          ▼                ▼                ▼
  ┌──────────────┐         │         ┌─────────────┐
  │  [retrieve]  │         │         │  [respond]  │
  │  RAG Node    │         │         │  Node 4     │
  │ ChromaDB +   │         │         │  LangChain  │
  │ HuggingFace  │         │         │  ChatOpenAI │
  └──────┬───────┘         │         └──────┬──────┘
         │                 │                │
         │(needs_web)       │                │
         ▼                 ▼                │
  ┌──────────────────────────┐              │
  │      [web_search] Node   │              │
  │       Node 3             │              │
  │     DuckDuckGo API       │              │
  └──────────────┬───────────┘              │
                 │                          │
                 ▼                          │
          ┌─────────────┐                   │
          │  [respond]  │ ◄─────────────────┘
          │  Node 4     │
          └──────┬──────┘
                 │
                END
```

### Bileşenler

| Bileşen | Teknoloji | Açıklama |
|---------|-----------|----------|
| **LangGraph** | `StateGraph` | 4 node, 2 conditional edge ile durum makinesi |
| **LangChain** | `ChatOpenAI`, `ChatPromptTemplate` | LLM çağrıları ve prompt yönetimi |
| **RAG** | `ChromaDB` + `HuggingFaceEmbeddings` | 17 belge, lokal vektör deposu |
| **Web Araması** | `DuckDuckGo Search` | Ücretsiz, API anahtarsız web araması |
| **LLM** | GitHub Models (`gpt-4o-mini`) | Ücretsiz GitHub Student erişimi |

---

## Kurulum

### 1. Depoyu Klonla

```bash
git clone https://github.com/KULLANICI_ADINIZ/university-course-agent.git
cd university-course-agent
```

### 2. Sanal Ortam Oluştur

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Bağımlılıkları Yükle

```bash
pip install -r requirements.txt
```

> **Not:** `sentence-transformers` ilk yüklemede ~400 MB model indirir.

### 4. Ortam Değişkenlerini Ayarla

```bash
cp .env.example .env
```

`.env` dosyasını açın ve GitHub token'ınızı girin:

```
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
MODEL_NAME=openai/gpt-4o-mini
```

**GitHub Token Alma:**
1. [GitHub Student Developer Pack](https://education.github.com/pack) başvurun
2. Onay sonrası GitHub'da Settings → Developer Settings → Personal Access Tokens
3. `models:read` scope'u seçerek token oluşturun

### 5. RAG Vektör Deposunu Oluştur

```bash
python ingest.py
```

Beklenen çıktı:
```
[1/3] Belgeler yükleniyor...
  ✓ Yüklendi: 01_bilgisayar_muhendisligi_temel.txt
  ...
✅ Veri aktarımı tamamlandı!
```

### 6. Ajanı Çalıştır

```bash
# Etkileşimli mod
python main.py

# Demo sorguları (hem RAG hem web araması kullanır)
python main.py --demo

# Tek sorgu
python main.py --query "Makine öğrenmesi dersinin ön koşulları nelerdir?"
```

---

## Örnek Sorgu ve Beklenen Çıktı

**Sorgu:**
```
Derin öğrenme seçmeli dersini almadan önce hangi dersleri tamamlamalıyım?
Ayrıca PyTorch'un güncel sürümü nedir?
```

**Ajan Akışı:**
```
[AJAN] 🔍 Sorgu analiz ediliyor...
         RAG gerekli: True | Web gerekli: True
         Neden: Ön koşullar için yerel veritabanı, PyTorch sürümü için web araması

[AJAN] 📚 Yerel veritabanından bilgi alınıyor (RAG)...
         4 belge parçası bulundu.

[AJAN] 🌐 Web araması yapılıyor...
         Web araması tamamlandı.

[AJAN] ✍️  Yanıt oluşturuluyor...
```

**Yanıt:**
```
BM451 Derin Öğrenme dersi için aşağıdaki ön koşulları tamamlamanız gerekmektedir:

• BM402 - Makine Öğrenmesi (zorunlu ön koşul)
  - Bu ders için de MAT301 (Olasılık ve İstatistik) ve BM401 (YZ'ye Giriş) gereklidir.

Dolayısıyla tam yol: BM401 → BM402 → BM451

PyTorch hakkında: Web araması sonuçlarına göre PyTorch'un güncel kararlı sürümü
2.x serisidir. Güncel bilgi için pytorch.org adresini kontrol edebilirsiniz.
```

---

## Proje Yapısı

```
university-course-agent/
├── agent.py          # LangGraph state machine (ana ajan kodu)
├── ingest.py         # RAG veri aktarımı scripti
├── main.py           # CLI giriş noktası
├── requirements.txt  # Python bağımlılıkları
├── .env.example      # Ortam değişkenleri şablonu
├── AI_USAGE.md       # Yapay zeka kullanım belgesi
├── README.md         # Bu dosya
└── corpus/           # RAG için 17 metin belgesi
    ├── 01_bilgisayar_muhendisligi_temel.txt
    ├── 02_yapay_zeka_dersleri.txt
    ├── ...
    └── 17_sss_ve_genel_bilgi.txt
```

---

## Teknik Detaylar

### LangGraph State Machine

```python
class AgentState(TypedDict):
    query: str
    messages: Annotated[list, operator.add]
    retrieved_docs: str
    web_results: str
    needs_rag: bool
    needs_web: bool
    rag_done: bool
    web_done: bool
    response: str
```

**Conditional Edge Mantığı:**
- `analyze → retrieve`: `needs_rag=True` ve `rag_done=False`
- `analyze → web_search`: `needs_rag=False` ve `needs_web=True`
- `analyze → respond`: Her ikisi de gerekli değil
- `retrieve → web_search`: `needs_web=True` ve `web_done=False`
- `retrieve → respond`: Web araması gerekmiyorsa

### RAG Pipeline

- **Embedding Modeli:** `sentence-transformers/all-MiniLM-L6-v2` (yerel, ücretsiz)
- **Vektör Deposu:** ChromaDB (yerel disk)
- **Chunk Size:** 500 karakter, 100 örtüşme
- **Top-K:** 4 en yakın belge parçası

### Korpus

17 metin belgesi, toplam ~200 ders/bilgi girişi:
- Temel ve ileri BM dersleri
- Matematik ve temel bilimler
- Seçmeli ders katalogları
- Mezuniyet gereksinimleri
- Kariyer bilgileri

---

## Lisans

MIT License
