# AI Kullanım Belgesi (AI_USAGE.md)

Bu dosya, proje geliştirme sürecinde yapay zeka araçlarının nasıl ve nerede kullanıldığını belgeler.

## Kullanılan Yapay Zeka Araçları

| Araç | Kullanım Amacı |
|------|----------------|
| Claude (Anthropic) | Kod iskeleti oluşturma, hata ayıklama önerileri |
| GitHub Copilot | Otomatik tamamlama ve küçük kod parçaları |

## Detaylı Kullanım

### 1. Başlangıç İskeleti
- **Dosyalar:** `agent.py`, `ingest.py`
- **Kullanım:** LangGraph StateGraph yapısının genel iskeletini oluşturmak için Claude'dan yardım alındı.
- **İnsan katkısı:** State tanımları, conditional edge mantığı ve node içerikleri ekip tarafından yazıldı ve test edildi.

### 2. Hata Ayıklama
- **Konu:** ChromaDB ve HuggingFace Embeddings entegrasyonu sırasında oluşan `ImportError`ların çözümü.
- **Kullanım:** Hata mesajları Claude'a gösterildi, önerilen düzeltmeler değerlendirilerek uygulandı.

### 3. Dokümantasyon
- **Dosyalar:** `README.md`, docstring'ler
- **Kullanım:** README taslağı için Claude kullanıldı; içerik ekip tarafından gözden geçirildi ve güncellendi.

### 4. Corpus İçeriği
- **Dosyalar:** `corpus/*.txt`
- **Kullanım:** Ders açıklamalarının yapılandırılması için Claude'dan öneri alındı.
- **İnsan katkısı:** Tüm ders bilgileri gerçek üniversite kataloglarından kontrol edilerek düzenlendi.

## Önemli Not

Proje ekibinin tüm üyeleri, bu depodaki her kod satırını anlayabilmekte ve sunum sırasında savunabilmektedir. Yapay zeka yalnızca geliştirme sürecini hızlandırmak için kullanılmış; üretilen kod körü körüne kopyalanmamıştır.

## Kullanılmayan Alanlar

- LangGraph mimari kararları (tamamen ekip tarafından tasarlandı)
- RAG pipeline tasarımı (bölüm dersleri temel alınarak tasarlandı)
- Demo sorguları ve test senaryoları (ekip tarafından belirlendi)
