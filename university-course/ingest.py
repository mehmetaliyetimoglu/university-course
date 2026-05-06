"""
ingest.py
---------
RAG korpus belgelerini okur, parçalara böler ve ChromaDB vektör deposuna aktarır.
Kullanım: python ingest.py
"""

import os
import glob
import shutil
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CORPUS_DIR      = "./corpus"
CHROMA_DIR      = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 100


def load_documents(corpus_dir: str) -> list:
    txt_files = glob.glob(os.path.join(corpus_dir, "*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"'{corpus_dir}' klasöründe .txt dosyası bulunamadı.")

    documents = []
    for file_path in sorted(txt_files):
        loader = TextLoader(file_path, encoding="utf-8")
        docs   = loader.load()
        for doc in docs:
            doc.metadata["source"] = Path(file_path).name
        documents.extend(docs)
        print(f"  ✓ Yüklendi: {Path(file_path).name}")

    print(f"\nToplam {len(documents)} belge yüklendi.")
    return documents


def split_documents(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n---\n", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(documents)
    print(f"Toplam {len(chunks)} parçaya bölündü.")
    return chunks


def build_vectorstore(chunks: list) -> Chroma:
    print(f"\nEmbedding modeli yükleniyor: {EMBEDDING_MODEL}")
    print("(İlk çalıştırmada model indiriliyor, birkaç dakika sürebilir...)")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"ChromaDB deposu oluşturuluyor: {CHROMA_DIR}")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="university_courses",
    )

    print(f"✓ {len(chunks)} chunk başarıyla vektör deposuna eklendi.")
    return vectorstore


def main():
    print("=" * 60)
    print("  ÜNİVERSİTE DERS ASISTANI - RAG Veri Aktarımı")
    print("=" * 60)

    # Eski veritabanını temizle (model değiştiğinde zorunlu)
    if os.path.exists(CHROMA_DIR):
        print(f"\n⚠️  Eski vektör deposu siliniyor: {CHROMA_DIR}")
        shutil.rmtree(CHROMA_DIR)

    print(f"\n[1/3] Belgeler yükleniyor...")
    documents = load_documents(CORPUS_DIR)

    print("\n[2/3] Belgeler parçalara bölünüyor...")
    chunks = split_documents(documents)

    print("\n[3/3] Vektör deposu oluşturuluyor...")
    build_vectorstore(chunks)

    print("\n" + "=" * 60)
    print("✅ Veri aktarımı tamamlandı!")
    print("   Artık 'python main.py' komutuyla ajanı başlatabilirsiniz.")
    print("=" * 60)


if __name__ == "__main__":
    main()
