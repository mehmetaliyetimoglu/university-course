"""
main.py
-------
Üniversite Ders Seçim Asistanı - Komut Satırı Arayüzü

Kullanım:
    python main.py                    # Etkileşimli mod
    python main.py --demo             # Demo sorguları çalıştır
    python main.py --query "soru"     # Tek sorgu modu
"""

import sys
import os
import argparse
import time

from agent import run_agent

# ─────────────────────────────────────────────
# Demo sorguları (hem RAG hem web aramasını tetikler)
# ─────────────────────────────────────────────
DEMO_QUERIES = [
    {
        "query": "Makine öğrenmesi dersinin ön koşulları ve içeriği nedir?",
        "desc": "RAG sorgusu — yerel veritabanından ders bilgisi",
        "expected_tools": "RAG",
    },
    {
        "query": "2024 yılında Türkiye'de yapay zeka mühendislerinin ortalama maaşı ne kadar?",
        "desc": "Web araması sorgusu — güncel maaş bilgisi",
        "expected_tools": "Web Araması",
    },
    {
        "query": "Derin öğrenme seçmeli dersini almadan önce hangi matematik derslerini tamamlamalıyım? Ayrıca PyTorch'un son sürümü nedir?",
        "desc": "Karma sorgu — hem RAG (ön koşullar) hem web (PyTorch güncel sürüm)",
        "expected_tools": "RAG + Web Araması",
    },
]

BANNER = """
╔════════════════════════════════════════════════════════╗
║     ÜNİVERSİTE DERS SEÇİM ASİSTANI 🎓                ║
║     LangGraph + LangChain + RAG + Web Araması          ║
╚════════════════════════════════════════════════════════╝
Çıkmak için: 'quit', 'exit' veya Ctrl+C
"""


def print_separator(char="─", width=60):
    print(char * width)


def check_setup():
    """Kurulum kontrolü yapar."""
    issues = []

    # .env kontrolü
    if not os.path.exists(".env"):
        issues.append("❌ .env dosyası bulunamadı! '.env.example'ı kopyalayın.")
    else:
        from dotenv import load_dotenv
        load_dotenv()
        if not os.getenv("GITHUB_TOKEN"):
            issues.append("❌ GITHUB_TOKEN .env dosyasında boş!")

    # Vektör deposu kontrolü
    if not os.path.exists("./chroma_db"):
        issues.append("❌ Vektör deposu bulunamadı! 'python ingest.py' çalıştırın.")

    if issues:
        print("\n⚠️  Kurulum sorunları tespit edildi:")
        for issue in issues:
            print(f"   {issue}")
        print()

        if any("GITHUB_TOKEN" in i for i in issues) or any(".env" in i for i in issues):
            print("📋 Hızlı kurulum:")
            print("   1. cp .env.example .env")
            print("   2. .env dosyasını açıp GITHUB_TOKEN değerinizi girin")
            print("   3. python ingest.py")
            print("   4. python main.py")
            return False

    return True


def run_single_query(query: str):
    """Tek sorgu modu."""
    print(f"\n❓ Sorgu: {query}")
    print_separator()

    start = time.time()
    response = run_agent(query)
    elapsed = time.time() - start

    print_separator()
    print(f"\n💬 Yanıt:\n{response}")
    print(f"\n⏱️  Süre: {elapsed:.1f} saniye")


def run_demo():
    """Demo sorguları çalıştırır."""
    print(BANNER)
    print("🎬 DEMO MODU — Örnek Sorgular Çalıştırılıyor\n")

    for i, demo in enumerate(DEMO_QUERIES, 1):
        print_separator("═")
        print(f"Demo {i}/{len(DEMO_QUERIES)}: {demo['desc']}")
        print(f"Beklenen araçlar: {demo['expected_tools']}")
        print_separator()

        start = time.time()
        response = run_agent(demo["query"])
        elapsed = time.time() - start

        print_separator()
        print(f"\n❓ Sorgu: {demo['query']}")
        print(f"\n💬 Yanıt:\n{response}")
        print(f"\n⏱️  Süre: {elapsed:.1f} saniye\n")

        if i < len(DEMO_QUERIES):
            input("  ↩  Devam etmek için Enter'a basın...")
            print()

    print_separator("═")
    print("✅ Demo tamamlandı!")


def run_interactive():
    """Etkileşimli sohbet modu."""
    print(BANNER)

    print("💡 Örnek sorular:")
    print("   • 'Makine öğrenmesi dersinin ön koşulları nelerdir?'")
    print("   • 'Hangi seçmeli dersleri 3. yılda alabilirim?'")
    print("   • 'Yapay zeka alanında kariyer için hangi dersler önemli?'")
    print("   • '2024 Python en son sürümü ne ve üniversitede hangi derslerde kullanılıyor?'")
    print()

    while True:
        try:
            print_separator()
            user_input = input("❓ Sorunuz: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "çıkış", "q"):
                print("\n👋 Görüşmek üzere!")
                break

            print()
            start = time.time()
            response = run_agent(user_input)
            elapsed = time.time() - start

            print_separator()
            print(f"\n💬 Yanıt:\n{response}")
            print(f"\n⏱️  Yanıt süresi: {elapsed:.1f} saniye\n")

        except KeyboardInterrupt:
            print("\n\n👋 Görüşmek üzere!")
            break
        except Exception as e:
            print(f"\n❌ Hata oluştu: {str(e)}")
            print("Lütfen kurulum adımlarını kontrol edin.")


def main():
    parser = argparse.ArgumentParser(
        description="Üniversite Ders Seçim Asistanı",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python main.py                                    # Etkileşimli mod
  python main.py --demo                             # Demo sorgular
  python main.py --query "ML dersi ön koşulları?"  # Tek sorgu
        """
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Örnek demo sorguları çalıştır"
    )
    parser.add_argument(
        "--query", type=str,
        help="Tek bir sorgu çalıştır"
    )
    parser.add_argument(
        "--skip-check", action="store_true",
        help="Kurulum kontrolünü atla"
    )

    args = parser.parse_args()

    # Kurulum kontrolü
    if not args.skip_check:
        if not check_setup():
            sys.exit(1)

    # Mod seçimi
    if args.query:
        run_single_query(args.query)
    elif args.demo:
        run_demo()
    else:
        run_interactive()


if __name__ == "__main__":
    main()
