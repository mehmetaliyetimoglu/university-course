"""
main.py
-------
Üniversite Ders Seçim Asistanı - Komut Satırı Arayüzü

Kullanım:
    python main.py                    # Etkileşimli mod
    python main.py --demo             # Run demo queries
    python main.py --query "question" # Single-query mode
"""
# ...
import sys
import os
import argparse
import time
import warnings
import urllib.error
import urllib.request
import logging
import contextlib
import io

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")
warnings.warn = lambda *args, **kwargs: None
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from agent import run_agent

# ─────────────────────────────────────────────
# Demo queries (cover RAG, web search, and both together)
# ─────────────────────────────────────────────
DEMO_QUERIES = [
    {
        "query": "What are the prerequisites and content of the Machine Learning course?",
        "desc": "RAG query — course information from the local knowledge base",
        "expected_tools": "RAG",
    },
    {
        "query": "What was the average salary of AI engineers in Turkey in 2024?",
        "desc": "Web search query — current salary information",
        "expected_tools": "Web Search",
    },
    {
        "query": "Which math courses should I complete before taking the Deep Learning elective? Also, what is the latest PyTorch version?",
        "desc": "Hybrid query — both RAG (prerequisites) and web search (latest PyTorch version)",
        "expected_tools": "RAG + Web Search",
    },
]

BANNER = """
╔════════════════════════════════════════════════════════╗
║     UNIVERSITY COURSE SELECTION ASSISTANT 🎓          ║
║     LangGraph + LangChain + RAG + Web Search          ║
╚════════════════════════════════════════════════════════╝
Type 'quit', 'exit', or press Ctrl+C to leave.
"""


def print_separator(char="─", width=60):
    print(char * width)


def check_setup():
    """Check whether the project is ready to run."""
    issues = []

    # .env kontrolü
    if not os.path.exists(".env"):
        issues.append("❌ .env file was not found. Copy .env.example to .env.")
    else:
        from dotenv import load_dotenv
        load_dotenv()
        github_token = os.getenv("GITHUB_TOKEN")
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not (github_token or groq_api_key):
            issues.append("❌ GITHUB_TOKEN or GROQ_API_KEY is missing in .env.")
        elif github_token and not groq_api_key:
            try:
                req = urllib.request.Request(
                    "https://api.github.com/user",
                    headers={
                        "Authorization": f"Bearer {github_token}",
                        "Accept": "application/vnd.github+json",
                    },
                )
                with urllib.request.urlopen(req, timeout=10):
                    pass
            except urllib.error.HTTPError as e:
                if e.code == 401:
                    issues.append("❌ GITHUB_TOKEN is invalid or revoked (GitHub: Bad credentials).")
            except Exception:
                pass

    # Vektör deposu kontrolü
    if not os.path.exists("./chroma_db"):
        issues.append("❌ Vector store was not found. Run 'python ingest.py'.")

    if issues:
        print("\n⚠️  Setup issues detected:")
        for issue in issues:
            print(f"   {issue}")
        print()

        if any("GITHUB_TOKEN" in i or "GROQ_API_KEY" in i for i in issues) or any(".env" in i for i in issues):
            print("📋 Quick setup:")
            print("   1. cp .env.example .env")
            print("   2. Open .env and enter your GITHUB_TOKEN or GROQ_API_KEY")
            print("   3. python ingest.py")
            print("   4. python main.py")
            return False

    return True


def run_single_query(query: str):
    """Single-query mode."""
    print(f"\n❓ Query: {query}")
    print_separator()

    start = time.time()
    with contextlib.redirect_stderr(io.StringIO()):
        response = run_agent(query)
    elapsed = time.time() - start

    print_separator()
    print(f"\n💬 Answer:\n{response}")
    print(f"\n⏱️  Time: {elapsed:.1f} seconds")


def run_demo():
    """Run demo queries."""
    print(BANNER)
    print("🎬 DEMO MODE — Running example queries\n")

    for i, demo in enumerate(DEMO_QUERIES, 1):
        print_separator("═")
        print(f"Demo {i}/{len(DEMO_QUERIES)}: {demo['desc']}")
        print(f"Expected tools: {demo['expected_tools']}")
        print_separator()

        start = time.time()
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                response = run_agent(demo["query"])
        except RuntimeError as e:
            print(f"\n❌ Demo stopped:\n{e}")
            return
        elapsed = time.time() - start

        print_separator()
        print(f"\n❓ Query: {demo['query']}")
        print(f"\n💬 Answer:\n{response}")
        print(f"\n⏱️  Time: {elapsed:.1f} seconds\n")

        if i < len(DEMO_QUERIES):
            input("  ↩  Press Enter to continue...")
            print()

    print_separator("═")
    print("✅ Demo completed!")


def run_interactive():
    """Interactive chat mode."""
    print(BANNER)

    print("💡 Example questions:")
    print("   • 'What are the prerequisites for the Machine Learning course?'")
    print("   • 'Which electives can I take in the third year?'")
    print("   • 'Which courses are important for a career in artificial intelligence?'")
    print("   • 'What is the latest Python version and which university courses use Python?'")
    print()

    while True:
        try:
            print_separator()
            user_input = input("❓ Your question: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "çıkış", "q"):
                print("\n👋 See you!")
                break

            print()
            start = time.time()
            with contextlib.redirect_stderr(io.StringIO()):
                response = run_agent(user_input)
            elapsed = time.time() - start

            print_separator()
            print(f"\n💬 Answer:\n{response}")
            print(f"\n⏱️  Response time: {elapsed:.1f} seconds\n")

        except KeyboardInterrupt:
            print("\n\n👋 See you!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please check the setup steps.")


def main():
    parser = argparse.ArgumentParser(
        description="University Course Selection Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                                        # Interactive mode
  python main.py --demo                                                 # Demo queries
  python main.py --query "What are the prerequisites for ML?"            # Single query
        """
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run example demo queries"
    )
    parser.add_argument(
        "--query", type=str,
        help="Run a single query"
    )
    parser.add_argument(
        "--skip-check", action="store_true",
        help="Skip setup checks"
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
