import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

# Silence Streamlit bare-mode console warnings
os.environ["STREAMLIT_LOG_LEVEL"] = "error"
warnings.filterwarnings("ignore")
logging.getLogger("streamlit").setLevel(logging.ERROR)

# Ensure UTF-8 output encoding for Windows consoles (e.g. cp1256 / cp1252)
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path so modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval import retrieve


def test_retrieval_query(query: str, k: int = 5, save_output: bool = False):
    print("=" * 80)
    print(f"[Query] \"{query}\"")
    print(f"[Target Top K] {k}")
    print("=" * 80)

    try:
        results = retrieve(query, k)
    except Exception as e:
        print(f"[ERROR] Retrieval failed: {e}")
        return

    print(f"\n[Results] Retrieved Chunks: {len(results)}\n")

    if not results:
        print("[WARNING] No matching chunks found. (Ensure CVs have been uploaded & processed in the app).")
        return

    for idx, r in enumerate(results, 1):
        cv_name = r.get("cv_name", "Unknown")
        content = r.get("content", "")
        print(f"+-- [Rank #{idx}] Source: {cv_name} ({len(content)} chars)")
        print("|")
        for line in content.splitlines():
            print(f"|  {line}")
        print("+--" + "-" * 60)
        print()

    if save_output:
        out_dir = PROJECT_ROOT / "tests" / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_query = "".join(c if c.isalnum() else "_" for c in query)[:40].strip("_")
        out_file = out_dir / f"retrieval_{safe_query}.md"

        with open(out_file, "w", encoding="utf-8") as f:
            f.write(f"# 🔍 Retrieval Results: \"{query}\"\n\n")
            f.write(f"- **Query:** `{query}`\n")
            f.write(f"- **Top K Requested:** {k}\n")
            f.write(f"- **Chunks Retrieved:** {len(results)}\n\n")
            f.write("---\n\n")
            f.write("## 📦 Retrieved Chunks\n\n")
            for idx, r in enumerate(results, 1):
                f.write(f"### Rank {idx}: `{r.get('cv_name')}`\n\n")
                f.write(f"- **Length:** {len(r.get('content', ''))} characters\n\n")
                f.write("```text\n")
                f.write(r.get("content", "").strip() + "\n")
                f.write("```\n\n")

        print(f"[Saved] Markdown results saved to: file:///{out_file.as_posix()}")


def main():
    parser = argparse.ArgumentParser(description="Test Azure AI Search retrieval and inspect chunks.")
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Search query to test. If omitted, enters interactive mode.",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save retrieved chunks to tests/output/.",
    )
    args = parser.parse_args()

    if args.query:
        test_retrieval_query(args.query, k=args.k, save_output=args.save)
    else:
        print("[Mode] Interactive Retrieval Test Mode (Type 'exit' or 'quit' to stop)")
        print("[Examples] 'Who knows Flutter?', 'Python backend Django', 'Graphic design'\n")
        while True:
            try:
                user_q = input("\nEnter query > ").strip()
                if not user_q:
                    continue
                if user_q.lower() in ["exit", "quit", "q"]:
                    break
                test_retrieval_query(user_q, k=args.k, save_output=args.save)
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
