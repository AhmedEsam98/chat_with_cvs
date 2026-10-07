import argparse
import os
import sys
from pathlib import Path

# Ensure UTF-8 output encoding for Windows consoles (e.g. cp1256 / cp1252)
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to sys.path so modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.chunker import chunk_text
from ingestion.extractor import extract_text


def save_single_md(file_path: Path, extracted: str, chunks: list[str], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_file = out_dir / f"{file_path.stem}.md"

    with open(md_file, "w", encoding="utf-8") as f:
        f.write(f"# Extraction Report: `{file_path.name}`\n\n")
        f.write("## 📊 Summary\n\n")
        f.write(f"- **File Name:** `{file_path.name}`\n")
        f.write(f"- **File Size:** {file_path.stat().st_size:,} bytes\n")
        f.write(f"- **Characters Extracted:** {len(extracted):,}\n")
        f.write(f"- **Words Extracted:** {len(extracted.split()):,}\n")
        f.write(f"- **Total Chunks:** {len(chunks)}\n\n")

        f.write("---\n\n")
        f.write("## 🧩 Chunks Breakdown\n\n")
        for idx, chunk in enumerate(chunks, 1):
            f.write(f"### Chunk {idx} ({len(chunk)} characters)\n\n")
            f.write("```text\n")
            f.write(chunk.strip() + "\n")
            f.write("```\n\n")

        f.write("---\n\n")
        f.write("## 📄 Full Extracted Text\n\n")
        f.write("<details>\n<summary>Click to view full extracted text</summary>\n\n")
        f.write("```text\n")
        f.write(extracted.strip() + "\n")
        f.write("```\n\n")
        f.write("</details>\n")

    return md_file


def save_aggregated_md(results: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "all_extractions_report.md"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# 📚 All CVs Extraction & Chunks Report\n\n")
        f.write(f"Total Files Processed: **{len(results)}**\n\n")

        f.write("## 📋 Overview Table\n\n")
        f.write("| # | File Name | Size (bytes) | Chars | Words | Chunks |\n")
        f.write("|---|-----------|--------------|-------|-------|--------|\n")
        for idx, r in enumerate(results, 1):
            f.write(
                f"| {idx} | `{r['name']}` | {r['size']:,} | {r['chars']:,} | {r['words']:,} | {len(r['chunks'])} |\n"
            )
        f.write("\n---\n\n")

        f.write("## 📑 Detailed Breakdown by CV\n\n")
        for idx, r in enumerate(results, 1):
            f.write(f"## {idx}. `{r['name']}`\n\n")
            f.write(f"- **Size:** {r['size']:,} bytes | **Chars:** {r['chars']:,} | **Chunks:** {len(r['chunks'])}\n\n")

            f.write("### Chunks:\n\n")
            for c_idx, c in enumerate(r["chunks"], 1):
                f.write(f"#### Chunk {c_idx} ({len(c)} chars)\n\n")
                f.write("```text\n")
                f.write(c.strip() + "\n")
                f.write("```\n\n")
            f.write("---\n\n")

    return report_file


def test_file_extraction(file_path: Path, max_preview: int = 400) -> dict | None:
    print("=" * 80)
    print(f"[File] {file_path.name}")
    print(f"[Path] {file_path}")
    print("=" * 80)

    if not file_path.exists():
        print(f"[ERROR] File not found at {file_path}")
        return None

    data = file_path.read_bytes()
    file_size = len(data)
    print(f"[Size] {file_size:,} bytes")

    # 1. Text Extraction
    try:
        extracted = extract_text(file_path.name, data)
        print(f"[SUCCESS] Extraction completed")
        print(f"[Stats] Extracted characters: {len(extracted):,}")
        print(f"[Stats] Extracted words: {len(extracted.split()):,}")
        print("\n--- [Extracted Text Preview] ---")
        preview = extracted[:max_preview] + ("..." if len(extracted) > max_preview else "")
        print(preview)
        print("-" * 32)
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        return None

    # 2. Text Chunking
    try:
        chunks = chunk_text(extracted)
        print(f"\n[Chunks] Total chunks generated: {len(chunks)}")
        for idx, chunk in enumerate(chunks, 1):
            print(f"\n  Chunk #{idx} (Length: {len(chunk)} chars):")
            chunk_prev = chunk[:200] + ("..." if len(chunk) > 200 else "")
            print(f"  | {chunk_prev}")
    except Exception as e:
        print(f"[ERROR] Chunking failed: {e}")
        return None

    return {
        "file_path": file_path,
        "name": file_path.name,
        "size": file_size,
        "chars": len(extracted),
        "words": len(extracted.split()),
        "extracted": extracted,
        "chunks": chunks,
    }


def main():
    parser = argparse.ArgumentParser(description="Test CV text extraction and chunking with Markdown export.")
    parser.add_argument(
        "file",
        nargs="?",
        default=None,
        help="Path to CV file (.pdf, .docx, .txt). If omitted, tests the first file in data/",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all files in data/ directory.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save output to .md file in tests/output/ (enabled by default).",
    )
    parser.add_argument(
        "--no-save",
        dest="save",
        action="store_false",
        help="Do not save .md file.",
    )
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / "data"
    out_dir = PROJECT_ROOT / "tests" / "output"

    if args.all:
        files = sorted([p for p in data_dir.iterdir() if p.suffix.lower() in [".pdf", ".docx", ".txt"]])
        if not files:
            print("[ERROR] No supported files found in data/")
            return

        all_results = []
        for f in files:
            res = test_file_extraction(f)
            if res:
                all_results.append(res)
                if args.save:
                    md_path = save_single_md(f, res["extracted"], res["chunks"], out_dir)
                    print(f"💾 Saved MD: {md_path.name}")
            print("\n")

        if args.save and all_results:
            agg_report = save_aggregated_md(all_results, out_dir)
            print("=" * 80)
            print(f"✅ Aggregated Markdown report generated at:")
            print(f"   file:///{agg_report.as_posix()}")
            print("=" * 80)

    elif args.file:
        target = Path(args.file)
        if not target.is_absolute():
            target = PROJECT_ROOT / target
        res = test_file_extraction(target)
        if res and args.save:
            md_path = save_single_md(target, res["extracted"], res["chunks"], out_dir)
            print("\n" + "=" * 80)
            print(f"✅ Markdown report generated at:")
            print(f"   file:///{md_path.as_posix()}")
            print("=" * 80)
    else:
        files = sorted([p for p in data_dir.iterdir() if p.suffix.lower() in [".pdf", ".docx", ".txt"]])
        if files:
            print(f"[Info] No file specified. Using default: {files[0].name}")
            print(f"[Hint] Specify a file: python tests/test_extraction.py data/{files[0].name}")
            print(f"[Hint] Or test all: python tests/test_extraction.py --all\n")
            res = test_file_extraction(files[0])
            if res and args.save:
                md_path = save_single_md(files[0], res["extracted"], res["chunks"], out_dir)
                print("\n" + "=" * 80)
                print(f"✅ Markdown report generated at:")
                print(f"   file:///{md_path.as_posix()}")
                print("=" * 80)
        else:
            print("[Error] No files found in data/ folder to test.")


if __name__ == "__main__":
    main()
