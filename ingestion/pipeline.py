import hashlib
import logging
import re
import threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from ingestion import indexer, storage
from ingestion.chunker import chunk_text
from ingestion.embeddings import embed
from ingestion.extractor import extract_text

from clients import get_openai_client, get_search_client

logger = logging.getLogger(__name__)


def _run_phase_parallel(items, fn, max_workers, phase_name, on_progress, progress_base, progress_weight):
    """Run *fn(key, item)* on every item in parallel, collecting results and failures.

    Returns (results_dict, failed_list) where results_dict maps item key to result.
    """
    workers = min(len(items), max_workers)
    results = {}
    failed = []
    lock = threading.Lock()
    done_count = 0
    total = len(items)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fn, key, item): key for key, item in items.items()}
        for future in as_completed(futures):
            key = futures[future]
            with lock:
                done_count += 1
                try:
                    results[key] = future.result()
                    if on_progress:
                        p = progress_base + progress_weight * (done_count / total)
                        on_progress(p, f"{phase_name}: ✅ {key} ({done_count}/{total})")
                except Exception as exc:
                    failed.append({"name": key, "error": str(exc), "phase": phase_name})
                    logger.error(f"{phase_name} failed for {key}: {exc}")
                    if on_progress:
                        p = progress_base + progress_weight * (done_count / total)
                        on_progress(p, f"{phase_name}: ❌ {key} ({done_count}/{total})")

    return results, failed


COMMON_SKILLS = [
    "Python", "Flutter", "Dart", "RAG", "LangChain", "LLM", "Agentic AI", "Multi-Agent",
    "Docker", "Git", "SQL", "PostgreSQL", "MongoDB", "FastAPI", "Flask", "Django",
    "React", "Node.js", "TensorFlow", "PyTorch", "Scikit-learn", "Bloc", "Provider",
    "Firebase", "MVVM", "Clean Architecture", "REST", "GraphQL", "C++", "Java",
    "JavaScript", "TypeScript", "Linux", "AWS", "Azure", "Figma", "Photoshop", "Illustrator"
]

COMMON_TITLE_PATTERNS = [
    re.compile(r"(?:Trainee\s*[–\-]\s*)?Agentic AI Developer(?:\s+(?:Intern|Trainee))?", re.IGNORECASE),
    re.compile(r"AI\s*&\s*Machine Learning Engineer", re.IGNORECASE),
    re.compile(r"(?:Junior\s+)?AI Engineer", re.IGNORECASE),
    re.compile(r"(?:Mid-Senior\s+|Junior\s+|Freelance\s+)?Flutter Developer", re.IGNORECASE),
    re.compile(r"Graphic Designer", re.IGNORECASE),
    re.compile(r"(?:Backend|Node\.js|Python)\s+Developer", re.IGNORECASE),
    re.compile(r"Frontend Developer", re.IGNORECASE),
    re.compile(r"Software Engineer", re.IGNORECASE),
]


def _extract_candidate_name(filename: str) -> str:
    base = filename.rsplit(".", 1)[0]
    base = re.sub(r"[\(\)]", " ", base)
    base = re.sub(r"([a-z])([A-Z])", r"\1 \2", base)
    base = re.sub(r"[-_\d]+", " ", base)
    base = re.sub(r"\b(?:CV|Resume|FlowCV|Flow|Graphic|Designer|Flutter|Developer)\b", " ", base, flags=re.IGNORECASE)
    base = re.sub(r"\s+", " ", base).strip()
    return base.title() if base else filename


def _extract_skills_from_text(text: str) -> list[str]:
    found = []
    text_lower = text.lower()
    for s in COMMON_SKILLS:
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(s.lower()) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, text_lower):
            found.append(s)
    return found


def _extract_job_titles_from_text(text: str) -> list[str]:
    found = set()
    for pat in COMMON_TITLE_PATTERNS:
        for match in pat.finditer(text):
            t = match.group(0).strip()
            if t:
                found.add(t)
    return sorted(found)


def process_cvs(files, on_progress: Optional[Callable[[float, str], None]] = None, max_workers: int = 4):
    """Phased parallel CV ingestion.

    All CVs pass through each phase together in parallel before the next phase starts:
      Phase 1 – Upload to Blob Storage
      Phase 2 – Extract text (PDF / DOCX / TXT)
      Phase 3 – Chunk text
      Phase 4 – Embed chunks (Azure OpenAI)
      Phase 5 – Index into Azure AI Search

    If a CV fails at any phase, it is dropped from later phases without
    affecting the others.

    Returns a dict with 'succeeded' and 'failed' lists.
    """
    if not files:
        return {"succeeded": [], "failed": []}

    indexer.ensure_index()
    container = storage.ensure_container()
    openai_client = get_openai_client()
    search_client = get_search_client()

    all_failed = []

    # Build initial data map: name -> file object
    file_map = {f.name: f for f in files}

    # ── Phase 1: Upload to Blob Storage (20%) ─────────────────────────
    if on_progress:
        on_progress(0.0, "Phase 1/5: Uploading to Blob Storage...")

    def _upload(name, f):
        storage.upload_cv(container, f.name, f.getvalue())
        return True

    upload_results, upload_failed = _run_phase_parallel(
        file_map, _upload, max_workers,
        "Upload", on_progress, 0.0, 0.20,
    )
    all_failed.extend(upload_failed)
    failed_names = {f["name"] for f in upload_failed}

    # ── Phase 2: Extract text (20%) ───────────────────────────────────
    extract_inputs = {name: f for name, f in file_map.items() if name not in failed_names}

    if on_progress:
        on_progress(0.20, "Phase 2/5: Extracting text...")

    def _extract(name, f):
        return extract_text(f.name, f.getvalue())

    if extract_inputs:
        extract_results, extract_failed = _run_phase_parallel(
            extract_inputs, _extract, max_workers,
            "Extract", on_progress, 0.20, 0.20,
        )
        all_failed.extend(extract_failed)
        failed_names.update(f["name"] for f in extract_failed)
    else:
        extract_results = {}

    # ── Phase 3: Chunk text (10%) ─────────────────────────────────────
    chunk_inputs = {name: text for name, text in extract_results.items() if name not in failed_names}

    if on_progress:
        on_progress(0.40, "Phase 3/5: Chunking text...")

    def _chunk(name, text):
        return chunk_text(text)

    if chunk_inputs:
        chunk_results, chunk_failed = _run_phase_parallel(
            chunk_inputs, _chunk, max_workers,
            "Chunk", on_progress, 0.40, 0.10,
        )
        all_failed.extend(chunk_failed)
        failed_names.update(f["name"] for f in chunk_failed)
    else:
        chunk_results = {}

    # ── Phase 4: Embed chunks (40%) ───────────────────────────────────
    embed_inputs = {name: chunks for name, chunks in chunk_results.items() if name not in failed_names}

    if on_progress:
        on_progress(0.50, "Phase 4/5: Generating embeddings...")

    def _embed(name, chunks):
        contents = [f"[CV: {name}]\n{c}" for c in chunks]
        vectors = embed(contents, client=openai_client)
        return {"contents": contents, "vectors": vectors}

    if embed_inputs:
        embed_results, embed_failed = _run_phase_parallel(
            embed_inputs, _embed, max_workers,
            "Embed", on_progress, 0.50, 0.40,
        )
        all_failed.extend(embed_failed)
        failed_names.update(f["name"] for f in embed_failed)
    else:
        embed_results = {}

    # ── Phase 5: Index into Azure AI Search (10%) ─────────────────────
    index_inputs = {name: data for name, data in embed_results.items() if name not in failed_names}

    if on_progress:
        on_progress(0.90, "Phase 5/5: Indexing into Azure AI Search...")

    def _index(name, data):
        indexer.delete_cv_chunks(name, search_client=search_client)
        candidate_name = _extract_candidate_name(name)
        file_ext = name.rsplit(".", 1)[-1].lower() if "." in name else "pdf"
        now_iso = datetime.now(timezone.utc).isoformat()

        docs = []
        for i in range(len(data["contents"])):
            content = data["contents"][i]
            m_sec = re.search(r"\[Section:\s*([^\]]+?)(?:\s*\(Part\s*\d+/\d+\))?\s*\]", content)
            section_name = m_sec.group(1).strip() if m_sec else "General"

            docs.append({
                "id": hashlib.md5(f"{name}-{i}".encode()).hexdigest(),
                "cv_name": name,
                "candidate_name": candidate_name,
                "section_name": section_name,
                "chunk_index": i,
                "content": content,
                "content_vector": data["vectors"][i],
                "skills": _extract_skills_from_text(content),
                "job_titles": _extract_job_titles_from_text(content),
                "file_type": file_ext,
                "uploaded_at": now_iso,
            })
        if docs:
            indexer.upload_chunks(docs, search_client=search_client)
        return True

    if index_inputs:
        index_results, index_failed = _run_phase_parallel(
            index_inputs, _index, max_workers,
            "Index", on_progress, 0.90, 0.10,
        )
        all_failed.extend(index_failed)
        failed_names.update(f["name"] for f in index_failed)

    # ── Summary ───────────────────────────────────────────────────────
    succeeded = [name for name in file_map if name not in failed_names]

    if on_progress:
        on_progress(1.0, f"Done: {len(succeeded)} succeeded, {len(all_failed)} failed")

    return {"succeeded": succeeded, "failed": all_failed}