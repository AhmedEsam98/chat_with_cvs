import time
import streamlit as st

import config
from cache_manager import cache_clear, is_redis_available
from generation import answer_stream, check_hallucination, route_question
from ingestion import process_cvs
from retrieval import retrieve

st.set_page_config(page_title="Chat with CVs", page_icon="📄", layout="wide")
st.title("📄 Chat with CVs")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed" not in st.session_state:
    st.session_state.indexed = []

# ---------- Sidebar ----------
with st.sidebar:
    st.header("1. Upload CVs")
    files = st.file_uploader(
        f"Upload at least {config.MIN_CVS} CVs (PDF / DOCX / TXT)",
        type=["pdf", "docx", "txt"], accept_multiple_files=True)
    count = len(files) if files else 0
    st.caption(f"{count} file(s) selected")
    if count and count < config.MIN_CVS:
        st.warning(f"Please upload at least {config.MIN_CVS} CVs.")

    if st.button("Process CVs", type="primary", disabled=count < config.MIN_CVS):
        t_start_ingest = time.perf_counter()
        progress = st.progress(0.0, text="Starting...")
        result = process_cvs(files, on_progress=lambda p, msg: progress.progress(p, text=msg))
        progress.empty()
        ingest_elapsed = time.perf_counter() - t_start_ingest

        # Clear Redis caches so new CVs are reflected in queries
        cache_clear("cv:")
        st.session_state.indexed = result["succeeded"]
        if result["failed"]:
            for fail in result["failed"]:
                st.error(f"❌ **{fail['name']}** failed: {fail['error']}")
        if result["succeeded"]:
            st.success(f"✅ {len(result['succeeded'])} CV(s) processed in **{ingest_elapsed:.2f}s**!")

    st.divider()
    with st.expander("⚙️ Pipeline Configuration", expanded=True):
        st.markdown("##### 🔍 Retrieval & Search")
        top_k = st.slider(
            "Chunks passed to LLM (k)",
            min_value=1,
            max_value=20,
            value=10,
            help="Number of most relevant CV excerpts provided to the model.",
        )
        use_reranker = st.toggle(
            "FlashRank Re-ranker",
            value=getattr(config, "RERANKER_ENABLED", True),
            help="Cross-encoder re-ranking (ms-marco-TinyBERT) for contextual precision.",
        )
        config.RERANKER_ENABLED = use_reranker

        rerank_top_n = st.slider(
            "Candidate Pool (Pre-Rerank)",
            min_value=10,
            max_value=50,
            value=getattr(config, "RERANK_TOP_N", 20),
            disabled=not use_reranker,
            help="Number of initial candidate chunks retrieved from Azure AI Search before re-ranking.",
        )
        config.RERANK_TOP_N = rerank_top_n

        st.markdown("##### ⚡ Semantic Caching")
        use_semantic_cache = st.toggle(
            "Enable Semantic Cache",
            value=getattr(config, "SEMANTIC_CACHE_ENABLED", True),
            help="Reuses answers and retrieval results for semantically equivalent queries.",
        )
        config.SEMANTIC_CACHE_ENABLED = use_semantic_cache

        cache_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.80,
            max_value=0.99,
            value=float(getattr(config, "SEMANTIC_CACHE_THRESHOLD", 0.92)),
            step=0.01,
            disabled=not use_semantic_cache,
            help="Cosine similarity threshold (higher = stricter match requirement).",
        )
        config.SEMANTIC_CACHE_THRESHOLD = cache_threshold

        if st.button("🗑️ Purge Cache", help="Clears Redis query vector and semantic answer caches."):
            cache_clear("cv:")
            st.success("Caches purged successfully!")

        st.markdown("##### 🤖 Generation & Guardrails")
        temperature = st.slider(
            "LLM Temperature",
            min_value=0.0,
            max_value=1.0,
            value=float(getattr(config, "TEMPERATURE", 0.0)),
            step=0.05,
            help="0.0 = deterministic and strictly factual; higher = more varied phrasing.",
        )
        config.TEMPERATURE = temperature

        use_auditor = st.toggle(
            "Faithfulness Auditor",
            value=getattr(config, "AUDITOR_ENABLED", True),
            help="Audits answer claims against CV sources to detect hallucinations.",
        )
        config.AUDITOR_ENABLED = use_auditor

    # Backend Status Indicator
    redis_status = "🟢 Redis Connected" if is_redis_available() else "⚪ In-Memory Fallback"
    st.caption(f"Backend: **{redis_status}** • Model: `{config.CHAT_MODEL}`")

    if st.session_state.indexed:
        st.subheader("Indexed CVs")
        for name in st.session_state.indexed:
            st.write(f"• {name}")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- Chat ----------
if not st.session_state.indexed:
    st.info(f"Upload and process at least {config.MIN_CVS} CVs from the sidebar to begin.")
else:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if "sources" in m and m["sources"]:
                with st.expander("Sources"):
                    for h in m["sources"]:
                        score_badge = f" *(Re-rank Score: `{h['rerank_score']:.4f}`)*" if "rerank_score" in h else ""
                        st.markdown(f"**{h['cv_name']}**{score_badge}")
                        st.caption(h["content"][:300] + "...")
            if "eval" in m and m["eval"]:
                ev = m["eval"]
                score_pct = int(ev.get("score", 1.0) * 100)
                if ev.get("is_grounded", True):
                    st.caption(
                        f"🛡️ **Faithfulness: 🟢 {score_pct}% Grounded** "
                        f"({ev.get('grounded_claims_count', 0)}/{ev.get('total_claims', 0)} claims verified)"
                    )
                else:
                    st.warning(
                        f"⚠️ **Potential Hallucination Detected (Faithfulness: {score_pct}%)**"
                    )
                    if ev.get("hallucinations"):
                        with st.expander("⚠️ Flagged Claims"):
                            for h in ev["hallucinations"]:
                                st.markdown(f"• **Claim:** *{h.get('claim')}*")
                                st.caption(f"  Reason: {h.get('reason')}")
            if "latency" in m:
                lat = m["latency"]
                st.caption(
                    f"⏱️ **{lat['total']:.2f}s** "
                    f"(Retrieval: `{lat['retrieval']:.2f}s` | Generation: `{lat['generation']:.2f}s`)"
                )

    if question := st.chat_input("Ask about the candidates..."):
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            # 1. Question Routing Gate
            route = route_question(question)
            if not route.is_cv_related:
                reply = route.direct_response
                st.markdown(reply)
                badge = "👋 *Greeting*" if route.is_greeting else "🛑 *Non-CV Query (Bypassed search & LLM)*"
                st.caption(f"{badge} • Router: `{route.matched_by}`")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply,
                })
            else:
                # Measure retrieval latency
                t0 = time.perf_counter()
                hits = retrieve(question, top_k)
                t_retrieval = time.perf_counter() - t0

                # Measure generation latency
                t1 = time.perf_counter()
                reply = st.write_stream(
                    answer_stream(question, st.session_state.messages, hits))
                t_generation = time.perf_counter() - t1
                t_total = t_retrieval + t_generation

                if hits:
                    with st.expander("Sources"):
                        for h in hits:
                            score_badge = f" *(Re-rank Score: `{h['rerank_score']:.4f}`)*" if "rerank_score" in h else ""
                            st.markdown(f"**{h['cv_name']}**{score_badge}")
                            st.caption(h["content"][:300] + "...")

                # Run Hallucination / Faithfulness Audit (if enabled)
                eval_result = None
                if getattr(config, "AUDITOR_ENABLED", True):
                    with st.spinner("Auditing answer faithfulness..."):
                        eval_result = check_hallucination(reply, hits)

                    score_pct = int(eval_result.get("score", 1.0) * 100)
                    if eval_result.get("is_grounded", True):
                        st.caption(
                            f"🛡️ **Faithfulness: 🟢 {score_pct}% Grounded** "
                            f"({eval_result.get('grounded_claims_count', 0)}/{eval_result.get('total_claims', 0)} claims verified)"
                        )
                    else:
                        st.warning(
                            f"⚠️ **Potential Hallucination Detected (Faithfulness: {score_pct}%)**"
                        )
                        if eval_result.get("hallucinations"):
                            with st.expander("⚠️ Flagged Claims"):
                                for h in eval_result["hallucinations"]:
                                    st.markdown(f"• **Claim:** *{h.get('claim')}*")
                                    st.caption(f"  Reason: {h.get('reason')}")

                st.caption(
                    f"⏱️ **{t_total:.2f}s** "
                    f"(Retrieval: `{t_retrieval:.2f}s` | Generation: `{t_generation:.2f}s`)"
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply,
                    "sources": hits,
                    "eval": eval_result,
                    "latency": {
                        "retrieval": t_retrieval,
                        "generation": t_generation,
                        "total": t_total,
                    },
                })