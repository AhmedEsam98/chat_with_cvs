import time
import streamlit as st

import config
from cache_manager import cache_clear, is_redis_available
from generation import answer_stream, check_hallucination
from ingestion import process_cvs
from retrieval import (
    classify_question,
    get_out_of_scope_message,
    retrieve,
    rewrite_query,
)

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

    top_k = st.slider("Chunks retrieved per question", 3, 20, 10)

    # Cache, Re-ranker & Rewriter indicator
    redis_status = "🟢 Redis Connected" if is_redis_available() else "⚪ In-Memory Fallback"
    sem_status = f"⚡ Semantic Active (≥{config.SEMANTIC_CACHE_THRESHOLD})" if getattr(config, "SEMANTIC_CACHE_ENABLED", True) else "Semantic Off"
    rerank_status = "🎯 FlashRank ON" if getattr(config, "RERANKER_ENABLED", True) else "Off"
    rewrite_status = "🔄 Rewriter ON" if getattr(config, "QUERY_REWRITER_ENABLED", True) else "Off"
    router_status = "🧭 Router ON" if getattr(config, "QUESTION_ROUTER_ENABLED", True) else "Off"
    st.caption(f"Cache: **{redis_status}** • **{sem_status}**")
    st.caption(f"Pipeline: **{rerank_status}** • **{rewrite_status}** • **{router_status}**")

    if st.session_state.indexed:
        st.subheader("Indexed CVs")
        for name in st.session_state.indexed:
            st.write(f"• {name}")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- Chat ----------
st.header("2. Chat")
if not st.session_state.indexed:
    st.info(f"Upload and process at least {config.MIN_CVS} CVs from the sidebar to begin.")
else:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("rewritten_query") and m["rewritten_query"] != m.get("original_question"):
                st.caption(f"🔄 **Rewritten Query:** *\"{m['rewritten_query']}\"*")
            if m.get("router_label") == "NOT_CV_RELATED":
                st.caption("🧭 **Question Router:** `NOT_CV_RELATED` (Bypassed search)")
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
                lat_str = f"Retrieval: `{lat['retrieval']:.2f}s` | Generation: `{lat['generation']:.2f}s`"
                if lat.get("rewrite", 0) > 0.05:
                    lat_str = f"Rewrite: `{lat['rewrite']:.2f}s` | " + lat_str
                if lat.get("router", 0) > 0.05:
                    lat_str = f"Router: `{lat['router']:.2f}s` | " + lat_str
                st.caption(f"⏱️ **{lat['total']:.2f}s** ({lat_str})")

    if question := st.chat_input("Ask about the candidates..."):
        with st.chat_message("user"):
            st.markdown(question)
        prior_history = list(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            # 0. Question Router (Domain Classification)
            t_rt0 = time.perf_counter()
            route_label = classify_question(question, history=prior_history)
            t_router = time.perf_counter() - t_rt0

            if route_label == "NOT_CV_RELATED":
                reply = get_out_of_scope_message(question)
                st.markdown(reply)
                st.caption(f"🧭 **Question Router:** `NOT_CV_RELATED` • Bypassed search (`{t_router:.2f}s`)")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply,
                    "original_question": question,
                    "router_label": route_label,
                    "latency": {
                        "router": t_router,
                        "rewrite": 0.0,
                        "retrieval": 0.0,
                        "generation": 0.0,
                        "total": t_router,
                    },
                })
            else:
                # 1. Query Rewriter (Conversational Context + Hybrid Expansion)
                t_rw0 = time.perf_counter()
                search_query = rewrite_query(question, history=prior_history)
                t_rewrite = time.perf_counter() - t_rw0

                # 2. Retrieval
                t0 = time.perf_counter()
                hits = retrieve(search_query, top_k)
                t_retrieval = time.perf_counter() - t0

                # 3. Generation
                t1 = time.perf_counter()
                reply = st.write_stream(
                    answer_stream(question, st.session_state.messages, hits, search_query=search_query))
                t_generation = time.perf_counter() - t1
                t_total = t_router + t_rewrite + t_retrieval + t_generation

                if search_query != question:
                    st.caption(f"🔄 **Rewritten Query:** *\"{search_query}\"*")

                if hits:
                    with st.expander("Sources"):
                        for h in hits:
                            score_badge = f" *(Re-rank Score: `{h['rerank_score']:.4f}`)*" if "rerank_score" in h else ""
                            st.markdown(f"**{h['cv_name']}**{score_badge}")
                            st.caption(h["content"][:300] + "...")

                # Run Hallucination / Faithfulness Audit
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

                lat_str = f"Retrieval: `{t_retrieval:.2f}s` | Generation: `{t_generation:.2f}s`"
                if t_rewrite > 0.05:
                    lat_str = f"Rewrite: `{t_rewrite:.2f}s` | " + lat_str
                if t_router > 0.05:
                    lat_str = f"Router: `{t_router:.2f}s` | " + lat_str
                st.caption(f"⏱️ **{t_total:.2f}s** ({lat_str})")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply,
                    "sources": hits,
                    "original_question": question,
                    "rewritten_query": search_query if search_query != question else None,
                    "router_label": route_label,
                    "eval": eval_result,
                    "latency": {
                        "router": t_router,
                        "rewrite": t_rewrite,
                        "retrieval": t_retrieval,
                        "generation": t_generation,
                        "total": t_total,
                    },
                })