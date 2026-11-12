"""SchemeSaathi - a web UI you can hand to someone who is not a programmer.

    streamlit run app.py

The UI is deliberately thin. All the intelligence is in rag/; this file only
collects a question, a couple of filters, and renders what comes back -
including the retrieved passages, because a user should be able to check the
answer without trusting it.
"""
from __future__ import annotations

import streamlit as st

import config
from rag.pipeline import RAG

st.set_page_config(page_title="SchemeSaathi", page_icon="🇮🇳", layout="centered")

DISCLAIMER = (
    "This is a learning project built on short, hand-written summaries of a few "
    "central schemes. It is **not** official advice and the amounts may be out of "
    "date. Always confirm on the official portal before you act."
)


@st.cache_resource(show_spinner="Loading the index...")
def load_rag() -> RAG:
    return RAG()


def sidebar(rag: RAG) -> dict:
    with st.sidebar:
        st.subheader("Knowledge base")
        st.caption(
            f"{len(rag.store)} passages · embedder **{rag.embedder.name}** · "
            f"answerer **{rag.provider}**"
        )
        if rag.provider.startswith("extractive"):
            st.info(
                "Running in extractive mode: answers are quoted from the "
                "documents. Set LLM_PROVIDER in .env (ollama / openai / gemini / "
                "anthropic) for a written answer.",
                icon="ℹ️",
            )

        st.divider()
        st.subheader("Filters")
        who = st.selectbox("Who is this for?", ["anyone", "rural", "urban"])
        categories = sorted(
            {c.meta.get("category", "") for c in rag.store.chunks if c.meta.get("category")}
        )
        category = st.selectbox("Category", ["any"] + categories)
        top_k = st.slider("Passages to retrieve", 3, 10, config.TOP_K)
        mode = st.radio(
            "Retrieval",
            ["hybrid", "dense", "bm25"],
            horizontal=True,
            help="hybrid = embeddings + keywords fused. Try bm25 for exact scheme names.",
        )
        show_debug = st.toggle("Show retrieval internals", value=False)

        filters = {}
        if who != "anyone":
            filters["who"] = who
        if category != "any":
            filters["category"] = category

        st.divider()
        st.caption(DISCLAIMER)
        return {"filters": filters, "top_k": top_k, "mode": mode, "debug": show_debug}


def render_answer(answer, show_debug: bool) -> None:
    if answer.abstained:
        st.warning(answer.text)
    else:
        st.markdown(answer.text)
        if not answer.grounded:
            st.warning(
                "This answer did not carry citations for most of its sentences. "
                "Treat it as unverified."
            )

    sources = answer.sources()
    if sources:
        with st.expander(f"Sources ({len(sources)})", expanded=False):
            for source in sources:
                st.markdown(f"**[{source['marker']}] {source['title']}** — {source['section']}")
                st.caption(source["excerpt"].strip().replace("\n", " ") + " ...")
                if source["source_url"]:
                    st.markdown(f"Official portal: {source['source_url']}")

    if show_debug:
        with st.expander("Why this answer", expanded=True):
            st.caption(
                f"coverage {answer.coverage:.0%} · grounded {answer.grounded} · "
                f"citations {answer.citations or 'none'} · provider {answer.provider}"
            )
            for note in answer.notes:
                st.caption(f"note: {note}")
            st.table(
                [
                    {
                        "marker": f"S{i}",
                        "document": hit.chunk.doc_id,
                        "section": hit.chunk.section,
                        "fused": round(hit.score, 4),
                        "dense rank": hit.dense_rank,
                        "bm25 rank": hit.bm25_rank,
                    }
                    for i, hit in enumerate(answer.hits, start=1)
                ]
            )


def main() -> None:
    st.title("SchemeSaathi")
    st.caption(
        "Ask about Indian government welfare schemes in plain language. "
        "Every answer is grounded in the documents in `data/schemes/`, with the "
        "passage it came from shown next to it."
    )

    try:
        rag = load_rag()
    except SystemExit as exc:
        st.error(str(exc))
        st.code("python ingest.py", language="bash")
        return

    options = sidebar(rag)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    examples = [
        "I am a small farmer with 2 acres. What money can I get?",
        "My mother is 72. Is she covered for hospital treatment?",
        "I sell vegetables on a footpath. Can I get a loan without collateral?",
        "I am an SC student doing B.Com, family income 1.8 lakh. Any scholarship?",
    ]
    if not st.session_state.messages:
        st.markdown("**Try one of these**")
        columns = st.columns(2)
        for index, example in enumerate(examples):
            if columns[index % 2].button(example, use_container_width=True):
                st.session_state.pending = example
                st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_answer(message["answer"], options["debug"])

    question = st.chat_input("Ask about a scheme...")
    if not question and st.session_state.get("pending"):
        question = st.session_state.pop("pending")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching the documents..."):
                answer = rag.ask(
                    question,
                    top_k=options["top_k"],
                    filters=options["filters"] or None,
                    mode=options["mode"],
                )
            render_answer(answer, options["debug"])
        st.session_state.messages.append({"role": "assistant", "answer": answer})


if __name__ == "__main__":
    main()
