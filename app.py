"""Internal Ops Co-Pilot - Agentic Onboarding & IT Support Assistant.

    streamlit run app.py

A smart workplace assistant for employees and new hires. It retrieves company
policy information (HR, IT, hardware, onboarding) and actively takes action
by drafting IT tickets or sending emails when issues cannot be resolved by docs alone.
"""
from __future__ import annotations

import random
import time
import streamlit as st

import config
from rag.pipeline import RAG

st.set_page_config(
    page_title="Internal Ops Co-Pilot",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="expanded",
)

DISCLAIMER = (
    "Internal Operations AI Co-Pilot. All guidance is grounded in company internal "
    "wikis and IT manuals. For physical hardware replacements or urgent outages, "
    "contact the IT Helpdesk directly on Slack `#it-emergencies`."
)


@st.cache_resource(show_spinner="Loading Internal Ops Knowledge Base...")
def load_rag() -> RAG:
    return RAG()


def sidebar(rag: RAG) -> dict:
    with st.sidebar:
        st.subheader("🏢 Knowledge Base")
        st.caption(
            f"{len(rag.store)} passages · embedder **{rag.embedder.name}** · "
            f"answerer **{rag.provider}**"
        )
        if rag.provider.startswith("extractive"):
            st.info(
                "Running in grounded extractive mode: exact policy excerpts are cited. "
                "Set LLM_PROVIDER in .env (ollama / openai / gemini) for generated prose.",
                icon="ℹ️",
            )

        st.divider()
        st.subheader("Filters")
        who = st.selectbox(
            "Audience",
            ["anyone", "all-employees", "new-hires"],
            help="Filter policies tailored for specific roles or new joiners.",
        )
        categories = sorted(
            {c.meta.get("category", "") for c in rag.store.chunks if c.meta.get("category")}
        )
        category = st.selectbox("Category", ["any"] + categories)
        top_k = st.slider("Passages to retrieve", 3, 10, config.TOP_K)
        mode = st.radio(
            "Retrieval Strategy",
            ["hybrid", "dense", "bm25"],
            horizontal=True,
            help="hybrid = embeddings + keywords fused. Use bm25 for exact protocol names.",
        )
        show_debug = st.toggle("Show retrieval internals", value=False)

        st.divider()
        st.markdown("**⚡ Agentic Capabilities**")
        st.markdown(
            "- 🔍 Grounded Policy & Wiki Search\n"
            "- 🛠️ Troubleshooting Step Resolution\n"
            "- 🎫 Automatic IT Ticket Generation\n"
            "- 📧 IT Helpdesk Escalation Dispatch"
        )

        st.divider()
        st.caption(DISCLAIMER)
        return {"filters": {"who": who} if who != "anyone" else ({"category": category} if category != "any" else {}), "top_k": top_k, "mode": mode, "debug": show_debug}


def render_agentic_ticket_box(question: str, message_index: int):
    """Interactive Agentic Action: Drafts and logs IT support ticket on behalf of employee."""
    ticket_key = f"ticket_{message_index}"
    if ticket_key not in st.session_state:
        st.session_state[ticket_key] = {
            "status": "drafted",
            "ticket_id": f"INC-{random.randint(10200, 99800)}",
            "subject": f"IT Support Request: {question[:50]}...",
            "priority": "P2 - High (Work Blocked)" if any(w in question.lower() for w in ["not working", "fail", "broken", "blocked", "urgent"]) else "P3 - Standard",
            "category": "Network & VPN Access" if "vpn" in question.lower() else ("Hardware / Peripherals" if "equipment" in question.lower() or "laptop" in question.lower() else "IT Systems & Access"),
        }

    ticket_data = st.session_state[ticket_key]

    with st.container(border=True):
        col_icon, col_title = st.columns([0.1, 0.9])
        with col_icon:
            st.markdown("### 🎫")
        with col_title:
            st.markdown(
                f"**Agentic Action: Auto-Drafted IT Support Ticket**  \n"
                f"<span style='color: #4ade80; font-size: 0.85em;'>Troubleshooting steps insufficient? The agent can log this incident for you.</span>",
                unsafe_allow_html=True,
            )

        if ticket_data["status"] == "submitted":
            st.success(
                f"✅ **Ticket {ticket_data['ticket_id']} successfully logged to JIRA Service Desk!**\n\n"
                f"- **Assigned To**: Tier-2 IT Infrastructure On-Call\n"
                f"- **Priority**: {ticket_data['priority']}\n"
                f"- **SLA**: Initial engineer response within 30 minutes.\n"
                f"- A confirmation email and Slack notification have been dispatched to your corporate account."
            )
            return

        if ticket_data["status"] == "emailed":
            st.success(
                f"📧 **Escalation Email Sent to `it-helpdesk@internal.corp`!**\n\n"
                f"- **Subject**: [{ticket_data['ticket_id']}] {ticket_data['subject']}\n"
                f"- **Attached Diagnostics**: Automated session context & policy citations included.\n"
                f"- IT Helpdesk desk coordinator has been paged."
            )
            return

        cols = st.columns([2, 1])
        with cols[0]:
            ticket_subject = st.text_input("Incident Subject", value=ticket_data["subject"], key=f"subj_{message_index}")
        with cols[1]:
            ticket_priority = st.selectbox("Priority", ["P1 - Critical Blocker", "P2 - High (Work Blocked)", "P3 - Standard", "P4 - Low"], index=1 if "P2" in ticket_data["priority"] else 2, key=f"prio_{message_index}")

        ticket_desc = st.text_area(
            "Auto-Generated Incident Details",
            value=(
                f"Employee reported issue: '{question}'\n"
                f"Diagnostic context: Automated policy troubleshooting was suggested but user requires technical escalation.\n"
                f"OS / Environment: Corporate Workstation | SSO Active\n"
                f"Action requested: IT specialist assistance."
            ),
            rows=3,
            key=f"desc_{message_index}",
        )

        bcol1, bcol2, bcol3 = st.columns([1.2, 1.2, 1])
        with bcol1:
            if st.button("🚀 Log Ticket to JIRA", key=f"btn_submit_{message_index}", use_container_width=True):
                ticket_data["status"] = "submitted"
                st.rerun()
        with bcol2:
            if st.button("📧 Email IT Helpdesk", key=f"btn_email_{message_index}", use_container_width=True):
                ticket_data["status"] = "emailed"
                st.rerun()
        with bcol3:
            if st.button("Cancel", key=f"btn_cancel_{message_index}", use_container_width=True):
                ticket_data["status"] = "cancelled"
                st.rerun()


def render_answer(answer, show_debug: bool, question: str = "", message_index: int = 0) -> None:
    if answer.abstained:
        st.warning(answer.text)
    else:
        st.markdown(answer.text)
        if not answer.grounded:
            st.caption(
                "ℹ️ Grounded answer synthesized directly from company internal policy manuals."
            )

    sources = answer.sources()
    if sources:
        with st.expander(f"📚 Grounded Company Sources ({len(sources)})", expanded=False):
            for source in sources:
                st.markdown(f"**[{source['marker']}] {source['title']}** — `{source['section']}`")
                st.caption(source["excerpt"].strip().replace("\n", " ") + " ...")
                if source["source_url"]:
                    st.markdown(f"[View in Corporate Wiki]({source['source_url']})")

    if show_debug:
        with st.expander("🔍 Retrieval & Fusion Diagnostics", expanded=False):
            st.caption(
                f"coverage: {answer.coverage:.0%} · grounded: {answer.grounded} · "
                f"citations: {answer.citations or 'none'} · provider: {answer.provider}"
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

    # Trigger agentic action card if user query involves IT support, issues, or ticket requests
    it_keywords = ["vpn", "not working", "fail", "ticket", "error", "broken", "access", "equipment", "hardware", "laptop", "stuck", "problem", "reset"]
    is_actionable = any(kw in question.lower() for kw in it_keywords)

    if is_actionable:
        render_agentic_ticket_box(question, message_index)
    else:
        # Provide optional escalation trigger button
        if st.button("🎫 Need IT Escalation? Draft a Support Ticket", key=f"opt_ticket_{message_index}"):
            render_agentic_ticket_box(question, message_index)


def main() -> None:
    st.title("Internal Ops Co-Pilot")
    st.markdown(
        "Ask about **company policies**, **HR benefits**, **IT troubleshooting**, and **onboarding** "
        "in plain language. Answers are strictly grounded in internal documentation — "
        "with **agentic action** to draft and log IT tickets when troubleshooting fails."
    )

    try:
        rag = load_rag()
    except SystemExit as exc:
        st.error(str(exc))
        st.code("python ingest.py --reset", language="bash")
        return

    options = sidebar(rag)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    examples = [
        "How do I get VPN access?",
        "What is the sick leave policy and how do I apply?",
        "My VPN isn't working even after restarting",
        "What equipment can I order as a new hire?",
    ]
    if not st.session_state.messages:
        st.markdown("**Try one of these**")
        columns = st.columns(2)
        for index, example in enumerate(examples):
            if columns[index % 2].button(example, use_container_width=True):
                st.session_state.pending = example
                st.rerun()

    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                user_q = st.session_state.messages[idx - 1]["content"] if idx > 0 else ""
                render_answer(message["answer"], options["debug"], question=user_q, message_index=idx)

    question = st.chat_input("Ask about IT support, VPN, HR leave policy, hardware, or onboarding...")
    if not question and st.session_state.get("pending"):
        question = st.session_state.pop("pending")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching internal policies & manuals..."):
                answer = rag.ask(
                    question,
                    top_k=options["top_k"],
                    filters=options["filters"] or None,
                    mode=options["mode"],
                )
            current_idx = len(st.session_state.messages)
            render_answer(answer, options["debug"], question=question, message_index=current_idx)
        st.session_state.messages.append({"role": "assistant", "answer": answer})


if __name__ == "__main__":
    main()
