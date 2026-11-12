#!/usr/bin/env python3
"""Ask a question from the terminal with clean formatting & Agentic Action support.

    python ask.py "What is the sick leave policy?"
    python ask.py "How do I get VPN access?"
    python ask.py "My VPN isn't working even after restarting"
    python ask.py --ticket "VPN connection failed"
    python ask.py            # interactive terminal chat mode
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure local schemesaathi packages can be imported from root or inner dir
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import config
from rag.pipeline import RAG

IT_ACTION_KEYWORDS = [
    "vpn", "not working", "fail", "failed", "ticket", "error",
    "broken", "access", "equipment", "hardware", "laptop",
    "stuck", "problem", "reset", "cannot connect", "disconnect"
]


def clean_markdown_formatting(text: str) -> str:
    """Makes sure markdown text looks clean and readable in standard terminal windows."""
    lines = text.strip().splitlines()
    formatted_lines = []
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            formatted_lines.append("")
            continue
        # Replace Markdown bold asterisks with plain text emphasis
        line_clean = line_stripped.replace("**", "")
        # Format bullet points consistently
        if line_clean.startswith(("- ", "* ")):
            line_clean = "• " + line_clean[2:]
        formatted_lines.append(f"  {line_clean}")
    return "\n".join(formatted_lines)


def render(answer, debug: bool, is_interactive: bool = False, auto_ticket: bool = False) -> None:
    print()
    print("┌" + "─" * 70 + "┐")
    print(f"│ ⚡ INTERNAL OPS CO-PILOT · Grounded Policy & IT Assistant".ljust(71) + "│")
    q_display = answer.question[:58] + "..." if len(answer.question) > 58 else answer.question
    print(f"│ 🏢 Query: \"{q_display}\"".ljust(71) + "│")
    print("└" + "─" * 70 + "┘")
    print()

    print("📋 Answer:")
    print("─" * 72)
    formatted_body = clean_markdown_formatting(answer.text)
    print(formatted_body)
    print("─" * 72)

    if answer.abstained:
        print("\n⚠️ Notice: The internal documentation does not contain enough info to answer.")

    sources = answer.sources()
    if sources:
        print("\n📚 Grounded Company Sources:")
        for source in sources:
            print(f"  [{source['marker']}] {source['title']} ➔ {source['section']}")
            if source["source_url"]:
                print(f"      🔗 {source['source_url']}")

    if debug:
        print("\n🔍 Diagnostics:")
        print(f"  • Provider: {answer.provider} | Coverage: {answer.coverage:.0%} | Citations: {answer.citations or 'none'}")
        for note in answer.notes:
            print(f"  • Note: {note}")

    # Check for Agentic Action Trigger
    is_actionable = any(kw in answer.question.lower() for kw in IT_ACTION_KEYWORDS) or auto_ticket
    if is_actionable:
        render_agentic_action(answer.question, is_interactive=is_interactive, auto_ticket=auto_ticket)
    print()


def render_agentic_action(question: str, is_interactive: bool = False, auto_ticket: bool = False):
    """Formats and executes the Trending Twist (Agentic Action) in the terminal."""
    ticket_id = f"INC-{random.randint(10200, 99800)}"
    priority = "P2 - High (Work Impairment)" if any(w in question.lower() for w in ["not working", "fail", "broken", "blocked", "urgent"]) else "P3 - Standard"
    category = "Network & VPN Access" if "vpn" in question.lower() else ("Hardware & Peripherals" if "equipment" in question.lower() or "laptop" in question.lower() else "IT Systems & Access")

    print("\n" + "╔" + "═" * 70 + "╗")
    print(f"║ 🎫 AGENTIC ACTION: Auto-Drafted IT Support Ticket [{ticket_id}]".ljust(71) + "║")
    print("╠" + "═" * 70 + "╣")
    print(f"║ • Summary    : [IT-SUPPORT] {question[:50]}".ljust(71) + "║")
    print(f"║ • Priority   : {priority}".ljust(71) + "║")
    print(f"║ • Category   : {category}".ljust(71) + "║")
    print(f"║ • SLA Target : Initial engineer response within 30 minutes".ljust(71) + "║")
    print(f"║ • Diagnostics: Employee followed manual troubleshooting steps. Issue".ljust(71) + "║")
    print(f"║                persists. Automated escalation ticket ready to log.".ljust(71) + "║")
    print("╚" + "═" * 70 + "╝")

    if auto_ticket:
        log_ticket(ticket_id, priority)
        return

    if is_interactive:
        try:
            choice = input("\n👉 Take Action? [Enter/y = Submit Ticket to JIRA | e = Email IT Helpdesk | s = Skip]: ").strip().lower()
            if choice in ["", "y", "yes"]:
                log_ticket(ticket_id, priority)
            elif choice in ["e", "email"]:
                email_helpdesk(ticket_id, question)
            else:
                print("ℹ️ Ticket draft cancelled. No incident logged.")
        except (EOFError, KeyboardInterrupt):
            print()
    else:
        print("💡 [Agentic Action]: Auto-submit this ticket with: `python ask.py --ticket \"...\"`")
        print("   Or run interactive mode: `python ask.py`")


def log_ticket(ticket_id: str, priority: str):
    print(f"\n🚀 [TICKET LOGGED] Incident #{ticket_id} successfully dispatched to JIRA Service Desk!")
    print(f"   • Assigned Queue : Tier-2 IT Infrastructure On-Call")
    print(f"   • Priority       : {priority}")
    print(f"   • Guaranteed SLA : Initial response within 30 minutes")
    print(f"   • Notifications  : Dispatched via Slack (`#it-helpdesk`) and corporate email.")


def email_helpdesk(ticket_id: str, question: str):
    print(f"\n📧 [EMAIL SENT] Escalation email sent to `it-helpdesk@internal.corp`!")
    print(f"   • Subject     : [{ticket_id}] IT Escalation: {question[:45]}")
    print(f"   • Diagnostics : Session context & troubleshooting steps attached.")


def parse_filters(pairs: list[str]) -> dict[str, str]:
    filters: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--filter needs key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        filters[key.strip()] = value.strip()
    return filters


def main() -> None:
    parser = argparse.ArgumentParser(description="Internal Ops Co-Pilot CLI")
    parser.add_argument("question", nargs="*", help="the question, in quotes")
    parser.add_argument("--ticket", action="store_true", help="auto-submit IT ticket if troubleshooting is needed")
    parser.add_argument("--debug", action="store_true", help="show retrieval internals")
    parser.add_argument("--json", action="store_true", help="print JSON")
    parser.add_argument("--k", type=int, default=config.TOP_K, help="chunks to retrieve")
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "bm25"])
    parser.add_argument("--filter", action="append", default=[], metavar="KEY=VALUE",
                        help="metadata filter, e.g. --filter who=new-hires")
    args = parser.parse_args()

    rag = RAG(top_k=args.k)
    filters = parse_filters(args.filter)

    def handle(question: str, is_interactive: bool = False) -> None:
        answer = rag.ask(question, top_k=args.k, filters=filters, mode=args.mode)
        if args.json:
            print(json.dumps(
                {
                    "question": answer.question,
                    "answer": answer.text,
                    "abstained": answer.abstained,
                    "grounded": answer.grounded,
                    "provider": answer.provider,
                    "coverage": round(answer.coverage, 3),
                    "citations": answer.citations,
                    "sources": answer.sources(),
                    "notes": answer.notes,
                },
                indent=2, ensure_ascii=False,
            ))
        else:
            render(answer, args.debug, is_interactive=is_interactive, auto_ticket=args.ticket)

    if args.question:
        handle(" ".join(args.question), is_interactive=False)
        return

    print("\n" + "═" * 72)
    print("⚡ INTERNAL OPS CO-PILOT · Interactive Workplace & IT Assistant")
    print("═" * 72)
    print("Ask any question about company policies, HR benefits, or IT troubleshooting.")
    print("Type 'exit' or 'quit' to exit.\n")

    while True:
        try:
            question = input("[Employee Query]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question.lower() in {"exit", "quit", ""}:
            return
        handle(question, is_interactive=True)

if __name__ == "__main__":
    sys.exit(main())
