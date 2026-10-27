"""
research.py — Run the multi-agent research team from the command line.

How to run:
    python research.py "Compare LangGraph, CrewAI and AutoGen for multi-agent systems"
    python research.py "..." --verbose      # live trace of every agent
    python research.py "..." -v --concurrency 1   # run researchers one at a time,
                                                  # to compare against concurrent

The report is saved to reports/<topic>-<date>.md and its path is printed.
See agent/team.py for how the agents work together.
"""

import argparse
import asyncio
import re
import sys
import time
from datetime import date
from pathlib import Path
from agent.config import settings
from agent.llm import AsyncGroqLLM
from agent.team import ResearchTeam

class TracePrinter:
    """
    Prints agent events with the time since the run started. With several
    researchers running at once, their lines interleave, which is the
    visible proof that they're running concurrently.
    """
    WIDTH = 28  # width of the "who" column

    def __init__(self) -> None:
        self.start = time.perf_counter()

    def __call__(self, kind: str, data: dict) -> None:
        t = f"[{time.perf_counter() - self.start:5.1f}s]"
        agent = data.get("agent", "")
        # keep columns aligned for long subject names
        if len(agent) > self.WIDTH: 
            agent = agent[: self.WIDTH - 2] + "…)"
        who = f"{agent:<{self.WIDTH}}"
        blank = " " * self.WIDTH
        if kind == "phase":
            print(f"{t} {who} {data['message']}")
        elif kind == "plan":
            print(f"{t} {who} subjects: {', '.join(data['subjects'])}")
            print(f"{t} {blank} dimensions: {', '.join(data['dimensions'])}")
        elif kind == "assign":
            print(f"{t} {who} researcher {data['number']} → {data['subject']}")
        elif kind == "tool_call":
            print(f"{t} {who} → {data['name']}({data['arguments']})")
        elif kind == "tool_rejected":
            print(f"{t} {who} invalid tool call rejected, retrying: {data['generation'][:80]}")
        elif kind == "rate_limited":
            print(f"{t} {who} rate limited, retrying in {data['retry_after']}s")
        elif kind == "step_limit":
            print(f"{t} {who} tool budget used up, writing up findings")
        elif kind == "finding":
            status = "done" if data["ok"] else f"FAILED: {data['error']}"
            print(f"{t} {who} {status} ({data['seconds']}s)")
        elif kind == "critique":
            verdict = "approved" if data["approved"] else f"{data['issues']} issue(s) found"
            print(f"{t} {who} {verdict}")
        elif kind == "warning":
            print(f"{t} {who} warning: {data['message']}")

def slugify(text: str, max_len: int = 50) -> str:
    """
    'Compare LangGraph & CrewAI!' -> 'compare-langgraph-crewai'
    """
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:max_len].rstrip("-") or "report"

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent research team.")
    parser.add_argument("request", help='e.g. "Compare LangGraph, CrewAI and AutoGen"')
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="show what every agent is doing")
    parser.add_argument("--concurrency", type=int, default=settings.team_max_concurrency,
                        help="researchers running at once (default from .env)")
    parser.add_argument("--out", default="reports", help="folder for the report")
    args = parser.parse_args()
    # One printer for everything, so all events share the same start time.
    printer = TracePrinter() if args.verbose else None
    try:
        llm = AsyncGroqLLM(api_key=settings.groq_api_key, model=settings.team_model,
                           on_event=printer)
    except ValueError as e:
        sys.exit(f"Setup error: {e}")

    team = ResearchTeam(
        llm,
        max_concurrency=args.concurrency,
        researcher_timeout=settings.researcher_timeout,
        researcher_max_steps=settings.researcher_max_steps,
        max_revisions=settings.max_revisions,
        on_event=printer,
    )

    print(f"Research team ({settings.team_model}), up to {args.concurrency} researchers at once.\n")
    try:
        # asyncio.run() starts the event loop that schedules all the agents.
        report = asyncio.run(team.run(args.request))
    except KeyboardInterrupt:
        sys.exit("\nCancelled.")
    except Exception as e:
        sys.exit(f"Research failed: {type(e).__name__}: {e}")

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{slugify(args.request)}-{date.today()}.md"
    path.write_text(report.markdown, encoding="utf-8")

    s = report.stats
    print(f"\nReport saved to {path}")
    print(f"  researchers: {s['researchers_ok']} ok, {s['researchers_failed']} failed")
    print(f"  research phase: {s['research_wall_seconds']}s "
          f"(one at a time would be ~{s['research_sequential_seconds']}s)")
    print(f"  sources: {s['sources_cited']} cited of {s['sources_found']} found")
    print(f"  critic: {'approved' if s['approved'] else 'unresolved issues listed in report'}"
          f" after {s['revisions']} revision(s)")
    print(f"  total: {s['total_seconds']}s")

if __name__ == "__main__":
    main()
