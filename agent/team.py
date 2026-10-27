"""
team.py — A multi-agent research team that writes technical "teardowns",
e.g. "Compare LangGraph, CrewAI and AutoGen for building multi-agent systems".

    request
       │
       ▼
   PLANNER ──── splits the request into subjects + shared comparison dimensions
       │
       ├──────────────┬──────────────┐        asyncio.gather():
       ▼              ▼              ▼        all researchers run AT THE SAME TIME,
  RESEARCHER     RESEARCHER     RESEARCHER    each with its own timeout, sharing one
  (LangGraph)    (CrewAI)       (AutoGen)     SourceRegistry for citations
       │              │              │
       └──────────────┴──────┬───────┘
                             ▼
                          WRITER ── drafts the report, citing [S#] ids
                             │
                             ▼
                          CRITIC ── checks every claim against the cited sources
                             │
                 issues? ────┴──── approved
                    │                 │
              WRITER revises          ▼
            (max_revisions)     final report + Sources list built by code

Each role is just a different system prompt. Only the researchers get tools
(web search). The planner and critic answer in JSON so code can act on it.

Fault tolerance, borrowed from distributed systems:
  - Timeout per researcher: a hung worker can't stall the team.
  - Graceful degradation: a failed researcher becomes a "research failed"
    note in the report instead of crashing the whole run.
  - Bounded concurrency: a semaphore caps how many researchers call Groq at
    once, so we don't blow through the free-tier rate limit.
  - Bounded revision loop: writer <-> critic can't ping-pong forever.
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable
from agent.async_core import AsyncAgent
from agent.sources import SourceRegistry, cited_ids, render_sources_section
from agent.tools import ToolRegistry, _duckduckgo_search

EventCallback = Callable[[str, dict], None]

# Role prompts. Each agent is the same LLM with a different job description.
PLANNER_PROMPT = """You are the PLANNER of a research team that writes technical teardowns.
Split the user's request into independent research subjects, one per framework,
product or option being compared (at most {max_subjects}). If the request is about a
single thing, split it into its most important distinct aspects instead.
Also choose 4-6 comparison dimensions that every researcher will cover, so their
findings line up (e.g. architecture, multi-agent support, state & memory,
tool calling, async support, maturity).
Reply with JSON only:
{{"subjects": [{{"name": "...", "question": "what this researcher must find out"}}],
  "dimensions": ["...", "..."]}}"""

RESEARCHER_PROMPT = """You are a RESEARCHER on a team. Your ONLY subject is: {subject}.
Cover each of these dimensions: {dimensions}.
Use web_search with short, focused queries. Prefer official docs and GitHub.
Rules:
- End every factual claim with the id of the search result that supports it, e.g. [S3].
- Only use ids that appeared in YOUR search results. Never invent an id.
- If you can't find something, write "not found" for that dimension. Don't guess.
Reply with concise bullet notes grouped by dimension (under 250 words)."""

WRITER_PROMPT = """You are the WRITER. Turn the researchers' notes into a technical teardown
in Markdown:
1. A title and a 2-3 sentence summary.
2. A comparison table: one row per dimension, one column per subject.
3. One section per subject with its strengths and weaknesses.
4. "When to choose which": short, practical guidance.
Rules:
- Use ONLY facts from the notes, and keep their [S#] citation on every claim.
- If a subject's research failed, say so plainly instead of guessing.
- Do not write a sources list; it is added automatically."""

CRITIC_PROMPT = """You are the CRITIC, a strict fact-checker. You get a draft report and the
sources it cites (id, title, snippet, url).
For each cited claim, check that the cited source's snippet actually supports it.
Also flag important factual claims with no citation.
Ignore style and wording. Only report problems that matter.
Reply with JSON only:
{"approved": true|false,
 "issues": [{"claim": "...", "problem": "...", "fix": "..."}]}
Set approved to true only if there are no issues."""

REVISE_PROMPT = """The fact-checker found these problems in your draft:
{issues}

Rewrite the full report to fix them. If a source doesn't support a claim,
remove or soften the claim; do not invent new support. Keep all valid [S#]
citations. Return the complete revised report only."""


# Data passed between the agents. Plain dataclasses keep each hand-off explicit
# (and type-checked), instead of passing loose dicts around.

@dataclass
class Subject:
    name: str
    question: str

@dataclass
class Plan:
    subjects: list[Subject]
    dimensions: list[str]

@dataclass
class Finding:
    subject: str
    notes: str
    ok: bool
    seconds: float
    error: str | None = None

@dataclass
class Critique:
    approved: bool
    issues: list[dict] = field(default_factory=list)

@dataclass
class Report:
    markdown: str
    plan: Plan
    findings: list[Finding]
    critiques: list[Critique]
    stats: dict

# Helpers
def parse_json(text: str) -> dict:
    """
    Parse an LLM's JSON reply. Models sometimes wrap JSON in ```fences``` or
    add a sentence around it, so if plain parsing fails, fall back to the
    outermost {...} block. Raises ValueError if there's no usable JSON.
    """
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"no JSON object in reply: {text[:80]!r}")
        return json.loads(match.group(0))

def build_research_tools(sources: SourceRegistry,
                         search_fn: Callable | None = None) -> ToolRegistry:
    """
    The researchers' only tool: a web search that registers every result in
    the shared SourceRegistry and shows it with its citation id.
    """
    registry = ToolRegistry()

    @registry.register(
        name="web_search",
        description="Search the web. Each result starts with a source id like [S3]; "
                    "cite that id for any claim you take from the result.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string",
                                     "description": "A short search query."}},
            "required": ["query"],
        },
    )
    def web_search(query: str) -> str:
        results = (search_fn or _duckduckgo_search)(query)
        rendered = [sources.add(r.get("title", ""), r["href"], r.get("body", "")).render()
                    for r in results if r.get("href")]
        return "\n\n".join(rendered) if rendered else "No results found."

    return registry

class ResearchTeam:
    def __init__(self, llm, search_fn: Callable | None = None,
                 max_subjects: int = 4, max_concurrency: int = 3,
                 researcher_timeout: float = 120, researcher_max_steps: int = 4,
                 max_revisions: int = 1, on_event: EventCallback | None = None):
        """
        llm:                  an async LLM (see AsyncGroqLLM); all roles share it
        search_fn:            web search function; tests pass a fake one
        max_subjects:         cap on how many researchers the planner can create
        max_concurrency:      how many researchers may run at the same moment
        researcher_timeout:   seconds before a researcher is cancelled
        researcher_max_steps: tool rounds per researcher
        max_revisions:        how many times the writer may revise after critique
        """
        self.llm = llm
        self.search_fn = search_fn
        self.max_subjects = max_subjects
        self.max_concurrency = max_concurrency
        self.researcher_timeout = researcher_timeout
        self.researcher_max_steps = researcher_max_steps
        self.max_revisions = max_revisions
        self.on_event = on_event or (lambda kind, data: None)

    def _emit(self, kind: str, agent: str, **data) -> None:
        self.on_event(kind, {"agent": agent, **data})

    async def run(self, request: str) -> Report:
        """
        Run the whole pipeline for one request and return the report.
        """
        started = time.perf_counter()
        # A fresh registry per run: ids start at S1 for every report.
        sources = SourceRegistry()

        plan = await self._plan(request)

        research_started = time.perf_counter()
        findings = await self._research_all(plan, sources)
        research_wall = time.perf_counter() - research_started

        if not any(f.ok for f in findings):
            # Nothing to write about. Say so instead of letting the writer guess.
            raise RuntimeError("every researcher failed: "
                               + "; ".join(f"{f.subject}: {f.error}" for f in findings))

        draft = await self._write(request, plan, findings)

        # Critic <-> writer loop. The critic always reviews the latest draft,
        # so the last critique describes the report we actually return.
        critiques: list[Critique] = []
        for round_number in range(self.max_revisions + 1):
            critique = await self._critique(draft, sources)
            critiques.append(critique)
            if critique.approved or round_number == self.max_revisions:
                break
            draft = await self._revise(draft, critique)

        markdown = self._finalize(draft, critiques[-1], findings, sources)
        stats = {
            "total_seconds": round(time.perf_counter() - started, 1),
            # Wall-clock time of the research phase vs. the time the
            # researchers would have taken one after another. The gap is
            # what running them concurrently saved.
            "research_wall_seconds": round(research_wall, 1),
            "research_sequential_seconds": round(sum(f.seconds for f in findings), 1),
            "researchers_ok": sum(f.ok for f in findings),
            "researchers_failed": sum(not f.ok for f in findings),
            "sources_found": len(sources),
            "sources_cited": len(cited_ids(draft)),
            "revisions": len(critiques) - 1,
            "approved": critiques[-1].approved,
        }
        self._emit("done", "team", stats=stats)
        return Report(markdown, plan, findings, critiques, stats)

    # 1. Planning Stage
    async def _plan(self, request: str) -> Plan:
        self._emit("phase", "planner", message="splitting the request")
        messages = [
            {"role": "system", "content": PLANNER_PROMPT.format(max_subjects=self.max_subjects)},
            {"role": "user", "content": request},
        ]
        try:
            data = parse_json((await self.llm.chat(messages, None, json_mode=True)).content)
            subjects, seen = [], set()
            for s in data.get("subjects", []):
                name = str(s.get("name", "")).strip()
                if name and name.lower() not in seen:
                    seen.add(name.lower())
                    subjects.append(Subject(name, str(s.get("question") or request)))
             # Can't trust the LLM with limits
            subjects = subjects[: self.max_subjects]
            dimensions = [str(d) for d in data.get("dimensions", [])][:6]
            if not subjects:
                raise ValueError("plan had no subjects")
        # Bad JSON or an empty plan (JSONDecodeError is a ValueError)
        except ValueError as e:
            # Degrade instead of failing: one researcher on the whole request.
            self._emit("warning", "planner", message=f"bad plan ({e}); using one researcher")
            subjects, dimensions = [Subject(request, request)], []

        plan = Plan(subjects, dimensions or ["overview", "strengths", "weaknesses"])
        self._emit("plan", "planner", subjects=[s.name for s in plan.subjects],
                   dimensions=plan.dimensions)
        return plan

    # 2. Research (concurrent) Stage
    async def _research_all(self, plan: Plan, sources: SourceRegistry) -> list[Finding]:
        tools = build_research_tools(sources, self.search_fn)
        # At most `max_concurrency` researchers are inside `async with` at once;
        # the rest wait their turn. This protects the shared rate limit.
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def research(number: int, subject: Subject) -> Finding:
            name = f"researcher {number} ({subject.name})"
            async with semaphore:
                # The timer starts after we get a slot, so time spent waiting
                # in the queue doesn't count against the researcher's timeout.
                started = time.perf_counter()
                self._emit("phase", name, message="started")
                agent = AsyncAgent(
                    name=name, llm=self.llm, tools=tools,
                    system_prompt=RESEARCHER_PROMPT.format(
                        subject=subject.name, dimensions=", ".join(plan.dimensions)),
                    max_steps=self.researcher_max_steps, on_event=self.on_event)
                try:
                    # Wait_for cancels the researcher if it runs too long: the
                    # pending Groq request is abandoned at its next `await`.
                    # (A search already running in a thread finishes on its own;
                    # its result is simply ignored. Threads can't be killed.)
                    notes = await asyncio.wait_for(agent.run(subject.question),
                                                   self.researcher_timeout)
                    finding = Finding(subject.name, notes, True,
                                      time.perf_counter() - started)
                except asyncio.TimeoutError:
                    finding = Finding(subject.name, "", False, time.perf_counter() - started,
                                      f"timed out after {self.researcher_timeout:g}s")
                except Exception as e:  # one bad worker must not sink the team
                    finding = Finding(subject.name, "", False, time.perf_counter() - started,
                                      f"{type(e).__name__}: {e}")
                self._emit("finding", name, ok=finding.ok, seconds=round(finding.seconds, 1),
                           error=finding.error)
                return finding

        # Researchers are numbered 1, 2, 3... in plan order, so the log says
        # which one is which before they all start talking at once.
        self._emit("phase", "team", message=f"starting {len(plan.subjects)} researcher(s), "
                                            f"up to {self.max_concurrency} at once")
        for number, subject in enumerate(plan.subjects, start=1):
            self._emit("assign", "team", number=number, subject=subject.name)

        # Gather() starts every researcher now and waits for all of them.
        # Results come back in the same order as plan.subjects. research()
        # catches its own errors, so one failure can't cancel the others.
        return list(await asyncio.gather(*(
            research(number, subject)
            for number, subject in enumerate(plan.subjects, start=1))))

    # 3. Write Stage
    async def _write(self, request: str, plan: Plan, findings: list[Finding]) -> str:
        self._emit("phase", "writer", message="drafting the report")
        notes = "\n\n".join(
            f"### {f.subject}\n{f.notes}" if f.ok
            else f"### {f.subject}\nRESEARCH FAILED ({f.error}). Say this in the report."
            for f in findings)
        content = (f"Request: {request}\n"
                   f"Dimensions: {', '.join(plan.dimensions)}\n\n"
                   f"Research notes:\n\n{notes}")
        messages = [{"role": "system", "content": WRITER_PROMPT},
                    {"role": "user", "content": content}]
        return (await self.llm.chat(messages, None)).content or ""

    async def _revise(self, draft: str, critique: Critique) -> str:
        self._emit("phase", "writer", message=f"revising ({len(critique.issues)} issue(s))")
        issues = "\n".join(f"- Claim: {i.get('claim', '')}\n  Problem: {i.get('problem', '')}"
                           f"\n  Fix: {i.get('fix', '')}" for i in critique.issues)
        messages = [{"role": "system", "content": WRITER_PROMPT},
                    {"role": "assistant", "content": draft},
                    {"role": "user", "content": REVISE_PROMPT.format(issues=issues)}]
        return (await self.llm.chat(messages, None)).content or draft

    # 4. Critique Stage
    async def _critique(self, draft: str, sources: SourceRegistry) -> Critique:
        self._emit("phase", "critic", message="fact-checking the draft")

        # Cheap deterministic check first: a citation id that doesn't exist
        # is always wrong, and code catches it with 100% reliability.
        unknown = [i for i in cited_ids(draft) if sources.get(i) is None]
        issues = [{"claim": f"[{i}]", "problem": "cites a source id that doesn't exist",
                   "fix": "remove the claim or cite a real source"} for i in unknown]

        # Then the LLM check. It only sees the sources the draft cites, which
        # keeps the prompt small.
        cited = [s for i in cited_ids(draft) if (s := sources.get(i)) is not None]
        evidence = "\n\n".join(s.render() for s in cited) or "(no sources cited)"
        messages = [{"role": "system", "content": CRITIC_PROMPT},
                    {"role": "user", "content": f"DRAFT:\n{draft}\n\nSOURCES:\n{evidence}"}]
        try:
            data = parse_json((await self.llm.chat(messages, None, json_mode=True)).content)
            issues += [i for i in data.get("issues", []) if isinstance(i, dict)]
            approved = bool(data.get("approved")) and not issues
        except ValueError as e:
            # If the critic's reply is unreadable, don't block the report on it,
            # but don't pretend it was approved either.
            self._emit("warning", "critic", message=f"unreadable critique ({e})")
            approved = not issues

        self._emit("critique", "critic", approved=approved, issues=len(issues))
        return Critique(approved, issues)

    # 5. Finalize Stage
    def _finalize(self, draft: str, last: Critique, findings: list[Finding],
                  sources: SourceRegistry) -> str:
        parts = [draft.strip()]
        if not last.approved and last.issues:
            # Be transparent: list what the critic still disputes.
            parts.append("## Fact-check notes\n\nThe critic could not verify:\n\n" + "\n".join(
                f"- {i.get('claim', '')}: {i.get('problem', '')}" for i in last.issues))
        failed = [f for f in findings if not f.ok]
        if failed:
            parts.append("## Research gaps\n\n" + "\n".join(
                f"- {f.subject}: {f.error}" for f in failed))
        sources_section = render_sources_section(draft, sources)
        if sources_section:
            parts.append(sources_section)
        return "\n\n".join(parts) + "\n"
