"""
Tests for the multi-agent research team, using a fake async LLM.
No API key or network is needed.

Plain pytest can't run `async def` tests on its own, so each test wraps its
async code in asyncio.run(). That avoids an extra dependency (pytest-asyncio).
"""

import asyncio
import json
import threading
import time
from types import SimpleNamespace
import groq
import httpx
from agent.async_core import AsyncAgent
from agent.llm import AsyncGroqLLM, ToolCallRejected
from agent.sources import (SourceRegistry, cited_ids, normalize_citations,
                           render_sources_section)
from agent.team import ResearchTeam, build_research_tools, parse_json
from agent.tools import ToolRegistry

def reply(content):
    return SimpleNamespace(content=content, tool_calls=None)

def tool_call(call_id, name, arguments):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))

def fake_search(query):
    """
    One search result per query, with a URL based on the query.
    """
    slug = query.lower().replace(" ", "-")
    return [{"title": f"About {query}", "href": f"https://docs.example/{slug}",
             "body": f"{query} supports graphs and async."}]

class TeamLLM:
    """
    Fake LLM that plays every role. It recognises the role from the system
    prompt, the same way the real model is told who it is.

    It also counts how many calls are in flight at once, which lets tests
    check concurrency for real instead of trusting the code.
    """
    def __init__(self, subjects=("Alpha", "Beta", "Gamma"), research_delay=0.2,
                 critiques=None, plan_text=None, slow_subjects=(), broken_subjects=()):
        self.subjects = list(subjects)
        self.research_delay = research_delay
        self.critiques = list(critiques or [{"approved": True, "issues": []}])
        self.plan_text = plan_text
        self.slow_subjects = set(slow_subjects)
        self.broken_subjects = set(broken_subjects)
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls = []  # (role, messages)

    async def chat(self, messages, tools=None, json_mode=False):
        system = messages[0]["content"]
        role = next(r for r in ("PLANNER", "RESEARCHER", "WRITER", "CRITIC") if r in system)
        self.calls.append((role, messages))

        if role == "PLANNER":
            return reply(self.plan_text or json.dumps({
                "subjects": [{"name": s, "question": f"Research {s}"} for s in self.subjects],
                "dimensions": ["architecture", "async support"],
            }))

        if role == "RESEARCHER":
            subject = next(s for s in self.subjects if f"subject is: {s}." in system)
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            try:
                if subject in self.broken_subjects:
                    raise RuntimeError("API down")
                await asyncio.sleep(10 if subject in self.slow_subjects else self.research_delay)
            finally:
                self.in_flight -= 1
            # First call: search. Second call (after the tool result): answer,
            # citing the id that appeared in the tool result.
            if messages[-1]["role"] != "tool":
                return SimpleNamespace(content=None, tool_calls=[
                    tool_call(f"c-{subject}", "web_search", json.dumps({"query": subject}))])
            source_id = messages[-1]["content"].split("]")[0].lstrip("[")
            return reply(f"- architecture: {subject} uses graphs [{source_id}]")

        if role == "WRITER":
            if messages[-1]["role"] == "user" and "fact-checker" in messages[-1]["content"]:
                return reply("# Revised report\nAlpha uses graphs [S1].")
            return reply("# Draft report\nAlpha uses graphs [S1]. Beta is fast [S2].")

        # CRITIC
        return reply(json.dumps(self.critiques.pop(0) if self.critiques
                                else {"approved": True, "issues": []}))

    def count(self, role):
        return sum(1 for r, _ in self.calls if r == role)

def run_team(llm, **kwargs):
    team = ResearchTeam(llm, search_fn=fake_search, **kwargs)
    return asyncio.run(team.run("Compare Alpha, Beta and Gamma"))

# Concurrency Tests
def test_researchers_run_concurrently():
    """
    Three researchers that each wait 0.2s per LLM call finish in about the time
    of one, not three, and were all in flight at the same moment.
    """
    llm = TeamLLM(research_delay=0.2)
    started = time.perf_counter()
    report = run_team(llm, max_concurrency=3)
    elapsed = time.perf_counter() - started

    assert llm.max_in_flight == 3
    # Each researcher makes 2 calls (search, answer): 0.4s each. One after
    # another would take 1.2s; concurrently it takes about 0.4s.
    assert elapsed < 0.9
    assert report.stats["research_sequential_seconds"] > report.stats["research_wall_seconds"]

def test_semaphore_limits_concurrency():
    """
    With max_concurrency=1, researchers run one at a time even though they
    were all started together by gather().
    """
    llm = TeamLLM(research_delay=0.05)
    run_team(llm, max_concurrency=1)
    assert llm.max_in_flight == 1


def test_one_researcher_per_subject_in_plan_order():
    """
    The planner's subjects each get exactly one researcher, and the findings
    come back in plan order even though they ran concurrently.
    """
    llm = TeamLLM(subjects=("Alpha", "Beta", "Gamma"))
    report = run_team(llm)
    assert [f.subject for f in report.findings] == ["Alpha", "Beta", "Gamma"]
    assert all(f.ok for f in report.findings)

# Fault Tolerance TTests
def test_hung_researcher_times_out_and_team_still_reports():
    """
    A researcher that hangs is cancelled at the timeout, the others still
    finish, and the report lists the gap instead of crashing.
    """
    llm = TeamLLM(slow_subjects={"Gamma"}, research_delay=0.01)
    started = time.perf_counter()
    report = run_team(llm, researcher_timeout=0.3)

    assert time.perf_counter() - started < 3  # didn't wait the full 10s
    gamma = report.findings[2]
    assert not gamma.ok and "timed out" in gamma.error
    assert report.stats["researchers_ok"] == 2
    assert "## Research gaps" in report.markdown and "Gamma" in report.markdown

def test_crashing_researcher_is_isolated():
    """
    An exception in one researcher becomes a failed finding; the others are
    unaffected.
    """
    llm = TeamLLM(broken_subjects={"Beta"}, research_delay=0.01)
    report = run_team(llm)
    assert [f.ok for f in report.findings] == [True, False, True]
    assert "RuntimeError" in report.findings[1].error

def test_all_researchers_failing_raises():
    """
    If nobody found anything, the team stops instead of letting the writer
    invent a report.
    """
    llm = TeamLLM(broken_subjects={"Alpha", "Beta", "Gamma"})
    try:
        run_team(llm)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "every researcher failed" in str(e)
    assert llm.count("WRITER") == 0

def test_bad_plan_falls_back_to_single_researcher():
    """
    Unparseable planner output degrades to one researcher on the whole request.
    """
    llm = TeamLLM(subjects=("Compare Alpha, Beta and Gamma",), plan_text="sorry, no JSON here")
    report = run_team(llm)
    assert len(report.findings) == 1
    assert report.findings[0].subject == "Compare Alpha, Beta and Gamma"

def test_planner_subjects_are_capped_and_deduplicated():
    """
    The code enforces max_subjects and drops duplicate names, whatever the
    LLM returns.
    """
    plan = json.dumps({"subjects": [{"name": n} for n in ["Alpha", "alpha", "Beta", "Gamma"]],
                       "dimensions": ["x"]})
    llm = TeamLLM(plan_text=plan, research_delay=0.01)
    report = run_team(llm, max_subjects=2)
    assert [s.name for s in report.plan.subjects] == ["Alpha", "Beta"]

# Critic <-> Writer Loop
def test_approved_draft_is_not_revised():
    """
    If the critic approves the first draft, the writer is called once.
    """
    llm = TeamLLM(research_delay=0.01)
    report = run_team(llm)
    assert llm.count("WRITER") == 1 and llm.count("CRITIC") == 1
    assert report.stats["approved"] and report.stats["revisions"] == 0

def test_critic_issues_trigger_revision_then_recheck():
    """
    Rejected draft -> writer revises -> critic checks the revised draft.
    """
    llm = TeamLLM(research_delay=0.01, critiques=[
        {"approved": False, "issues": [{"claim": "Beta is fast", "problem": "not in S2"}]},
        {"approved": True, "issues": []},
    ])
    report = run_team(llm, max_revisions=1)
    assert llm.count("WRITER") == 2 and llm.count("CRITIC") == 2
    assert report.markdown.startswith("# Revised report")
    assert report.stats["approved"]

def test_revision_loop_is_bounded():
    """
    A critic that never approves can't cause an endless loop; leftover
    issues are listed.
    """
    never = {"approved": False, "issues": [{"claim": "X", "problem": "unsupported"}]}
    llm = TeamLLM(research_delay=0.01, critiques=[never] * 10)
    report = run_team(llm, max_revisions=2)
    assert llm.count("CRITIC") == 3 and llm.count("WRITER") == 3
    assert "## Fact-check notes" in report.markdown

def test_critic_only_sees_cited_sources():
    """
    The critic's prompt contains the cited sources' snippets, keeping it
    grounded and small.
    """
    llm = TeamLLM(research_delay=0.01)
    run_team(llm)
    critic_prompt = next(m for r, m in llm.calls if r == "CRITIC")[1]["content"]
    assert "SOURCES:" in critic_prompt and "[S1]" in critic_prompt
    assert "https://docs.example/" in critic_prompt

def test_unknown_citation_is_flagged_without_llm():
    """
    A citation to a source id that was never found is rejected by code, even if
    the LLM approves.
    """
    llm = TeamLLM(subjects=("Alpha",), research_delay=0.01)  # only S1 will exist

    async def go():
        team = ResearchTeam(llm, search_fn=fake_search)
        sources = SourceRegistry()
        sources.add("A", "https://a", "alpha")
        return await team._critique("Alpha is great [S1]. Beta too [S7].", sources)

    critique = asyncio.run(go())
    assert not critique.approved
    assert any("[S7]" in i["claim"] for i in critique.issues)

def test_sources_section_lists_only_cited_real_sources():
    """
    The final Sources list is built by code from the citations, never from
    LLM text.
    """
    llm = TeamLLM(research_delay=0.01)
    report = run_team(llm)
    sources_part = report.markdown.split("## Sources")[1]
    assert "[S1]" in sources_part and "https://docs.example/" in sources_part
    assert "[S3]" not in sources_part  # found but never cited

# Source Registry Tests
def test_same_url_gets_same_id():
    """
    Finding the same page twice (different case or trailing slash) reuses
    its id.
    """
    sources = SourceRegistry()
    a = sources.add("Docs", "https://x.dev/page", "one")
    b = sources.add("Docs again", "HTTPS://x.dev/page/", "two")
    assert a.id == b.id == "S1" and len(sources) == 1

def test_registry_is_thread_safe():
    """
    Many threads adding at once never hand out the same id twice.
    """
    sources = SourceRegistry()
    
    def worker(n):
        for i in range(200):
            sources.add("t", f"https://site/{n}/{i}", "s")

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(sources) == 1600
    assert {sources.get(f"S{i}").url for i in range(1, 1601)} == {
        f"https://site/{n}/{i}" for n in range(8) for i in range(200)}

def test_cited_ids_formats():
    """
    Citations are found in [S1], [S2, S5] and [S3; S4] forms, in order,
    without repeats.
    """
    text = "a [S1] b [S2, S5] c [S3; S4] d [S1] e S9 storage"
    assert cited_ids(text) == ["S1", "S2", "S5", "S3", "S4"]

def test_render_sources_skips_unknown_ids():
    sources = SourceRegistry()
    sources.add("Real", "https://real", "x")
    section = render_sources_section("claim [S1] and [S4]", sources)
    assert "https://real" in section and "S4" not in section

def test_research_tool_registers_results_with_ids():
    """
    web_search output shows each result with its new id, ready to cite.
    """
    sources = SourceRegistry()
    tools = build_research_tools(sources, search_fn=fake_search)
    out = tools.run("web_search", '{"query": "LangGraph"}')
    assert out.startswith("[S1] About LangGraph")
    assert sources.get("S1").url == "https://docs.example/langgraph"

# JSON parsing Tests
def test_parse_json_handles_fences_and_chatter():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('Sure!\n```json\n{"a": 2}\n```') == {"a": 2}

def test_parse_json_raises_without_json():
    try:
        parse_json("no json")
        assert False
    except ValueError:
        pass

# AsyncAgent Tests
class ScriptedAsyncLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def chat(self, messages, tools=None, json_mode=False):
        self.calls.append((list(messages), tools))
        return self.replies.pop(0)

def slow_tools():
    registry = ToolRegistry()

    @registry.register(name="slow", description="sleeps", parameters={"type": "object", "properties": {}})
    def slow():
        time.sleep(0.3)  # blocking, like a real network call
        return "done"
    return registry


def test_async_agent_runs_parallel_tool_calls_concurrently():
    """
    Three blocking tool calls in one reply run in threads at once: ~0.3s,
    not 0.9s.
    """
    llm = ScriptedAsyncLLM([
        SimpleNamespace(content=None, tool_calls=[tool_call(f"c{i}", "slow", "{}") for i in range(3)]),
        reply("all done"),
    ])
    agent = AsyncAgent("w", llm, slow_tools(), "system")
    started = time.perf_counter()
    assert asyncio.run(agent.run("go")) == "all done"
    assert time.perf_counter() - started < 0.7
    # Results are matched to the right call ids, in order.
    tool_msgs = [m for m in llm.calls[1][0] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["c0", "c1", "c2"]


def test_async_agent_forces_final_answer_at_step_limit():
    """
    Out of tool budget, the agent makes one last call WITHOUT tools and
    returns its answer.
    """
    looping = SimpleNamespace(content=None, tool_calls=[tool_call("c", "slow", "{}")])
    llm = ScriptedAsyncLLM([looping, looping, reply("summary of what I found")])
    agent = AsyncAgent("w", llm, slow_tools(), "system", max_steps=2)
    assert asyncio.run(agent.run("go")) == "summary of what I found"
    last_messages, last_tools = llm.calls[-1]
    assert last_tools is None
    assert "no tools" in last_messages[-1]["content"]


# Invalid tool calls (the "Tool choice is none, but model called a tool" bug)
class RejectingLLM(ScriptedAsyncLLM):
    """
    Scripted LLM where an item can be an exception to raise instead of a reply.
    """
    async def chat(self, messages, tools=None, json_mode=False):
        self.calls.append((list(messages), tools))
        item = self.replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

def test_final_write_up_has_no_tool_history():
    """
    The write-up starts a fresh conversation with the search results as plain
    text: no tool calls or tool messages for the model to imitate.
    """
    looping = SimpleNamespace(content=None, tool_calls=[tool_call("c", "slow", "{}")])
    llm = ScriptedAsyncLLM([looping, looping, reply("notes")])
    asyncio.run(AsyncAgent("w", llm, slow_tools(), "system", max_steps=2).run("go"))
    final_messages, _ = llm.calls[-1]
    assert [m["role"] for m in final_messages] == ["system", "user"]
    assert all("tool_calls" not in m for m in final_messages)
    # the tool results are included as text
    assert "done" in final_messages[1]["content"]

def test_rejected_final_write_up_is_retried():
    """
    If the model still tries a tool during the write-up, it gets one more try.
    """
    looping = SimpleNamespace(content=None, tool_calls=[tool_call("c", "slow", "{}")])
    llm = RejectingLLM([looping, ToolCallRejected('{"name": "web_open"}'), reply("notes")])
    agent = AsyncAgent("w", llm, slow_tools(), "system", max_steps=1)
    assert asyncio.run(agent.run("go")) == "notes"

def test_failed_write_up_falls_back_to_raw_results():
    """
    If the write-up keeps failing, the agent returns its raw search results
    instead of an error, so the research isn't lost.
    """
    looping = SimpleNamespace(content=None, tool_calls=[tool_call("c", "slow", "{}")])
    rejected = ToolCallRejected('{"name": "web_search"}')
    llm = RejectingLLM([looping, rejected, rejected])
    answer = asyncio.run(AsyncAgent("w", llm, slow_tools(), "system", max_steps=1).run("go"))
    assert answer.startswith("Raw search results") and "done" in answer

def test_invalid_tool_mid_research_is_corrected():
    """
    A call to a tool that doesn't exist (like gpt-oss's built-in `web_open`)
    gets a correction message listing the real tools, and the agent carries on.
    """
    llm = RejectingLLM([ToolCallRejected('{"name": "web_open"}'), reply("answer")])
    events = []
    agent = AsyncAgent("w", llm, slow_tools(), "system",
                       on_event=lambda kind, data: events.append(kind))
    assert asyncio.run(agent.run("go")) == "answer"
    correction = llm.calls[1][0][-1]["content"]
    assert "web_open" in correction and "slow" in correction
    assert "tool_rejected" in events

def make_bad_request(code):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    body = {"error": {"message": "...", "code": code,
                      "failed_generation": '{"name": "web_open"}'}}
    return groq.BadRequestError("400", response=httpx.Response(400, request=request), body=body)

def test_groq_tool_use_failed_becomes_tool_call_rejected():
    """
    Groq's 400 "tool_use_failed" is translated into ToolCallRejected; other
    400s are not.
    """
    llm = AsyncGroqLLM("fake-key", "model")

    async def call_with(error):
        async def raise_it(**kwargs):
            raise error
        llm.client.chat.completions.create = raise_it
        await llm.chat([{"role": "user", "content": "hi"}])
    try:
        asyncio.run(call_with(make_bad_request("tool_use_failed")))
        assert False, "expected ToolCallRejected"
    except ToolCallRejected as e:
        assert "web_open" in e.generation
    try:
        asyncio.run(call_with(make_bad_request("something_else")))
        assert False, "expected BadRequestError"
    except groq.BadRequestError:
        pass

def test_rate_limit_hook_reports_429():
    """
    The httpx hook reports a 429 and Groq's retry-after, and ignores other
    responses.
    """
    events = []
    llm = AsyncGroqLLM("fake-key", "model", on_event=lambda kind, data: events.append((kind, data)))
    asyncio.run(llm._report_rate_limit(httpx.Response(429, headers={"retry-after": "7"})))
    asyncio.run(llm._report_rate_limit(httpx.Response(200)))
    assert events == [("rate_limited", {"agent": "groq", "retry_after": "7"})]

# Test proper logging
def test_researchers_are_numbered_in_plan_order():
    """
    Researchers are labelled 1, 2, 3... and each number is announced with
    its subject."""
    events = []
    llm = TeamLLM(research_delay=0.01)
    team = ResearchTeam(llm, search_fn=fake_search,
                        on_event=lambda kind, data: events.append((kind, data)))
    asyncio.run(team.run("Compare Alpha, Beta and Gamma"))
    assigned = [(d["number"], d["subject"]) for k, d in events if k == "assign"]
    assert assigned == [(1, "Alpha"), (2, "Beta"), (3, "Gamma")]
    started = {d["agent"] for k, d in events if k == "phase" and d.get("message") == "started"}
    assert started == {"researcher 1 (Alpha)", "researcher 2 (Beta)", "researcher 3 (Gamma)"}


# Citation formats (the "no Sources list" bug: gpt-oss cites with 【S1】)

def test_normalize_citations_handles_model_styles():
    """Full-width and dagger-style citations become plain [S#]; other text is untouched."""
    text = "a【S1】 b【S2】【S5】 c【S3†L4-L9】 d ［S4］ e [S6†L1] f [S7] g【note】 h 【4†source】"
    assert normalize_citations(text) == (
        "a[S1] b[S2][S5] c[S3] d [S4] e [S6] f [S7] g【note】 h 【4†source】")


def test_lenticular_citations_still_get_sources_and_fact_check():
    """
    Reproduces the dogs-vs-cats run: the writer cites with 【S1】. The report
    must still get a Sources list, and the critic must still see the sources.
    """
    class LenticularWriterLLM(TeamLLM):
        async def chat(self, messages, tools=None, json_mode=False):
            if "WRITER" in messages[0]["content"]:
                self.calls.append(("WRITER", messages))
                return reply("# Draft report\nAlpha uses graphs【S1】. Beta is fast【S2】.")
            return await super().chat(messages, tools, json_mode)

    llm = LenticularWriterLLM(research_delay=0.01)
    report = run_team(llm)

    assert "【" not in report.markdown and "Alpha uses graphs[S1]" in report.markdown
    assert "## Sources" in report.markdown
    assert "https://docs.example/" in report.markdown.split("## Sources")[1]
    critic_prompt = next(m for r, m in llm.calls if r == "CRITIC")[1]["content"]
    assert "(no sources cited)" not in critic_prompt and "[S1]" in critic_prompt

