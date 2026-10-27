# 🤖 Tiny Agent

A minimal, tool-using AI agent built **from scratch** in Python — no agent framework. It reasons with an LLM on **Groq**, uses tools (web search, calculator, clock), and has **long-term semantic memory** stored locally in **Chroma**.

It also includes a **multi-agent research team** (`research.py`): a planner splits a comparison request, researcher agents investigate each subject **concurrently with `asyncio`**, a writer drafts a cited report, and a critic fact-checks it against the sources before it's finalized.

```
You: Remember that my favourite language is Go.
TOOL_CALL: [step 1] save_note({"text": "Joshua's favourite language is Go."})
TOOL_RESULT: Saved to memory: Joshua's favourite language is Go.
Agent: Got it — I'll remember that.

You: /reset
(conversation cleared)

You: Which language do I like best?
MEMORY: 1 note(s) added to context
Agent: Go.

You: What's 18% of 2,450?
TOOL_CALL: [step 1] calculator({"expression": "0.18 * 2450"})
TOOL_RESULT: 441
Agent: 441.
```

*(Shown with `--verbose`.)*

## Tech stack (all free)

| Piece | What | Why |
|---|---|---|
| LLM | Groq API (`openai/gpt-oss-120b`) | Free tier, very fast, supports tool calling |
| Vector DB | Chroma (local, on disk) | Free, no server, built-in local embedding model |
| Web search | `ddgs` (DuckDuckGo) | No API key needed |
| Tests | pytest | |

## Features

- **Agent loop from first principles** — the LLM requests tools, the app executes them and feeds results back until there's a final answer (`agent/core.py`).
- **Tool registry** — add a tool with one decorator; JSON schemas are generated alongside the function (`agent/tools.py`).
- **Persistent semantic memory** — notes are embedded and searched by meaning with Chroma; they survive restarts.
- **Automatic recall (RAG)** — every message is checked against memory, and close matches are added to the context, so the agent remembers you even after `/reset` without having to decide to search.
- **Safety built in** — step limit to stop runaway loops, a sandboxed AST calculator (no `eval`), and tool errors returned to the model instead of crashing.
- **Swappable LLM** — the rest of the code only calls `llm.chat()`, so changing providers touches one file.
- **Multi-agent research team** — planner → concurrent researchers → writer → critic, with per-agent timeouts, bounded concurrency and graceful degradation (see below).
- **Tested offline** — pytest suite uses a scripted fake LLM, so tests need no API key or network.

## Project structure

```
tiny-agent/
├── agent/
│   ├── config.py    # settings loaded from .env
│   ├── llm.py       # Groq wrapper (one method: chat)
│   ├── memory.py    # Chroma long-term memory
│   ├── tools.py     # tool registry + tools
│   ├── core.py      # the chat agent loop
│   ├── async_core.py  # async worker loop used by the team
│   ├── sources.py   # shared, thread-safe citation registry
│   └── team.py      # planner / researchers / writer / critic
├── tests/           # offline tests with fake LLMs
├── main.py          # command-line chat
├── research.py      # command-line research team
├── requirements.txt
└── .env.example     # copy to .env and add your key
```

## Setup

Requires Python 3.10 or newer.

```bash
git clone https://github.com/kyle-hopping/tiny-agent.git
cd tiny-agent
python -m venv .venv
# Windows:      .venv\Scripts\Activate.ps1
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # Windows: copy .env.example .env
# put your Groq key (https://console.groq.com/keys) in .env
python main.py --verbose
```

The first time memory is used, Chroma downloads a small embedding model (~80 MB).

In-chat commands: `/reset` clears the conversation (saved notes are kept), `/exit` quits.

## Configuration

All settings live in `.env` (see `.env.example`):

| Variable | Default | What it does |
|---|---|---|
| `GROQ_API_KEY` | — | Required. Your Groq API key. |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq model; must support tool calling. |
| `CHROMA_PATH` | `./chroma_db` | Folder where memory is stored. |
| `MAX_STEPS` | `5` | Most LLM calls per question before the agent gives up. |
| `MEMORY_MAX_DISTANCE` | `1.5` | How close a note must be to be auto-added to context. Lower = stricter. |
| `TEAM_MODEL` | `GROQ_MODEL` | Model for the research team. `llama-3.3-70b-versatile` has a higher free-tier tokens-per-minute limit. |
| `TEAM_MAX_CONCURRENCY` | `3` | Researchers allowed to call Groq at the same moment. |
| `RESEARCHER_TIMEOUT` | `120` | Seconds before a researcher is cancelled and reported as a gap. |
| `RESEARCHER_MAX_STEPS` | `4` | Web-search rounds per researcher before it must write up. |
| `MAX_REVISIONS` | `1` | Writer revisions allowed after the critic finds problems. |

## Run the tests

```bash
pytest -q
```

## How the agent loop works

1. Memory is searched with the user's message; matching notes are added as a system message, then the user's message is appended.
2. The conversation + tool schemas go to the LLM.
3. If the reply contains **tool calls**, each tool runs and its result is appended as a `tool` message (linked by `tool_call_id`), then go back to step 2.
4. If the reply is plain text, that's the final answer.
5. After `MAX_STEPS` rounds without an answer, the agent stops gracefully.

If a turn fails (an API or tool error), everything it added is rolled back, so the conversation stays valid.

## How memory works

Memory is a small retrieval-augmented generation (RAG) setup:

- **Saving** — the model calls `save_note`. The text is embedded locally (all-MiniLM-L6-v2) and stored in Chroma. Saving the same fact twice overwrites it instead of duplicating it.
- **Automatic recall** — before every message, the agent finds the 3 closest notes and keeps only those within `MEMORY_MAX_DISTANCE`. Notes that pass are given to the model as context, so a question like "what's my dog's name?" works without the model choosing to search.
- **Manual recall** — the model can also call `search_notes` with its own query (unfiltered top 3), which helps with vague questions like "what did I tell you before?".

Tuning the cutoff: Chroma's distance here is `2 − 2 × cosine similarity`, so `0` is identical and `2` is unrelated. In testing, correct matches scored about 0.8–1.4 and unrelated notes 1.6 or higher. If the agent misses facts it should know, raise the value. If unrelated notes show up (watch the `MEMORY:` lines with `--verbose`), lower it.

## Multi-agent research team

```bash
python research.py "Compare LangGraph, CrewAI and AutoGen for building multi-agent systems" --verbose
```

```
[  0.0s] planner                      splitting the request
[  2.1s] planner                      subjects: LangGraph, CrewAI, AutoGen
[  2.1s] team                         starting 3 researcher(s), up to 3 at once
[  2.1s] team                         researcher 1 → LangGraph
[  2.1s] team                         researcher 2 → CrewAI
[  2.1s] team                         researcher 3 → AutoGen
[  2.1s] researcher 1 (LangGraph)     started
[  2.1s] researcher 2 (CrewAI)        started
[  2.1s] researcher 3 (AutoGen)       started
[  3.0s] researcher 2 (CrewAI)        → web_search({"query": "CrewAI architecture agents tasks crews"})
[  3.2s] researcher 1 (LangGraph)     → web_search({"query": "LangGraph state graph multi-agent"})
...
```

The interleaved lines are the researchers running at the same time. The report is saved to `reports/`, and the summary compares the research phase's real time with how long it would take one researcher at a time.

```
   request ──▶ PLANNER ──▶ subjects + shared comparison dimensions
                  │
       ┌──────────┼──────────┐      asyncio.gather(), each with a timeout,
       ▼          ▼          ▼      at most TEAM_MAX_CONCURRENCY at once
  RESEARCHER  RESEARCHER  RESEARCHER ──▶ web_search ──▶ SourceRegistry [S1], [S2]…
       └──────────┼──────────┘
                  ▼
               WRITER ──▶ draft with [S#] citations
                  ▼
               CRITIC ──▶ checks each claim against the cited source
          issues │   │ approved
                 ▼   ▼
     WRITER revises   final report + Sources list built by code
```

**Roles.** Every agent is the same LLM with a different system prompt. Only researchers get a tool (web search). The planner and critic reply in JSON so code can act on their output.

**Design decisions**

- **Async where it pays off.** The chat agent waits for one reply at a time, so it stays synchronous. The team has several agents waiting on the network at once, so it uses `asyncio`: researchers run concurrently with `asyncio.gather`, and blocking tools run in threads with `asyncio.to_thread`.
- **Fresh context per worker.** Researchers don't share chat history. Each gets only its subject and the shared dimensions, which keeps prompts small and stops one agent's notes from leaking into another's.
- **Grounded citations.** Every search result is registered in a shared `SourceRegistry` and gets an id. Researchers and the writer cite ids, the critic checks claims against the snippets behind those ids, and the final Sources list is generated by code, so it can't contain a made-up URL.
- **Shared state done safely.** The registry is written by several researchers at once, so it uses a lock, and the same URL always maps to the same id (idempotent), so duplicate finds don't conflict.
- **Fault tolerance.** A hung researcher is cancelled by `asyncio.wait_for`; a crashing one is isolated. Either becomes a "Research gaps" note in the report instead of failing the run. A semaphore bounds concurrency to respect the shared rate limit, and the Groq SDK retries `429`s using Groq's `retry-after`.
- **Deterministic checks before LLM checks.** Citations to ids that don't exist are caught by code. The LLM critic then judges whether real sources support the claims. Issues the critic still disputes after the last revision are listed in a "Fact-check notes" section rather than hidden.
- **Invalid tool calls are recoverable.** Models sometimes call tools that don't exist (gpt-oss was trained with built-in tools like `web_open`), which Groq rejects with a `400 tool_use_failed`. `llm.py` turns that into a provider-neutral `ToolCallRejected`. Mid-research, the agent gets a correction listing the real tools and carries on. For the final write-up, the agent starts a fresh conversation with the search results as plain text, so there's no tool-calling pattern for the model to continue. If even that fails, the raw results (with their `[S#]` ids) are passed to the writer instead of being lost.
- **Visible rate limiting.** An httpx response hook reports every `429` and Groq's `retry-after`, so `--verbose` shows `rate limited, retrying in 12s` instead of an unexplained pause.
- **Bounded loops everywhere.** Researchers have a tool budget (then must write up what they found), and the writer ↔ critic loop has a revision cap.

**Limitations.** The critic checks claims against search-result snippets, not full pages, so it can miss nuance. Fetching and chunking the cited pages would make verification stronger. On Groq's free tier, three concurrent agents can hit the tokens-per-minute limit; the run slows down (the SDK waits and retries) but still completes.
