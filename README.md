# 🤖 Tiny Agent

A minimal, tool-using AI agent built **from scratch** in Python — no agent framework. It reasons with an LLM on **Groq**, uses tools (web search, calculator, clock), and has **long-term semantic memory** stored locally in **Chroma**.

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
- **Tested offline** — pytest suite uses a scripted fake LLM, so tests need no API key or network.

## Project structure

```
tiny-agent/
├── agent/
│   ├── config.py    # settings loaded from .env
│   ├── llm.py       # Groq wrapper (one method: chat)
│   ├── memory.py    # Chroma long-term memory
│   ├── tools.py     # tool registry + tools
│   └── core.py      # the agent loop
├── tests/           # offline tests with a fake LLM
├── main.py          # command-line chat
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
