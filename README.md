# 🤖 Tiny Agent

A minimal, tool-using AI agent built **from scratch** in Python — no agent framework. It reasons with an LLM on **Groq**, uses tools (web search, calculator, clock), and has **long-term semantic memory** stored locally in **Chroma**.

```
You: Remember that my favourite language is Go.
  🔧 [step 1] save_note({"text": "Joshua's favourite language is Go."})
Agent: Got it — I'll remember that.

You: What's 18% of 2,450?
  🔧 [step 1] calculator({"expression": "0.18 * 2450"})
Agent: 441.
```

## Tech stack (all free)

| Piece | What | Why |
|---|---|---|
| LLM | Groq API (`llama-3.3-70b-versatile`) | Free tier, very fast, supports tool calling |
| Vector DB | Chroma (local, on disk) | Free, no server, built-in local embedding model |
| Web search | `ddgs` (DuckDuckGo) | No API key needed |
| Tests | pytest | |

## Features

- **Agent loop from first principles** — the LLM requests tools, the app executes them and feeds results back until there's a final answer (`agent/core.py`).
- **Tool registry** — add a tool with one decorator; JSON schemas are generated alongside the function (`agent/tools.py`).
- **Persistent semantic memory** — notes are embedded and searched by meaning with Chroma; they survive restarts.
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

## Run the tests

```bash
pytest -q
```

## How the agent loop works

1. The user's message is appended to the conversation.
2. The conversation + tool schemas go to the LLM.
3. If the reply contains **tool calls**, each tool runs and its result is appended as a `tool` message (linked by `tool_call_id`), then go back to step 2.
4. If the reply is plain text, that's the final answer.
5. After `MAX_STEPS` rounds without an answer, the agent stops gracefully.