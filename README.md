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
