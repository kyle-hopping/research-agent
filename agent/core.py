"""
core.py — The agent loop.

The agent is essentially an LLM in a loop that can use tools:

    user question
         │
         │
         ▼
    ┌──────────────┐   wants a tool?  ┌──────────────┐
    │  ask the LLM │ ───────yes─────▶ │ run the tool │
    └──────────────┘ ◀── result ───── └──────────────┘
         │
         │ no (plain answer)
         ▼
    final output

The LLM never runs code itself. It only REQUESTS tool calls; we execute them
and feed the results back so it can keep reasoning. We cap the number of
rounds (max_steps in .env) so a confused model can't loop forever and burn
through tokens.
"""

from datetime import date

SYSTEM_PROMPT = """You are Tiny Agent, a concise, helpful assistant.
You have tools. Rules:
- Use `calculator` for any arithmetic. Don't do math in your head.
- Use `web_search` for current events or facts you're unsure about, and mention the source.
- Use `save_note` when the user shares a lasting fact about themselves or asks you to remember something.
- Use `search_notes` when the user refers to something from a past conversation,
  or asks about themselves. Never say you don't know something about the user
  without checking memory first.
- Notes from long-term memory may be provided as context; trust and use them.
- If no tool is needed, just answer.
Today's date is {today}. Keep answers short and clear."""

class Agent:
    def __init__(self, llm, tools, memory=None, memory_max_distance: float | None = None,
                 max_steps: int = 6, on_event=None):
        """
        llm:       anything with a .chat(messages, tools) method (see llm.py)
        tools:     a ToolRegistry with all the tools the agent can use
        memory:    optional Memory; relevant notes are looked up automatically
                   every turn, so recall doesn't depend on the model choosing
                   to call search_notes
        memory_max_distance: cutoff for those automatic lookups (see
                   Memory.search); None adds the top matches unfiltered
        max_steps: maximum LLM calls per user question, set in .env
        on_event:  optional callback(kind, data) so the CLI can show what the
                   agent is doing (used by --verbose). Keeps printing out of here.
        """
        self.llm = llm
        self.tools = tools
        self.memory = memory
        self.memory_max_distance = memory_max_distance
        self.max_steps = max_steps
        # Note: no verbose, on_event will do nothing 
        self.on_event = on_event or (lambda kind, data: None)
        self.reset()

    def reset(self):
        """
        Start a new conversation. Long term memory is kept, but chat history
        is cleared.
        """
        self.messages = [{"role": "system",
                          "content": SYSTEM_PROMPT.format(today=date.today())}]

    def run(self, user_input: str) -> str:
        """
        Answer user message, and use tools as needed.
        """
        start = len(self.messages)  # remember where this turn began
        try:
            # Pull related notes into context up front. Kept inside the turn so
            # a failed turn rolls them back too.
            if self.memory is not None:
                notes = self.memory.search(user_input,
                                           max_distance=self.memory_max_distance)
                if notes:
                    self.on_event("memory", {"notes": notes})
                    self.messages.append({
                        "role": "system",
                        "content": "Possibly relevant notes from long-term memory:\n"
                                   + "\n".join(f"- {n}" for n in notes),
                    })
            self.messages.append({"role": "user", "content": user_input})
            for step in range(1, self.max_steps + 1):
                reply = self.llm.chat(self.messages, self.tools.schemas())

                # Case 1: no tool requested -> this is the final answer.
                if not reply.tool_calls:
                    answer = reply.content or ""
                    self.messages.append({"role": "assistant", "content": answer})
                    return answer

                # Case 2: the model wants tools. First record its request in the
                # history (the API requires the request to appear before results).
                self.messages.append({
                    "role": "assistant",
                    "content": reply.content or "",
                    "tool_calls": [
                        {"id": call.id, "type": "function",
                        "function": {"name": call.function.name,
                                    "arguments": call.function.arguments}}
                        for call in reply.tool_calls
                    ],
                })

                # Run each requested tool and add its result, linked by tool_call_id
                # so the model knows which result belongs to which request.
                for call in reply.tool_calls:
                    self.on_event("tool_call",
                                {
                                    "step": step, "name": call.function.name,
                                    "arguments": call.function.arguments
                                })
                    result = self.tools.run(call.function.name,
                                            call.function.arguments)
                    self.on_event("tool_result", {"name": call.function.name,
                                                "result": result})
                    self.messages.append({"role": "tool", "tool_call_id": call.id,
                                        "content": result})

            # Case 3: the model never gave a final answer within max_steps.
            fallback = ("I couldn't finish that within my step limit. "
                        "Try a simpler or more specific question.")
            self.messages.append({"role": "assistant", "content": fallback})
        except Exception as e:
            # If the model or tools throw an error, don't crash the whole program.
            # Instead, clear this turn and return a fallback answer.
            self.messages = self.messages[:start]
            return f"Something went wrong: {e.__class__.__name__}: {e}"
        return fallback
