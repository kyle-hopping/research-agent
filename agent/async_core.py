"""
async_core.py — An async, single-task version of the agent loop.

core.Agent is a chat assistant: it keeps a conversation history across turns.
The research team needs something different: WORKERS. Each worker gets one
task, works on it with a fresh context, returns a result, and is thrown away.
Fresh context per worker matters in a team:
  - workers can't see (or be confused by) each other's half-finished work,
  - every request stays small, which keeps us under Groq's tokens-per-minute.

Two things are async here:
  1. `await self.llm.chat(...)`: while this worker waits for Groq, other
     workers keep running.
  2. Tools: ToolRegistry.run() is ordinary blocking code (DuckDuckGo search
     uses the network). asyncio.to_thread() runs it in a background thread so
     it doesn't freeze the event loop, and asyncio.gather() runs several tool
     calls from the same reply at the same time.
"""

import asyncio
from typing import Callable
from agent.llm import ToolCallRejected

EventCallback = Callable[[str, dict], None]

FINALIZE_PROMPT = ("You have no tools any more. Write your final answer now, "
                   "using only the search results above. Do not call any tools.")

# Used when the model calls a tool that doesn't exist. gpt-oss in particular
# was trained with built-in tools like `web_open` and sometimes calls them.
INVALID_TOOL_PROMPT = ("Your last tool call was rejected: {generation}\n"
                       "The only tools that exist are: {tools}. Use one of "
                       "them, or answer if you have enough information.")

# Cap on raw search results returned when the write-up itself fails, so the
# writer's prompt stays within the token limit.
MAX_RAW_RESULTS_CHARS = 4000

class AsyncAgent:
    def __init__(self, name: str, llm, tools, system_prompt: str,
                 max_steps: int = 4, on_event: EventCallback | None = None):
        """
        name:          shown in --verbose output, e.g. "researcher 1 (LangGraph)"
        llm:           anything with `async def chat(messages, tools=None)`
        tools:         a ToolRegistry (see tools.py)
        system_prompt: the worker's role and rules
        max_steps:     tool rounds allowed before the agent must answer
        """
        self.name = name
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.on_event = on_event or (lambda kind, data: None)

    def _emit(self, kind: str, **data) -> None:
        """
        Report an event, tagged with this agent's name.
        """
        self.on_event(kind, {"agent": self.name, **data})

    async def run(self, task: str) -> str:
        """
        Work on one task and return the final text answer.
        """
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]

        # Every tool result, kept as plain text for the final write-up.
        results_log: list[str] = []

        for step in range(1, self.max_steps + 1):
            try:
                reply = await self.llm.chat(messages, self.tools.schemas())
            except ToolCallRejected as e:
                # The model asked for a tool we don't have. Tell it what went
                # wrong and let it try again; this attempt uses up one step.
                self._emit("tool_rejected", step=step, generation=e.generation)
                messages.append({"role": "user", "content": INVALID_TOOL_PROMPT.format(
                    generation=e.generation, tools=", ".join(self.tools.names()))})
                continue

            # No tool requested -> this is the final answer.
            if not reply.tool_calls:
                return reply.content or ""

            # Record the request before the results (the API requires it).
            messages.append({
                "role": "assistant",
                "content": reply.content or "",
                "tool_calls": [
                    {"id": call.id, "type": "function",
                     "function": {"name": call.function.name,
                                  "arguments": call.function.arguments}}
                    for call in reply.tool_calls
                ],
            })

            for call in reply.tool_calls:
                self._emit("tool_call", step=step, name=call.function.name,
                           arguments=call.function.arguments)

            # Run every requested tool at once. gather() returns results in
            # the same order as the calls, so zip() pairs them up correctly.
            results = await asyncio.gather(*(
                asyncio.to_thread(self.tools.run, call.function.name,
                                  call.function.arguments)
                for call in reply.tool_calls
            ))

            for call, result in zip(reply.tool_calls, results):
                self._emit("tool_result", name=call.function.name, result=result)
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": result})
                results_log.append(result)

        # Out of tool budget. Instead of returning "I give up" (and wasting the
        # research already done), make one last call WITHOUT tools so the
        # model has to write up what it has. This is "forced finalization".
        self._emit("step_limit", steps=self.max_steps)
        return await self._finalize(task, results_log)

    async def _finalize(self, task: str, results_log: list[str]) -> str:
        """
        Write up the findings without tools.

        We DON'T just reuse the conversation: it's full of tool calls, so the
        model tends to "continue the pattern" and call another tool, which
        Groq rejects. Instead we start a fresh conversation where the search
        results are plain text, so there's no tool-calling pattern to follow.
        """
        findings = "\n\n".join(results_log) or "(no search results)"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Task: {task}\n\n"
                                        f"Your search results:\n\n{findings}\n\n"
                                        f"{FINALIZE_PROMPT}"},
        ]
        for attempt in (1, 2):  # one retry: a model's output varies between calls
            try:
                reply = await self.llm.chat(messages, None)
                return reply.content or ""
            except ToolCallRejected as e:
                self._emit("tool_rejected", step="final", generation=e.generation)

        # Still failing. Hand over the raw results rather than losing them:
        # they carry [S#] ids, so the writer can still cite them.
        self._emit("warning", message="couldn't write up findings; passing on raw results")
        return ("Raw search results (the researcher could not summarize them):\n\n"
                + findings[:MAX_RAW_RESULTS_CHARS])
