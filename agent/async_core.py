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

EventCallback = Callable[[str, dict], None]
FINALIZE_PROMPT = ("You have used your whole tool budget. Do not call any more "
                   "tools. Write your final answer now, using only what you "
                   "already found.")

class AsyncAgent:
    def __init__(self, name: str, llm, tools, system_prompt: str,
                 max_steps: int = 4, on_event: EventCallback | None = None):
        """
        name:          shown in --verbose output, e.g. "researcher:LangGraph"
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

        for step in range(1, self.max_steps + 1):
            reply = await self.llm.chat(messages, self.tools.schemas())

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

        # Out of tool budget. Instead of returning "I give up" (and wasting the
        # research already done), make one last call WITHOUT tools so the
        # model has to write up what it has. This is "forced finalization".
        self._emit("step_limit", steps=self.max_steps)
        messages.append({"role": "user", "content": FINALIZE_PROMPT})
        reply = await self.llm.chat(messages, None)
        return reply.content or ""

