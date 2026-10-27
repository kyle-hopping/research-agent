"""
llm.py — wrapper made around the Groq API.

The rest of the code only depends on chat() method.
If we ever want to swap Groq for OpenAI, Ollama, etc., you only change
this file.
"""

from typing import Callable

from groq import AsyncGroq, BadRequestError, DefaultAsyncHttpxClient, Groq

class GroqLLM:
    def __init__(self, api_key: str, model: str, temperature: float = 0.2):
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing."
            )
        self.client = Groq(api_key=api_key)
        self.model = model
        # Low temperature = more predictable answers (rather pick the right tool
        # than be "creative").
        self.temperature = temperature

    def chat(self, messages: list[dict], tools: list[dict] | None = None):
        """
        Send the conversation so far (+ the list of tools the model may use)
        and return the model's reply message.

        The reply has either:
          - `.content`    -> a normal text answer, or
          - `.tool_calls` -> a request for US to run one or more tools.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools or None,
            # In Groq, model decides when to use tools when set to auto.
            tool_choice="auto" if tools else None,
            temperature=self.temperature,
        )
        return response.choices[0].message

class ToolCallRejected(Exception):
    """
    The model tried to call a tool that wasn't allowed: one that doesn't exist,
    or any tool when none were offered. Groq rejects these with a 400 before
    we ever see the reply.

    It's a provider-neutral exception, so agent code can handle it without
    knowing anything about Groq. `generation` is what the model tried to send.
    """

    def __init__(self, generation: str):
        super().__init__(f"model made an invalid tool call: {generation}")
        self.generation = generation

class AsyncGroqLLM:
    """
    Async twin of GroqLLM, used by the multi-agent research team (agent/team.py).

    The single chat loop doesn't need async: it waits for one reply at a time.
    A team does: while one researcher waits on Groq, the event loop can send
    another researcher's request instead of sitting idle. `await` marks the
    points where that hand-off can happen.
    """

    def __init__(self, api_key: str, model: str, temperature: float = 0.2,
                 max_retries: int = 5,
                 on_event: Callable[[str, dict], None] | None = None):
        """
        on_event: optional callback, told when Groq rate-limits a request
                  ("rate_limited" event), so --verbose can explain long pauses.
        """
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing.")
        self.on_event = on_event or (lambda kind, data: None)
        # The Groq SDK already retries 429 (rate limit) and 5xx errors with
        # exponential backoff, and honours Groq's `retry-after` header. Several
        # agents share one free-tier quota, so we allow more retries than the
        # default of 2 instead of writing our own retry loop.
        # Those retries are silent, which makes a run look frozen. An httpx
        # "response hook" runs on every HTTP response, so we use it to report
        # each 429 and how long Groq asked us to wait.
        http_client = DefaultAsyncHttpxClient(
            event_hooks={"response": [self._report_rate_limit]})
        self.client = AsyncGroq(api_key=api_key, max_retries=max_retries,
                                http_client=http_client)
        self.model = model
        self.temperature = temperature

    async def chat(self, messages: list[dict], tools: list[dict] | None = None,
                   json_mode: bool = False):
        """
        Same contract as GroqLLM.chat(), plus `json_mode`, which asks Groq to
        return a single valid JSON object (used by the planner and critic).
        Optional fields are only sent when needed.
        """
        kwargs: dict = {"model": self.model, "messages": messages,
                        "temperature": self.temperature}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = await self.client.chat.completions.create(**kwargs)
        except BadRequestError as e:
            # Groq reports an invalid tool call as a 400 with code
            # "tool_use_failed". Translate it so callers can react (nudge the
            # model, retry) instead of treating it as a fatal API error.
            error = e.body.get("error", {}) if isinstance(e.body, dict) else {}
            if error.get("code") == "tool_use_failed":
                raise ToolCallRejected(str(error.get("failed_generation", ""))) from e
            raise
        return response.choices[0].message

    async def _report_rate_limit(self, response) -> None:
        """
        httpx response hook: report a 429 before the SDK waits and retries.
        """
        if response.status_code == 429:
            wait = response.headers.get("retry-after", "?")
            self.on_event("rate_limited", {"agent": "groq", "retry_after": wait})
