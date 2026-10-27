"""
llm.py — wrapper made around the Groq API.

The rest of the code only depends on chat() method.
If we ever want to swap Groq for OpenAI, Ollama, etc., you only change
this file.
"""

from groq import AsyncGroq, Groq

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


class AsyncGroqLLM:
    """
    Async twin of GroqLLM, used by the multi-agent research team (agent/team.py).

    The single chat loop doesn't need async: it waits for one reply at a time.
    A team does: while one researcher waits on Groq, the event loop can send
    another researcher's request instead of sitting idle. `await` marks the
    points where that hand-off can happen.
    """

    def __init__(self, api_key: str, model: str, temperature: float = 0.2,
                 max_retries: int = 5):
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing.")
        # The Groq SDK already retries 429 (rate limit) and 5xx errors with
        # exponential backoff, and honours Groq's `retry-after` header. Several
        # agents share one free-tier quota, so we allow more retries than the
        # default of 2 instead of writing our own retry loop.
        self.client = AsyncGroq(api_key=api_key, max_retries=max_retries)
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
        response = await self.client.chat.completions.create(**kwargs)
        return response.choices[0].message
