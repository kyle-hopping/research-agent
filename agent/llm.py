"""
llm.py — wrapper made around the Groq API.

The rest of the code only depends on chat() method.
If we ever want to swap Groq for OpenAI, Ollama, etc., you only change
this file.
"""

from groq import Groq

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
