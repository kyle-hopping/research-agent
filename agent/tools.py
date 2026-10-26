"""
tools.py — Functions the LLM can ask to run.

An LLM can only produce text. To let it use tools, we describe each tol
with a JSON schema (name, what it does, what arguments it takes). The model
reads those descriptions and, when useful, replies "please call X with Y".
Our code runs the function and sends the result back.

The ToolRegistry below keeps each function and its schema together, so
adding a new tool is just: write a function + decorate it.
"""

import ast
import json
import operator
from datetime import datetime
from typing import Callable

class ToolRegistry:
    def __init__(self):
        self._functions: dict[str, Callable] = {}
        self._schemas: list[dict] = []

    def register(self, name: str, description: str, parameters: dict):
        """
        Decorator that adds a function as a tool.

        `parameters` is a JSON Schema describing the arguments. The model
        uses `description` fields to decide WHEN and HOW to call the tool,
        so clear descriptions matter.
        """
        def decorator(fn: Callable) -> Callable:
            self._functions[name] = fn
            self._schemas.append({
                "type": "function",
                "function": {"name": name, "description": description,
                             "parameters": parameters},
            })
            return fn
        return decorator

    def schemas(self) -> list[dict]:
        """
        The tool list in the format the Groq API expects.
        """
        return self._schemas

    def names(self) -> list[str]:
        return list(self._functions)

    def run(self, name: str, arguments_json: str) -> str:
        """
        Execute a tool the model asked for. Always returns a string, even on
        failure — an error message lets the model notice and try again,
        instead of crashing the whole program.
        """
        if name not in self._functions:
            return f"Error: unknown tool '{name}'. Available: {self.names()}"
        try:
            args = json.loads(arguments_json or "{}")
            return str(self._functions[name](**args))
        except Exception as e:
            return f"Error running {name}: {type(e).__name__}: {e}"

# Calculator — ast evaluates math expressions safely, without using eval().
# Never use Python's eval() on model output: it could run arbitrary code.
# Instead we parse the expression into a syntax tree and only allow numbers
# and basic math operators.
_ALLOWED_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.UAdd: operator.pos
}

def safe_eval(expression: str) -> float:
    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
            if isinstance(node.op, ast.Pow) and abs(_eval(node.right)) > 100:
                raise ValueError("exponent too large")
            return _ALLOWED_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"unsupported expression: {ast.dump(node)[:40]}")
    return _eval(ast.parse(expression, mode="eval").body)

# build_tools() wires every tool up. Memory tools need a Memory object, so we
# pass it in (this is "dependency injection" — handy for testing too).
def build_tools(memory, web_search_fn: Callable | None = None) -> ToolRegistry:
    registry = ToolRegistry()

    @registry.register(
        name="calculator",
        description="Evaluate a math expression, e.g. '(12.5 * 4) / 3'. "
                    "Use this for any arithmetic instead of doing it in your head.",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string",
                                          "description": "The math expression."
                                         }},
            "required": ["expression"],
        },
    )
    def calculator(expression: str) -> str:
        result = safe_eval(expression)
        # Show 12.0 as 12, keep decimals otherwise.
        return str(int(result)) if float(result).is_integer() else f"{result:.6g}"

    @registry.register(
        name="get_current_datetime",
        description="Get the current local date and time.",
        parameters={"type": "object", "properties": {}},
    )
    def get_current_datetime() -> str:
        return datetime.now().strftime("%A, %B %d, %Y %H:%M")

    @registry.register(
        name="web_search",
        description="Search the web for current information (news, facts that may "
                    "have changed, things you don't know). Returns top results.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string",
                                     "description": "A short search query."}},
            "required": ["query"],
        },
    )
    def web_search(query: str) -> str:
        search = web_search_fn or _duckduckgo_search
        results = search(query)
        if not results:
            return "No results found."
        # Keep it compact: the model only needs title, snippet and link.
        return "\n\n".join(
            f"{r.get('title', '')}\n{r.get('body', '')}\n{r.get('href', '')}"
            for r in results
        )

    @registry.register(
        name="save_note",
        description="Save a fact to long-term memory so you can recall it in future "
                    "conversations. Use when the user shares something worth "
                    "remembering (preferences, plans, names) or asks you to remember.",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string",
                                    "description": "The fact, as a full sentence."}},
            "required": ["text"],
        },
    )
    def save_note(text: str) -> str:
        memory.add(text)
        return f"Saved to memory: {text}"

    @registry.register(
        name="search_notes",
        description="Search long-term memory for facts saved earlier. Use when the "
                    "user refers to something they told you before.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string",
                                     "description": "What to look for."}},
            "required": ["query"],
        },
    )
    def search_notes(query: str) -> str:
        notes = memory.search(query)
        return "\n".join(f"- {n}" for n in notes) if notes else "No matching notes."
    return registry


def _duckduckgo_search(query: str, max_results: int = 4) -> list[dict]:
    """
    Free web search via DuckDuckGo (no API key). Imported lazily so tests
    and offline runs don't need the network. Will have to test if it works properly.
    """
    from ddgs import DDGS
    return DDGS().text(query, max_results=max_results)
