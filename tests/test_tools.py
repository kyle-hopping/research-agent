"""Unit tests for the tools. Run with:  pytest -q"""

import pytest
from agent.llm import GroqLLM
from agent.tools import build_tools, safe_eval

class FakeMemory:
    """Stands in for Chroma so tests are fast and offline."""
    def __init__(self):
        self.notes = []

    def add(self, text):
        self.notes.append(text)

    def search(self, query, k=3, max_distance=None):
        return [n for n in self.notes if any(w in n.lower() for w in query.lower().split())]

@pytest.fixture
def tools():
    fake_search = lambda q: [{"title": "Result", "body": f"about {q}", "href": "https://x.com"}]
    return build_tools(FakeMemory(), web_search_fn=fake_search)

def test_safe_eval_basic_math():
    """
    safe_eval correctly evaluates simple arithmetic, including operator precedence and parentheses.
    """
    assert safe_eval("2 + 3 * 4") == 14
    assert safe_eval("(10 - 4) / 3") == 2

@pytest.mark.parametrize("bad", ["__import__('os').system('ls')", "open('x')", "9 ** 9999",
                                 "True + 1", "x * 2"])
def test_safe_eval_rejects_dangerous_input(bad):
    """
    safe_eval refuses code, function calls, variable names, booleans and huge exponents.
    """
    with pytest.raises(Exception):
        safe_eval(bad)

def test_calculator_tool(tools):
    """
    The calculator tool returns whole-number results without a trailing .0.
    """
    assert tools.run("calculator", '{"expression": "12.5 * 4"}') == "50"

def test_unknown_tool_returns_error_not_crash(tools):
    """
    Calling a tool that doesn't exist returns an error message instead of raising.
    """
    assert "unknown tool" in tools.run("does_not_exist", "{}")

def test_bad_arguments_return_error_not_crash(tools):
    """
    Invalid JSON arguments return an error message instead of raising.
    """
    assert tools.run("calculator", "not json").startswith("Error")

def test_notes_round_trip(tools):
    """
    A note saved with save_note can be found again with search_notes.
    """
    tools.run("save_note", '{"text": "My dog is named Max."}')
    assert "Max" in tools.run("search_notes", '{"query": "dog"}')

def test_web_search_formats_results(tools):
    """
    web_search includes each result's snippet and link in its output.
    """
    out = tools.run("web_search", '{"query": "groq"}')
    assert "about groq" in out and "https://x.com" in out

def test_schemas_match_registered_tools(tools):
    """
    The tool schemas sent to the LLM list exactly the tools that are registered.
    """
    names = {s["function"]["name"] for s in tools.schemas()}
    assert names == {"calculator", "get_current_datetime", "web_search",
                     "save_note", "search_notes"}

@pytest.mark.parametrize("expr, expected", [("7/2", "3.5"), ("-3", "-3"), ("2**100", str(2**100))])
def test_calculator_formatting(tools, expr, expected):
    """
    The calculator formats decimals, negative numbers and large integers correctly.
    """
    assert tools.run("calculator", f'{{"expression": "{expr}"}}') == expected

def test_divide_by_zero_returns_error(tools):
    """
    Dividing by zero returns an error message instead of crashing.
    """
    assert "ZeroDivisionError" in tools.run("calculator", '{"expression": "1/0"}')

def test_web_search_no_results():
    """
    web_search returns "No results found." when the search finds nothing.
    """
    tools = build_tools(FakeMemory(), web_search_fn=lambda q: [])
    assert tools.run("web_search", '{"query": "x"}') == "No results found."

def test_search_notes_no_matches(tools):
    """
    search_notes returns "No matching notes." when nothing matches.
    """
    assert tools.run("search_notes", '{"query": "nothing"}') == "No matching notes."

def test_missing_api_key_raises():
    """
    GroqLLM raises ValueError when no API key is set, before any network call.
    """
    with pytest.raises(ValueError):
        GroqLLM(api_key="", model="any")
