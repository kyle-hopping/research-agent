"""
Tests for the agent loop using a fake LLM — no API key or network is needed.
"""

from types import SimpleNamespace
from agent.core import Agent
from agent.memory import Memory
from agent.tools import build_tools

def tool_call(call_id, name, arguments):
    """
    Build an object shaped like a Groq tool call.
    """
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))

class ScriptedLLM:
    """
    Returns pre-written replies in order, and records what it was sent.
    """
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append([dict(m) for m in messages])
        return self.replies.pop(0)

class BrokenLLM:
    def chat(self, messages, tools=None):
        raise RuntimeError("API down")

class TinyEmbedding:
    """
    Deterministic bag-of-letters embedding so Chroma runs offline in tests.
    """
    def __call__(self, input):
        return [[float(t.lower().count(c)) for c in "abcdefghijklmnopqrstuvwxyz"] for t in input]

    def embed_query(self, input):
        return self(input)

    @staticmethod
    def name():
        return "tiny-test-embedding"

    def is_legacy(self):
        return False

    def get_config(self):
        return {}

    def default_space(self):
        return "cosine"

    def supported_spaces(self):
        return ["cosine", "l2", "ip"]

    @staticmethod
    def build_from_config(config):
        return TinyEmbedding()

def make_tools(tmp_path):
    memory = Memory(path=str(tmp_path / "db"), embedding_function=TinyEmbedding())
    return build_tools(memory, web_search_fn=lambda q: []), memory

def test_answers_directly_without_tools(tmp_path):
    """
    When the LLM replies with plain text, the agent returns it as the answer without running any tools.
    """
    tools, _ = make_tools(tmp_path)
    llm = ScriptedLLM([SimpleNamespace(content="Hello!", tool_calls=None)])
    assert Agent(llm, tools).run("hi") == "Hello!"

def test_uses_tool_then_answers(tmp_path):
    """
    The agent runs a requested tool, sends the result back linked by tool_call_id, and returns the LLM's follow-up answer.
    """
    tools, _ = make_tools(tmp_path)
    llm = ScriptedLLM([
        SimpleNamespace(content=None, tool_calls=[tool_call("c1", "calculator", '{"expression": "6*7"}')]),
        SimpleNamespace(content="It's 42.", tool_calls=None),
    ])
    agent = Agent(llm, tools)
    assert agent.run("what is 6 times 7?") == "It's 42."

    # The second LLM call must have received the tool result, linked by id.
    tool_msg = llm.calls[1][-1]
    assert tool_msg == {"role": "tool", "tool_call_id": "c1", "content": "42"}

def test_stops_at_max_steps(tmp_path):
    """
    If the LLM keeps asking for tools, the agent stops after max_steps LLM calls and returns the step-limit message.
    """
    tools, _ = make_tools(tmp_path)
    looping = SimpleNamespace(content=None, tool_calls=[tool_call("c", "get_current_datetime", "{}")])
    llm = ScriptedLLM([looping] * 3)
    answer = Agent(llm, tools, max_steps=3).run("loop forever")
    assert "step limit" in answer
    assert len(llm.calls) == 3

def test_memory_persists_in_chroma(tmp_path):
    """
    A note saved through the save_note tool is written to disk and can be found by a new Memory object on the same folder.
    """
    tools, memory = make_tools(tmp_path)
    llm = ScriptedLLM([
        SimpleNamespace(content=None, tool_calls=[tool_call("c1", "save_note", '{"text": "Joshua likes sushi."}')]),
        SimpleNamespace(content="Noted!", tool_calls=None),
    ])
    Agent(llm, tools).run("remember I like sushi")
    assert memory.count() == 1

    # A brand-new Memory object on the same folder still sees the note.
    reopened = Memory(path=str(tmp_path / "db"), embedding_function=TinyEmbedding())
    assert reopened.search("sushi") == ["Joshua likes sushi."]

def test_error_rolls_back_turn(tmp_path):
    """
    If the LLM raises an error, the agent returns an error message and the conversation history is left unchanged.
    """
    tools, _ = make_tools(tmp_path)
    agent = Agent(BrokenLLM(), tools)
    before = list(agent.messages)
    answer = agent.run("hello")
    assert "RuntimeError" in answer
    # Failed turn should leave no trace
    assert agent.messages == before

def test_duplicate_notes_are_stored_once(tmp_path):
    """
    Saving the same text twice (ignoring case and surrounding spaces) overwrites the note instead of duplicating it.
    """
    memory = Memory(path=str(tmp_path / "db"), embedding_function=TinyEmbedding())

    memory.add("I like sushi")
    # Same note, different case/spacing
    memory.add("  i like SUSHI ")
    assert memory.count() == 1

    # The latest version replaced the original text.
    stored = memory.collection.get()["documents"]
    assert stored == ["  i like SUSHI "]

    # New note
    memory.add("I like ramen")
    assert memory.count() == 2

# Automatic memory recall
# TinyEmbedding uses cosine distance: same letters -> 0, no shared letters -> 1.

def reply(content):
    return SimpleNamespace(content=content, tool_calls=None)

def memory_messages(messages):
    return [m for m in messages if m["role"] == "system" and m["content"].startswith("Possibly relevant notes")]

def test_relevant_note_is_added_before_user_message(tmp_path):
    """
    A note close to the question is sent to the LLM as a system message just before the user's message.
    """
    tools, memory = make_tools(tmp_path)
    memory.add("dog max")
    llm = ScriptedLLM([reply("Max")])
    Agent(llm, tools, memory=memory, memory_max_distance=0.5).run("max dog")

    sent = llm.calls[0]
    assert sent[-2]["role"] == "system" and "dog max" in sent[-2]["content"]
    assert sent[-1] == {"role": "user", "content": "max dog"}

def test_far_note_is_filtered_out(tmp_path):
    """
    A note farther away than memory_max_distance is not added to the LLM's context.
    """
    tools, memory = make_tools(tmp_path)
    memory.add("aaa")
    llm = ScriptedLLM([reply("ok")])
    Agent(llm, tools, memory=memory, memory_max_distance=0.5).run("bbb")
    assert memory_messages(llm.calls[0]) == []

def test_empty_memory_adds_nothing(tmp_path):
    """
    With no saved notes, only the system prompt and the user's message are sent to the LLM.
    """
    tools, memory = make_tools(tmp_path)
    llm = ScriptedLLM([reply("ok")])
    Agent(llm, tools, memory=memory).run("hi")
    assert [m["role"] for m in llm.calls[0]] == ["system", "user"]

def test_no_memory_means_no_lookup(tmp_path):
    """
    An agent created without memory never adds memory notes to the context, even if notes exist.
    """
    tools, memory = make_tools(tmp_path)
    memory.add("dog max")
    llm = ScriptedLLM([reply("ok")])
    Agent(llm, tools).run("max dog")
    assert memory_messages(llm.calls[0]) == []

def test_error_rolls_back_memory_context(tmp_path):
    """
    If the LLM fails, the memory message added for that turn is removed along with the rest of the turn.
    """
    tools, memory = make_tools(tmp_path)
    memory.add("dog max")
    agent = Agent(BrokenLLM(), tools, memory=memory)
    before = list(agent.messages)
    agent.run("max dog")
    assert agent.messages == before

def test_memory_event_is_emitted(tmp_path):
    """
    The agent reports added notes through on_event as a "memory" event, which --verbose prints.
    """
    tools, memory = make_tools(tmp_path)
    memory.add("dog max")
    events = []
    agent = Agent(ScriptedLLM([reply("ok")]), tools, memory=memory,
                  on_event=lambda kind, data: events.append((kind, data)))
    agent.run("max dog")
    assert ("memory", {"notes": ["dog max"]}) in events

def test_recall_survives_reset(tmp_path):
    """
    Reproduces the original bug: save a fact, reset the chat, ask again, and the saved note is still given to the LLM.
    """
    # The original bug: save -> reset -> ask again.
    tools, memory = make_tools(tmp_path)
    llm = ScriptedLLM([
        SimpleNamespace(content=None, tool_calls=[tool_call("c1", "save_note", '{"text": "dog max"}')]),
        reply("Noted!"),
        reply("Max"),
    ])
    agent = Agent(llm, tools, memory=memory, memory_max_distance=0.5)
    agent.run("remember my dog is max")
    agent.reset()
    agent.run("max dog")

    after_reset = llm.calls[2]
    assert len(after_reset) == 3  # system prompt, memory, user
    assert "dog max" in memory_messages(after_reset)[0]["content"]


# Memory.search filtering

def test_search_without_cutoff_returns_top_k(tmp_path):
    """
    Without max_distance, search returns the k closest notes, or every note if there are fewer than k.
    """
    memory = Memory(path=str(tmp_path / "db"), embedding_function=TinyEmbedding())
    for t in ["aaa", "bbb", "ccc", "ddd"]:
        memory.add(t)
    assert len(memory.search("aaa", k=3)) == 3
    assert len(memory.search("aaa", k=10)) == 4

def test_search_cutoff_filters_by_distance(tmp_path):
    """
    max_distance keeps notes within the cutoff and drops the rest; a loose cutoff keeps everything.
    """
    memory = Memory(path=str(tmp_path / "db"), embedding_function=TinyEmbedding())
    memory.add("abc")
    memory.add("xyz")
    assert memory.search("cab", max_distance=0.1) == ["abc"]
    assert memory.search("qqq", max_distance=0.1) == []
    assert sorted(memory.search("cab", max_distance=2.0)) == ["abc", "xyz"]

def test_search_empty_collection(tmp_path):
    """
    Searching an empty memory returns an empty list instead of raising an error.
    """
    memory = Memory(path=str(tmp_path / "db"), embedding_function=TinyEmbedding())
    assert memory.search("anything", max_distance=1.0) == []


# Agent loop

def test_multiple_tool_calls_in_one_reply(tmp_path):
    """
    When one LLM reply requests several tools, each result is sent back in order with its matching tool_call_id.
    """
    tools, _ = make_tools(tmp_path)
    llm = ScriptedLLM([
        SimpleNamespace(content=None, tool_calls=[
            tool_call("a", "calculator", '{"expression": "1+1"}'),
            tool_call("b", "calculator", '{"expression": "2*3"}'),
        ]),
        reply("2 and 6"),
    ])
    Agent(llm, tools).run("two sums")
    results = [m for m in llm.calls[1] if m["role"] == "tool"]
    assert results == [
        {"role": "tool", "tool_call_id": "a", "content": "2"},
        {"role": "tool", "tool_call_id": "b", "content": "6"},
    ]

def test_step_limit_leaves_history_valid(tmp_path):
    """
    After hitting the step limit, the history ends with the fallback answer and the next turn still works.
    """
    tools, _ = make_tools(tmp_path)
    looping = SimpleNamespace(content=None, tool_calls=[tool_call("c", "get_current_datetime", "{}")])
    llm = ScriptedLLM([looping, looping, reply("fine")])
    agent = Agent(llm, tools, max_steps=2)
    agent.run("loop")
    assert agent.messages[-1]["role"] == "assistant"
    assert "step limit" in agent.messages[-1]["content"]
    # The next turn still works.
    assert agent.run("again") == "fine"

def test_reset_clears_history(tmp_path):
    """
    reset() removes the chat history and leaves only the system prompt.
    """
    tools, _ = make_tools(tmp_path)
    agent = Agent(ScriptedLLM([reply("hi")]), tools)
    agent.run("hello")
    agent.reset()
    assert [m["role"] for m in agent.messages] == ["system"]

def test_none_content_returns_empty_string(tmp_path):
    """
    A reply with no content and no tool calls returns an empty string instead of None.
    """
    tools, _ = make_tools(tmp_path)
    assert Agent(ScriptedLLM([reply(None)]), tools).run("hi") == ""

def test_failure_after_tool_keeps_saved_note(tmp_path):
    """
    If the LLM fails after save_note ran, the turn is rolled back but the note stays saved in memory.
    """
    # Rollback undoes the chat history, not side effects: the note was
    # already written to Chroma before the LLM failed.
    tools, memory = make_tools(tmp_path)

    class FailsSecondCall(ScriptedLLM):
        def chat(self, messages, tools=None):
            if self.calls:
                raise RuntimeError("API down")
            return super().chat(messages, tools)

    llm = FailsSecondCall([
        SimpleNamespace(content=None, tool_calls=[tool_call("c1", "save_note", '{"text": "dog max"}')]),
    ])
    agent = Agent(llm, tools)
    before = list(agent.messages)
    assert "RuntimeError" in agent.run("remember my dog")
    assert agent.messages == before
    assert memory.count() == 1
