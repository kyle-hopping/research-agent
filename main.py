"""
main.py — Command-line chat with the Tiny Agent.

How to Run:
    python main.py            # normal chat
    python main.py --verbose  # also shows every tool the agent uses
    python main.py -v         # same as above

In-chat commands:
    /reset  -> forget this conversation (long-term notes are kept)
    /exit   -> quit
"""

import argparse
import sys
from agent.core import Agent
from agent.memory import Memory
from agent.tools import build_tools
from agent.config import settings
from agent.llm import GroqLLM

def print_event(kind: str, data: dict) -> None:
    """
    Log what the agent is doing behind the scenes (--verbose).
    """
    if kind == "tool_call":
        print(f"TOOL_CALL: [step {data['step']}] {data['name']}({data['arguments']})")
    elif kind == "tool_result":
        preview = data["result"].replace("\n", " ")
        print(f"TOOL_RESULT: {preview[:150]}{'…' if len(preview) > 150 else ''}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with Tiny Agent.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="show tool calls and results")
    args = parser.parse_args()

    # Try to build the setup objects (LLM, memory, tools, agent) and exit with
    # an error if any of them fail.
    try:
        llm = GroqLLM(api_key=settings.groq_api_key, model=settings.model)
    except ValueError as e:
        sys.exit(f"Setup error: {e}")

    memory = Memory(path=settings.chroma_path)
    tools = build_tools(memory)
    agent = Agent(llm, tools, max_steps=settings.max_steps,
                  on_event=print_event if args.verbose else None)

    print(f"Tiny Agent ({settings.model}) — {memory.count()} notes in memory.")
    print("Type /reset to clear the chat, /exit to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):  # Ctrl+D / Ctrl+C
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input == "/exit":
            print("Bye!")
            break
        if user_input == "/reset":
            agent.reset()
            print("(conversation cleared)\n")
            continue

        # Handle errors in the agent's run()
        answer = agent.run(user_input)
        print(f"Agent: {answer}\n")

if __name__ == "__main__":
    main()