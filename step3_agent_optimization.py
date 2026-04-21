"""
Step 3: Persistent conversation memory + multi-turn agent loop.

Builds on step2 by adding two things that make this a proper agent:

  1. Multi-turn conversation — the agent keeps MCP sessions alive across
     turns and loops, asking the user for follow-up questions instead of
     exiting after one answer.

  2. Persistent memory — the conversation history (all non-system messages)
     is saved to conversation_history.json after every turn. On the next run
     the agent loads that file, so it remembers what was suggested, what the
     group has tried, and what the user's follow-up questions were.

Compare to step2_single_prompt.py, which exits after a single request and
has no knowledge of past sessions.
"""

import asyncio
import json
import sys
from datetime import date
from pathlib import Path

import requests
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:14b"
HISTORY_FILE = Path("conversation_history.json")

SYSTEM_PROMPT_TEMPLATE = (
    "Today's date is {today}. "
    "You are a persistent assistant helping a friend group decide what to do together. "
    "You have access to the full conversation history, so you remember every activity "
    "you have already suggested and every preference the group has expressed. "
    "Use the available tools to fetch only the data you need, then answer the user's request. "
    "For each new activity you suggest: give it a name, explain why it fits the group, "
    "and note which shared interest it taps into. "
    "Never repeat an activity that appears earlier in this conversation. "
    "After deciding on activities, use web_search to find a relevant link for each one. "
    "Include those links in your final answer. "
    "Keep the tone friendly and concrete. "
    "If the user asks to schedule or save an activity, use create_event to add it to their Google Calendar. "
    "Always schedule events in the future. When the user gives a partial date like 'the 21st', "
    "infer the nearest upcoming occurrence of that date relative to today. "
    "If you schedule something on your own, ensure the event description includes a link to the event website."
)


def load_history() -> list[dict]:
    if HISTORY_FILE.exists():
        history = json.loads(HISTORY_FILE.read_text())
        print(f"(Loaded {len(history)} messages from previous session)\n")
        return history
    return []


def save_history(messages: list[dict]) -> None:
    history = [m for m in messages if m["role"] != "system"]
    HISTORY_FILE.write_text(json.dumps(history, indent=2))


def mcp_tool_to_ollama(tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema,
        },
    }


async def agent_turn(
    messages: list[dict],
    ollama_tools: list[dict],
    tool_registry: dict[str, ClientSession],
) -> str:
    """Run the tool-use loop for one user turn, return the assistant's final reply."""
    while True:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": messages,
                "tools": ollama_tools,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        msg = response.json()["message"]
        messages.append(msg)

        if not msg.get("tool_calls"):
            return msg["content"]

        for tc in msg["tool_calls"]:
            fn = tc["function"]
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args)

            print(f"  → {fn['name']}({args})")
            session = tool_registry.get(fn["name"])
            try:
                if session is None:
                    raise ValueError(f"unknown tool '{fn['name']}'")
                result = await session.call_tool(fn["name"], args)
                tool_output = result.content[0].text
            except Exception as e:
                tool_output = f"Error: tool '{fn['name']}' failed — {e}"
                print(f"     (tool error: {e})")

            messages.append({
                "role": "tool",
                "name": fn["name"],
                "content": tool_output,
            })


async def chat_loop() -> None:
    """
    Multi-turn conversation loop.

    Connects to MCP servers once, then keeps them alive across all turns.
    History is loaded from disk at startup and saved after every reply.
    """
    db_params = StdioServerParameters(command=sys.executable, args=["mcp_db_server.py"])
    web_params = StdioServerParameters(command=sys.executable, args=["mcp_web_server.py"])
    cal_params = StdioServerParameters(command=sys.executable, args=["mcp_calendar_server.py"])

    async with stdio_client(db_params) as (db_read, db_write), \
               stdio_client(web_params) as (web_read, web_write), \
               stdio_client(cal_params) as (cal_read, cal_write):
        async with ClientSession(db_read, db_write) as db_session, \
                   ClientSession(web_read, web_write) as web_session, \
                   ClientSession(cal_read, cal_write) as cal_session:
            await db_session.initialize()
            await web_session.initialize()
            await cal_session.initialize()

            db_tools = (await db_session.list_tools()).tools
            web_tools = (await web_session.list_tools()).tools
            cal_tools = (await cal_session.list_tools()).tools

            tool_registry: dict[str, ClientSession] = {
                **{t.name: db_session for t in db_tools},
                **{t.name: web_session for t in web_tools},
                **{t.name: cal_session for t in cal_tools},
            }
            ollama_tools = [mcp_tool_to_ollama(t) for t in db_tools + web_tools + cal_tools]
            print(f"Discovered tools: {[t['function']['name'] for t in ollama_tools]}\n")

            prior_history = load_history()
            messages: list[dict] = [
                {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(today=date.today().isoformat())},
                *prior_history,
            ]

            print("Agent ready. Type your question, or 'quit' / 'exit' to exit.\n")

            while True:
                try:
                    user_input = input("You (type 'quit'/'exit' to leave) > ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break

                if not user_input or user_input.lower() in {"quit", "exit", "q"}:
                    print("Goodbye!")
                    break

                messages.append({"role": "user", "content": user_input})

                print()
                answer = await agent_turn(messages, ollama_tools, tool_registry)
                save_history(messages)

                print(f"\nAgent > {answer}\n")


def main():
    asyncio.run(chat_loop())


if __name__ == "__main__":
    main()
