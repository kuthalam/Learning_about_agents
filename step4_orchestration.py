"""
Step 4: Multi-agent orchestration.

Instead of one agent with access to all tools, we have four specialized agents:

  Orchestrator   — receives the user's request, decides which sub-agents to
                   call and in what order, then synthesizes a final response.
                   It has no MCP tools; its "tools" are the sub-agents below.

  DB agent       — has access only to the DB MCP tools. Fetches friend
                   profiles when the orchestrator needs group data.

  Recommendations agent — has access only to web_search. Takes profile data
                   and generates activity suggestions with links.

  Calendar agent — has access only to create_event. Schedules events on
                   Google Calendar when asked.

The key insight: the orchestrator's tool-use loop is structurally identical
to the sub-agents' tool-use loops. The only difference is what gets called
when a "tool" fires — a sub-agent function instead of an MCP session.
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

# ---------------------------------------------------------------------------
# System prompts — each agent gets a focused, narrow directive
# ---------------------------------------------------------------------------

ORCHESTRATOR_PROMPT = (
    "Today's date is {today}. "
    "You are an orchestrator that coordinates three specialist agents to help a "
    "friend group decide what to do together. "
    "You have three tools: db_agent (fetches friend profiles), "
    "recommendations_agent (suggests activities and finds links), and "
    "calendar_agent (schedules events). "
    "Break the user's request into sub-tasks and call the right agents in order. "
    "Pass relevant context between agents — for example, pass profile data from "
    "db_agent directly into the task you give recommendations_agent. "
    "Synthesize the agents' results into a single, friendly final response. "
    "When the user gives a partial date like 'the 21st', resolve it to the "
    "nearest upcoming occurrence relative to today before passing it to calendar_agent."
)

DB_AGENT_PROMPT = (
    "You are a database agent. Your only job is to fetch friend profile data "
    "from the database using the available tools. Return the data completely "
    "and accurately. Do not make recommendations or add commentary."
)

RECOMMENDATIONS_AGENT_PROMPT = (
    "You are a recommendations agent. You receive friend profile data and a "
    "request, then suggest group activities. For each suggestion: give it a "
    "name, explain why it fits the group, note which shared interest it taps "
    "into, and use web_search to find a relevant link. "
    "Never suggest activities the group has already done together."
)

CALENDAR_AGENT_PROMPT = (
    "Today's date is {today}. "
    "You are a calendar scheduling agent. Your only job is to schedule events "
    "on Google Calendar using create_event. Parse the event details from the "
    "task you are given and create the event. Always schedule in the future. "
    "Return a short confirmation that includes the event link."
)

# ---------------------------------------------------------------------------
# Orchestrator's sub-agent tool definitions (not MCP — defined manually)
# ---------------------------------------------------------------------------

ORCHESTRATOR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "db_agent",
            "description": (
                "Fetches friend profiles from the database. Returns names, "
                "interests, and past solo and group activities."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "What to retrieve from the database."}
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommendations_agent",
            "description": (
                "Suggests group activities and finds web links for them. "
                "Pass it the friend profiles so it can tailor suggestions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Task description including friend profile data and any constraints.",
                    }
                },
                "required": ["task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calendar_agent",
            "description": "Schedules an event on Google Calendar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Event details: title, date, time, and any description.",
                    }
                },
                "required": ["task"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Shared tool-use loop for sub-agents (calls MCP sessions)
# ---------------------------------------------------------------------------

def mcp_tool_to_ollama(tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema,
        },
    }


async def _mcp_tool_loop(
    messages: list[dict],
    ollama_tools: list[dict],
    mcp_registry: dict[str, ClientSession],
    label: str,
) -> str:
    """Run the tool-use loop for a sub-agent, return its final reply."""
    while True:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "messages": messages, "tools": ollama_tools, "stream": False},
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

            print(f"    [{label}] → {fn['name']}({args})")
            session = mcp_registry.get(fn["name"])
            try:
                if session is None:
                    raise ValueError(f"unknown tool '{fn['name']}'")
                result = await session.call_tool(fn["name"], args)
                tool_output = result.content[0].text
            except Exception as e:
                tool_output = f"Error: tool '{fn['name']}' failed — {e}"
                print(f"    [{label}] tool error: {e}")

            messages.append({"role": "tool", "name": fn["name"], "content": tool_output})


# ---------------------------------------------------------------------------
# Sub-agent runners
# ---------------------------------------------------------------------------

async def run_db_agent(
    task: str,
    db_ollama_tools: list[dict],
    db_mcp_registry: dict[str, ClientSession],
) -> str:
    messages = [
        {"role": "system", "content": DB_AGENT_PROMPT},
        {"role": "user", "content": task},
    ]
    return await _mcp_tool_loop(messages, db_ollama_tools, db_mcp_registry, label="db_agent")


async def run_recommendations_agent(
    task: str,
    web_ollama_tools: list[dict],
    web_mcp_registry: dict[str, ClientSession],
) -> str:
    messages = [
        {"role": "system", "content": RECOMMENDATIONS_AGENT_PROMPT},
        {"role": "user", "content": task},
    ]
    return await _mcp_tool_loop(messages, web_ollama_tools, web_mcp_registry, label="recommendations_agent")


async def run_calendar_agent(
    task: str,
    cal_ollama_tools: list[dict],
    cal_mcp_registry: dict[str, ClientSession],
) -> str:
    messages = [
        {"role": "system", "content": CALENDAR_AGENT_PROMPT.format(today=date.today().isoformat())},
        {"role": "user", "content": task},
    ]
    return await _mcp_tool_loop(messages, cal_ollama_tools, cal_mcp_registry, label="calendar_agent")


# ---------------------------------------------------------------------------
# Orchestrator loop (calls sub-agents instead of MCP tools)
# ---------------------------------------------------------------------------

async def _orchestrator_loop(
    messages: list[dict],
    sub_agent_registry: dict,
) -> str:
    """Run the orchestrator's decision loop, dispatching to sub-agents."""
    while True:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "messages": messages, "tools": ORCHESTRATOR_TOOLS, "stream": False},
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

            task = args.get("task", "")
            print(f"  [orchestrator → {fn['name']}]: {task[:100]}")

            agent_fn = sub_agent_registry.get(fn["name"])
            try:
                if agent_fn is None:
                    raise ValueError(f"unknown sub-agent '{fn['name']}'")
                result = await agent_fn(task)
            except Exception as e:
                result = f"Error from sub-agent '{fn['name']}': {e}"
                print(f"  [orchestrator] sub-agent error: {e}")

            messages.append({"role": "tool", "name": fn["name"], "content": result})


# ---------------------------------------------------------------------------
# Persistent history helpers
# ---------------------------------------------------------------------------

def load_history() -> list[dict]:
    if HISTORY_FILE.exists():
        history = json.loads(HISTORY_FILE.read_text())
        print(f"(Loaded {len(history)} messages from previous session)\n")
        return history
    return []


def save_history(messages: list[dict]) -> None:
    history = [m for m in messages if m["role"] != "system"]
    HISTORY_FILE.write_text(json.dumps(history, indent=2))


# ---------------------------------------------------------------------------
# Main chat loop
# ---------------------------------------------------------------------------

async def chat_loop() -> None:
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

            db_ollama_tools = [mcp_tool_to_ollama(t) for t in db_tools]
            web_ollama_tools = [mcp_tool_to_ollama(t) for t in web_tools]
            cal_ollama_tools = [mcp_tool_to_ollama(t) for t in cal_tools]

            db_mcp_registry = {t.name: db_session for t in db_tools}
            web_mcp_registry = {t.name: web_session for t in web_tools}
            cal_mcp_registry = {t.name: cal_session for t in cal_tools}

            sub_agent_registry = {
                "db_agent": lambda task: run_db_agent(task, db_ollama_tools, db_mcp_registry),
                "recommendations_agent": lambda task: run_recommendations_agent(task, web_ollama_tools, web_mcp_registry),
                "calendar_agent": lambda task: run_calendar_agent(task, cal_ollama_tools, cal_mcp_registry),
            }

            prior_history = load_history()
            messages: list[dict] = [
                {"role": "system", "content": ORCHESTRATOR_PROMPT.format(today=date.today().isoformat())},
                *prior_history,
            ]

            print("Orchestrator ready. Type your question, or 'quit' / Ctrl+C to exit.\n")

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
                answer = await _orchestrator_loop(messages, sub_agent_registry)
                save_history(messages)

                print(f"\nOrchestrator > {answer}\n")


def main():
    asyncio.run(chat_loop())


if __name__ == "__main__":
    main()
