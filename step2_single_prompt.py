"""
Step 2: MCP-powered tool use with multiple servers.

Instead of hard-coding which data to fetch, we:
  1. Discover the available MCP tools at runtime via list_tools()
  2. Pass their schemas to the LLM alongside the user's request
  3. Let the LLM decide which tool(s) to call
  4. Execute those calls via the correct MCP server and feed results back

Two MCP servers are used:
  - mcp_db_server.py     : friend profiles from SQLite
  - mcp_web_server.py : web search via DuckDuckGo

This demonstrates MCP composability: the client merges tools from
multiple servers into one list and routes each call to the right server.

Compare to step1_single_prompt.py, which hard-codes get_all_profiles()
regardless of what the user asks.
"""

import asyncio
import json
import sys

import requests
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:14b"

SYSTEM_PROMPT = (
    "You are helping a friend group decide what to do together. "
    "Use the available tools to fetch only the data you need, then answer the user's request. "
    "For each activity you suggest: give it a name, explain why it fits the group, "
    "and note which shared interest it taps into. "
    "Avoid repeating activities they have already done. "
    "After deciding on activities, use web_search to find a relevant link for each one. "
    "Include those links in your final answer. "
    "Keep the tone friendly and concrete."
)


def mcp_tool_to_ollama(tool) -> dict:
    """Convert an MCP tool definition to Ollama's function-calling schema."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.inputSchema,
        },
    }


async def run(user_request: str) -> str:
    """
    Run the main loop for handling user requests and tool calls.

    Connects to two MCP servers (DB and web search), merges their tools
    into one list for the LLM, and routes each tool call to the correct server.

    :param user_request: The user's natural-language request for group activities.
    :return: The final answer from the LLM after any necessary tool calls.
    """
    db_params = StdioServerParameters(command=sys.executable, args=["mcp_db_server.py"])
    web_params = StdioServerParameters(command=sys.executable, args=["mcp_web_server.py"])

    async with stdio_client(db_params) as (db_read, db_write), \
               stdio_client(web_params) as (web_read, web_write):
        async with ClientSession(db_read, db_write) as db_session, \
                   ClientSession(web_read, web_write) as web_session:
            await db_session.initialize()
            await web_session.initialize()

            ## Discover tools from both servers and build a name → session registry
            ## so we know which server to call when the LLM picks a tool
            db_tools = (await db_session.list_tools()).tools
            web_tools = (await web_session.list_tools()).tools

            tool_registry: dict[str, ClientSession] = {
                **{t.name: db_session for t in db_tools},
                **{t.name: web_session for t in web_tools},
            }

            ollama_tools = [mcp_tool_to_ollama(t) for t in db_tools + web_tools]
            print(f"Discovered tools: {[t['function']['name'] for t in ollama_tools]}\n")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_request},
            ]

            ## Tool-use loop: keep going until the LLM stops calling tools
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

                ## LLM has enough data — return the final answer
                if not msg.get("tool_calls"):
                    return msg["content"]

                ## Execute each requested tool call via the correct MCP session
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


def main():
    user_request = input("What would you like to ask about group activities? > ").strip()
    if not user_request:
        user_request = "Suggest 3 group activities we haven't tried yet."

    print()
    answer = asyncio.run(run(user_request))

    print("\n--- Answer ---")
    print(answer)


if __name__ == "__main__":
    main()
