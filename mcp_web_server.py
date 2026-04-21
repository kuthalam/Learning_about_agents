"""
MCP server that exposes web search over stdio.

Tools provided:
  - web_search : searches the web via DuckDuckGo and returns the top results
"""

from ddgs import DDGS
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("web-search")


@mcp.tool()
def web_search(query: str, max_results: int = 3) -> list[dict]:
    """
    Search the web and return the top results.

    Args:
        query:       The search query string.
        max_results: How many results to return (default 3).
    """
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))

    return [
        {"title": r["title"], "url": r["href"], "snippet": r["body"]}
        for r in results
    ]


if __name__ == "__main__":
    mcp.run(transport="stdio")
