"""
MCP server that exposes the group_activity SQLite database over stdio.

Tools provided:
  - list_friends       : returns all friend names
  - get_friend_profile : returns full profile for a given friend name
  - get_all_profiles   : returns every profile in the database
"""

import json
import sqlite3
from pathlib import Path

from mcp.server.fastmcp import FastMCP

DB_PATH = Path(__file__).parent / "group_activity.db"

mcp = FastMCP("group-activity-db")


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


@mcp.tool()
def list_friends() -> list[str]:
    """Return the names of all friends stored in the database."""
    with _connect() as con:
        rows = con.execute("SELECT name FROM friends ORDER BY name").fetchall()
    return [row["name"] for row in rows]


@mcp.tool()
def get_friend_profile(name: str) -> dict:
    """
    Return the full profile for a single friend.

    Args:
        name: The friend's name (case-sensitive).
    """
    with _connect() as con:
        row = con.execute(
            "SELECT * FROM friends WHERE name = ?", (name,)
        ).fetchone()

    if row is None:
        return {"error": f"No friend named '{name}' found."}

    return {
        "name": row["name"],
        "interests": json.loads(row["interests"]),
        "activities_done_solo": json.loads(row["activities_done_solo"]),
        "activities_done_with_group": json.loads(row["activities_done_with_group"]),
    }


@mcp.tool()
def get_all_profiles() -> list[dict]:
    """Return every friend's full profile from the database."""
    with _connect() as con:
        rows = con.execute("SELECT * FROM friends ORDER BY name").fetchall()

    return [
        {
            "name": row["name"],
            "interests": json.loads(row["interests"]),
            "activities_done_solo": json.loads(row["activities_done_solo"]),
            "activities_done_with_group": json.loads(row["activities_done_with_group"]),
        }
        for row in rows
    ]


if __name__ == "__main__":
    mcp.run(transport="stdio")
