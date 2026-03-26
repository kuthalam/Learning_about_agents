"""
Step 1: Single LLM prompt for group activity recommendations.

The simplest possible AI integration — pack all friend profiles into one
prompt and ask the model for recommendations in a single round-trip.
"""

import json
import sqlite3
import requests
import textwrap

from seed_db import seed_db

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:latest"
DB_PATH = "group_activity.db"


def load_friends(db_path: str = DB_PATH) -> list[dict]:
    """Load all friend profiles from the local SQLite database."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM friends").fetchall()
    con.close()
    return [
        {
            "name": row["name"],
            "interests": json.loads(row["interests"]),
            "activities_done_solo": json.loads(row["activities_done_solo"]),
            "activities_done_with_group": json.loads(row["activities_done_with_group"]),
        }
        for row in rows
    ]


def build_prompt(friends: list[dict]) -> str:
    """
    Pack all profiles into a single recommendation prompt.
    
    :param friends: A list of profiles about each person in the group.

    :returns A prompt for our LLM that uses these profiles to prompt
             for an activity recommendation.
    """
    profiles_text = ""
    for f in friends:
        profiles_text += f"\n**{f['name']}**\n"
        profiles_text += f"  Interests: {', '.join(f['interests'])}\n"
        profiles_text += f"  Done solo: {', '.join(f['activities_done_solo'])}\n"
        profiles_text += f"  Done with the group: {', '.join(f['activities_done_with_group'])}\n"

    prompt = f"""You are helping a friend group decide what to do together.

                Here are their profiles:
                {profiles_text}
                Based on their shared interests and what they've already done (avoid repeating those),
                suggest 3 specific group activities they haven't tried yet. For each activity:
                - Give it a name
                - Explain in one sentence why it fits this particular group
                - Note which shared interest it taps into

                Keep the tone friendly and concrete."""

    return textwrap.dedent(prompt)


def ask_ollama(prompt: str, model: str = MODEL) -> str:
    """
    Function for asking Ollama for a recommendation.

    :param prompt: The recommendation-eliciting prompt
    :param model: The name of the Ollama model used
                  to elicit the recommendation

    :return The raw LLM response (specifically just the output
            text).
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["response"]


def main():
    seed_db()

    friends = load_friends()
    print("Building prompt from friend profiles...")
    prompt = build_prompt(friends)

    print(f"\n--- Prompt sent to {MODEL} ---")
    print(prompt)
    print("--- End of prompt ---\n")

    print("Asking Ollama for recommendations...\n")
    recommendation = ask_ollama(prompt)

    print("--- Recommendations ---")
    print(recommendation)


if __name__ == "__main__":
    main()
