"""
Initializes the local SQLite database and seeds it with friend profiles.

Run this once before step1_single_prompt.py (and future steps) to set up the DB.
"""

import json
import sqlite3

DB_PATH = "group_activity.db"

PROFILES = [
    {
        "name": "Alice",
        "interests": ["hiking", "photography", "coffee", "indie music"],
        "activities_done_solo": ["completed a 10k run", "attended a photography workshop"],
        "activities_done_with_group": ["brewery tour", "escape room"],
    },
    {
        "name": "Bob",
        "interests": ["cooking", "board games", "cycling", "coffee"],
        "activities_done_solo": ["took a knife-skills cooking class", "weekend bike trip"],
        "activities_done_with_group": ["brewery tour", "trivia night"],
    },
    {
        "name": "Carol",
        "interests": ["yoga", "travel", "food", "coffee", "photography"],
        "activities_done_solo": ["solo trip to Portugal", "joined a yoga retreat"],
        "activities_done_with_group": ["escape room", "trivia night"],
    },
]


def seed_db(db_path: str = DB_PATH):
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS friends (
            id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT    NOT NULL UNIQUE,
            interests               TEXT NOT NULL,  -- JSON array
            activities_done_solo    TEXT NOT NULL,  -- JSON array
            activities_done_with_group TEXT NOT NULL  -- JSON array
        )
    """)

    for profile in PROFILES:
        cur.execute("""
            INSERT OR IGNORE INTO friends
                (name, interests, activities_done_solo, activities_done_with_group)
            VALUES (?, ?, ?, ?)
        """, (
            profile["name"],
            json.dumps(profile["interests"]),
            json.dumps(profile["activities_done_solo"]),
            json.dumps(profile["activities_done_with_group"]),
        ))

    con.commit()
    con.close()
    print(f"Database ready at '{db_path}' with {len(PROFILES)} profiles.")


if __name__ == "__main__":
    seed_db()
