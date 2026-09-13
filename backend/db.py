import sqlite3
import json
import time
import os
from contextlib import contextmanager

DB_PATH = os.environ.get("LOG_DB_PATH", "interaction_logs.db")


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                category TEXT NOT NULL,
                mode TEXT NOT NULL,
                input_sequence TEXT NOT NULL,
                predicted_items TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                category TEXT NOT NULL,
                input_sequence TEXT NOT NULL,
                chosen_asin TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS item_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                category TEXT NOT NULL,
                url TEXT NOT NULL,
                extracted_asin TEXT,
                note TEXT
            )
        """)
        conn.commit()


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def log_prediction(category: str, mode: str, input_sequence: list[str], predicted_items: list[str]):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO predictions (timestamp, category, mode, input_sequence, predicted_items) VALUES (?, ?, ?, ?, ?)",
            (time.time(), category, mode, json.dumps(input_sequence), json.dumps(predicted_items)),
        )
        conn.commit()


def log_feedback(category: str, input_sequence: list[str], chosen_asin: str):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO feedback (timestamp, category, input_sequence, chosen_asin) VALUES (?, ?, ?, ?)",
            (time.time(), category, json.dumps(input_sequence), chosen_asin),
        )
        conn.commit()


def get_feedback_for_retraining(category: str):
    """Returns list of (input_sequence: list[str], chosen_asin: str) tuples."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT input_sequence, chosen_asin FROM feedback WHERE category = ?",
            (category,),
        ).fetchall()
    return [(json.loads(row[0]), row[1]) for row in rows]


def log_item_request(category: str, url: str, extracted_asin: str | None, note: str | None):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO item_requests (timestamp, category, url, extracted_asin, note) VALUES (?, ?, ?, ?, ?)",
            (time.time(), category, url, extracted_asin, note),
        )
        conn.commit()


def get_pending_item_requests(category: str | None = None):
    with get_conn() as conn:
        if category:
            rows = conn.execute(
                "SELECT id, timestamp, category, url, extracted_asin, note FROM item_requests WHERE category = ? ORDER BY timestamp DESC",
                (category,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, timestamp, category, url, extracted_asin, note FROM item_requests ORDER BY timestamp DESC"
            ).fetchall()
    return rows
