# src/agentic_backend/services/persistence.py
import os, json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from zoneinfo import ZoneInfo

from ..config import settings
from ..models.state_models import SupervisorState

# Ensure the directory exists
RUN_DIR = Path(settings.RUN_SAVE_DIR)
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Memory DB file path
MEMORY_DB_PATH = RUN_DIR / "memory.json"


def _load_memory_db() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load the entire memory database from JSON file."""
    # Create file if it doesn't exist
    if not MEMORY_DB_PATH.exists():
        MEMORY_DB_PATH.write_text("{}")
        return {}

    try:
        with open(MEMORY_DB_PATH, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        # If file is corrupted, reset it
        MEMORY_DB_PATH.write_text("{}")
        return {}


def _save_memory_db(db: Dict[str, Dict[str, Dict[str, Any]]]):
    """Save the entire memory database to JSON file."""
    # Ensure directory exists before writing
    MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(MEMORY_DB_PATH, "w") as f:
        json.dump(db, f, indent=2)


def get_thread_memory(user_id: str, thread_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve memory for a specific thread.

    Returns:
        {
            "request_summary": str,
            "response_summary": str,
            "raw_conversation": List[Dict]
        }
    """
    db = _load_memory_db()
    return db.get(user_id, {}).get(thread_id)


def update_thread_memory(
    user_id: str,
    thread_id: str,
    request_summary: Optional[str] = None,
    response_summary: Optional[str] = None,
    conversation_entry: Optional[Dict[str, Any]] = None
):
    """
    Update memory for a specific thread.

    Args:
        user_id: User identifier
        thread_id: Thread/conversation identifier
        request_summary: Summary of user requests (optional, will append if provided)
        response_summary: Summary of agent responses (optional, will append if provided)
        conversation_entry: Single conversation turn to append to raw_conversation
    """
    db = _load_memory_db()
    print(db)
    # Initialize user if not exists
    print(user_id not in db)
    if user_id not in db:
        db[user_id] = {}
    
    # Initialize thread if not exists
    if thread_id not in db[user_id]:
        db[user_id][thread_id] = {
            "request_summary": "",
            "response_summary": "",
            "raw_conversation": [],
            "last_updated": datetime.now(ZoneInfo("Europe/London")).isoformat()
        }

    thread_data = db[user_id][thread_id]

    # Update summaries (append if provided)
    if request_summary:
        current = thread_data["request_summary"]
        thread_data["request_summary"] = f"{current}\n{request_summary}".strip() if current else request_summary

    if response_summary:
        current = thread_data["response_summary"]
        thread_data["response_summary"] = f"{current}\n{response_summary}".strip() if current else response_summary

    # Append conversation entry
    if conversation_entry:
        thread_data["raw_conversation"].append({
            **conversation_entry,
            "timestamp": datetime.now(ZoneInfo("Europe/London")).isoformat()
        })

    # Update last_updated timestamp
    thread_data["last_updated"] = datetime.now(ZoneInfo("Europe/London")).isoformat()

    _save_memory_db(db)


def clear_thread_memory(user_id: str, thread_id: str):
    """Clear memory for a specific thread."""
    db = _load_memory_db()
    if user_id in db and thread_id in db[user_id]:
        del db[user_id][thread_id]
        _save_memory_db(db)


def get_user_threads(user_id: str) -> List[str]:
    """Get all thread IDs for a user."""
    db = _load_memory_db()
    return list(db.get(user_id, {}).keys())


# Legacy functions for backward compatibility
def save_state(state: SupervisorState):
    """Save state to individual run file (legacy)."""
    path = RUN_DIR / f"{state.run_id}.json"
    with open(path, "w") as f:
        json.dump(state.model_dump(), f, indent=2, default=str)


def load_state(run_id: str) -> SupervisorState:
    """Load state from individual run file (legacy)."""
    path = RUN_DIR / f"{run_id}.json"
    if not path.exists():
        raise FileNotFoundError
    with open(path) as f:
        data = json.load(f)
    return SupervisorState(**data)
