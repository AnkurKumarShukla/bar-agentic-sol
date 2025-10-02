# src/agentic_backend/services/persistence.py
import os, json
from pathlib import Path
from ..config import settings
from ..models.state_models import SupervisorState

RUN_DIR = Path(settings.RUN_SAVE_DIR)
RUN_DIR.mkdir(parents=True, exist_ok=True)

def save_state(state: SupervisorState):
    path = RUN_DIR / f"{state.run_id}.json"
    with open(path, "w") as f:
        json.dump(state.dict(), f, indent=2)

def load_state(run_id: str) -> SupervisorState:
    path = RUN_DIR / f"{run_id}.json"
    if not path.exists():
        raise FileNotFoundError
    with open(path) as f:
        data = json.load(f)
    return SupervisorState(**data)
