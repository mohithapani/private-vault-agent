"""
Lightweight JSON-backed registry of known people.

Stores, per person: a display name, when they were created, and the list of
files that have been indexed under their name. Deliberately simple (flat
JSON file + a lock) so it can be swapped for a real SQLite/DB-backed store
later without changing the calling code in main.py / app.py -- every
function here is the seam.
"""

import json
import os
from datetime import datetime, timezone
from threading import Lock

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "person_registry.json")

_lock = Lock()


def _empty() -> dict:
    return {"people": {}}


def _load() -> dict:
    if not os.path.exists(REGISTRY_PATH):
        return _empty()
    try:
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "people" not in data:
                data["people"] = {}
            return data
    except (json.JSONDecodeError, OSError):
        return _empty()


def _save(data: dict) -> None:
    tmp_path = REGISTRY_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, REGISTRY_PATH)


def normalize_person(name: str) -> str:
    """Canonical id form: lowercase, spaces -> underscores."""
    return name.strip().lower().replace(" ", "_")


def list_people() -> list[str]:
    """Returns sorted canonical ids of all known people."""
    return sorted(_load()["people"].keys())


def get_people_details() -> dict:
    """Returns the full {person_id: {display_name, created_at, files}} map."""
    return _load()["people"]


def add_person(name: str) -> str:
    """
    Registers a person if they don't already exist (idempotent).
    Returns the canonical person id, or "" if given an empty/blank name.
    """
    canonical = normalize_person(name)
    if not canonical:
        return ""
    with _lock:
        data = _load()
        if canonical not in data["people"]:
            data["people"][canonical] = {
                "display_name": name.strip(),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "files": [],
            }
            _save(data)
    return canonical


def record_file(person_id: str, file_path: str) -> None:
    """Appends a file record to a person's history, creating the person if needed."""
    if not person_id:
        return
    with _lock:
        data = _load()
        if person_id not in data["people"]:
            data["people"][person_id] = {
                "display_name": person_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "files": [],
            }
        data["people"][person_id]["files"].append(
            {
                "file_path": file_path,
                "indexed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        _save(data)