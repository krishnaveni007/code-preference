from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def pseudonym(user_id: str, salt: str) -> str:
    digest = hashlib.sha256(f"{salt}\0{user_id}".encode()).hexdigest()[:12]
    return f"user_{digest}"


def nested_find(value: Any, names: set[str]) -> Any:
    """Return a unique scalar matching one of names, or None if absent/ambiguous."""
    found: list[Any] = []
    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key.lower() in names and not isinstance(child, (dict, list)) and child not in (None, ""):
                    found.append(child)
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)
    visit(value)
    unique = list(dict.fromkeys(str(item) for item in found))
    return unique[0] if len(unique) == 1 else None
