"""Shared paths and credentials for pref_memory_replay. Every path can be overridden by an environment variable.

  TURN_REPLAY_DATA       run outputs (prep/, arms, judges, memory snapshots, viewer)   default <repo>/data/pref_memory_replay
  TURN_REPLAY_CASES      case JSON files (they quote participants: kept out of git)   default <DATA>/cases
  TURN_REPLAY_CLONES     full clones of the cases' repos, one per owner__name          default <DATA>/clones
  SWECHAT_SNAPSHOT       local SALT-NLP/SWE-chat snapshot (transcripts, parquet)       default <repo>/data/swechat_hf
  SWECHAT_MEMORY_STORES  recovered Claude Code memory stores (real-memory arm)         default <repo>/data/swechat_memory_stores
  PREF_EXPLORER_DATA     preference statements, explorer_data.json (C2 arm)            default <repo>/data/explorer_data.json
  TURN_REPLAY_ENV_FILE   .env holding CLAUDE_CODE_OAUTH_TOKEN (used when the variable itself is unset)
                                                                                       default <repo>/.env
  TURN_REPLAY_IMAGE      Docker image built from this folder's Dockerfile              default turn-replay:cc2.1.273
  TURN_REPLAY_JUDGE_CC   Claude Code >= 2.1.280 for judge/summary/C2 calls (Opus 5.5); the --cc-bin default
"""
from __future__ import annotations

import os
import pathlib

MODULE = pathlib.Path(__file__).resolve().parent
REPO = MODULE.parents[1]


def _path(var: str, default: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(os.environ[var]).expanduser().resolve() if os.environ.get(var) else default


DATA = _path("TURN_REPLAY_DATA", REPO / "data" / "pref_memory_replay")
CASES = _path("TURN_REPLAY_CASES", DATA / "cases")
CLONES = _path("TURN_REPLAY_CLONES", DATA / "clones")
SNAPSHOT = _path("SWECHAT_SNAPSHOT", REPO / "data" / "swechat_hf")
STORES = _path("SWECHAT_MEMORY_STORES", REPO / "data" / "swechat_memory_stores")
EXPLORER = _path("PREF_EXPLORER_DATA", REPO / "data" / "explorer_data.json")
ENV_FILE = _path("TURN_REPLAY_ENV_FILE", REPO / ".env")
IMAGE = os.environ.get("TURN_REPLAY_IMAGE", "turn-replay:cc2.1.273")
JUDGE_CC = os.environ.get("TURN_REPLAY_JUDGE_CC")
AGENT_MODEL = "claude-sonnet-5-5[1m]"  # [1m]: without it contexts over ~200K are silently auto-compacted
JUDGE_MODEL = "claude-opus-5-5"


def oauth_token() -> str:
    """CLAUDE_CODE_OAUTH_TOKEN from the environment, else from ENV_FILE."""
    if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return os.environ["CLAUDE_CODE_OAUTH_TOKEN"]
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            k, _, v = line.strip().partition("=")
            if k.strip() == "CLAUDE_CODE_OAUTH_TOKEN" and v:
                return v.strip().strip("\"'")
    raise SystemExit(f"CLAUDE_CODE_OAUTH_TOKEN not set and not found in {ENV_FILE}")


def transcript(path: str) -> pathlib.Path:
    """A case's transcript: stored relative to SWECHAT_SNAPSHOT (e.g. transcripts/<sid>.jsonl); absolute paths still work."""
    p = pathlib.Path(path)
    return p if p.is_absolute() else SNAPSHOT / p


def relative_transcript(path) -> str:
    """The form case files store: relative to the snapshot when the file is inside it."""
    p = pathlib.Path(path).resolve()
    try:
        return str(p.relative_to(SNAPSHOT))
    except ValueError:
        return str(p)
