"""Shared paths, API client and JSON parsing for the scope_manner line.

Paths are relative to the repository root. API access goes through LiteLLM; set LITELLM_API_KEY (and
optionally LITELLM_API_BASE, default: CMU AI gateway) in <repo>/.env or the environment.
"""
import json, os, re, time
from pathlib import Path

import litellm
from dotenv import load_dotenv

litellm.suppress_debug_info = True
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
API_BASE = os.getenv("LITELLM_API_BASE", "https://ai-gateway.andrew.cmu.edu/")
API_KEY = os.environ.get("LITELLM_API_KEY", "")
OUT = ROOT / "outputs/discovery/scope_manner"
DATA = OUT / "data"
FIG = OUT / "figures"
# Inputs produced by earlier lines of the repository.
POSITIVES = ROOT / "outputs/discovery/scope/gate/positives_dedup.jsonl"   # 19,944 gated messages; id = line index
CONVERSATIONS = ROOT / "data/swechat_v2/conversations.parquet"            # for prev_agent_reply
DEFAULT_CHAT_MODEL = os.getenv("SCOPE_CHAT_MODEL", "openai/wine-gemini-3-flash-preview")
DEFAULT_EMBED_MODEL = os.getenv("SCOPE_EMBED_MODEL", "openai/wine-gemini-embedding-001")


def load_messages():
    return [json.loads(l) for l in open(POSITIVES)]


def load_prev_replies():
    import pandas as pd
    conv = pd.read_parquet(CONVERSATIONS, columns=["session_id", "turn_number", "prev_agent_reply"])
    return dict(zip(zip(conv.session_id, conv.turn_number), conv.prev_agent_reply))


def load_records(path):
    """Latest successful record per id from an append-only jsonl (error lines are skipped)."""
    out = {}
    for l in open(path):
        o = json.loads(l)
        if "error" not in o:
            out[o["id"] if "id" in o else o["cluster"]] = o
    return out


def parse_json(text):
    """Parse model output that may be fenced or contain stray backslashes."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        t = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", t)
        t = re.sub(r"\\u(?![0-9a-fA-F]{4})", r"\\\\u", t)
        return json.loads(t)


def chat(messages, model, response_format, temperature=0, timeout=120, max_tokens=8000):
    r = litellm.completion(api_key=API_KEY, base_url=API_BASE, model=model, messages=messages,
                           response_format=response_format, temperature=temperature, timeout=timeout,
                           max_tokens=max_tokens)
    return r.choices[0].message.content


def embed(texts, model=DEFAULT_EMBED_MODEL, batch=64):
    import numpy as np
    out = []
    for i in range(0, len(texts), batch):
        r = litellm.embedding(api_key=API_KEY, api_base=API_BASE, model=model, input=texts[i:i + batch])
        out += [d["embedding"] for d in r.data]
    E = np.array(out, dtype=np.float32)
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def with_retries(fn, tries=4, first_sleep=3):
    last = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(60, first_sleep * 2 ** attempt))
    raise RuntimeError(str(last)[:300])
