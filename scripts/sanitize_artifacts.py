#!/usr/bin/env python3
"""Redact common API credential formats from local experiment artifacts."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


REPLACEMENTS = (
    (re.compile(rb"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"), b"[REDACTED_OPENAI_KEY]"),
    (re.compile(rb"AIza[0-9A-Za-z_-]{30,}"), b"[REDACTED_GOOGLE_KEY]"),
    (re.compile(rb"AKIA[0-9A-Z]{16}"), b"[REDACTED_AWS_KEY]"),
    (re.compile(rb"gh(?:p|o|u|s|r)_[A-Za-z0-9]{20,}"), b"[REDACTED_GITHUB_TOKEN]"),
    (re.compile(rb"github_pat_[A-Za-z0-9_]{40,}"), b"[REDACTED_GITHUB_TOKEN]"),
    (re.compile(rb"xox(?:b|a|p|r|s)-[A-Za-z0-9-]{10,}"), b"[REDACTED_SLACK_TOKEN]"),
)


def sanitize_file(path: Path) -> int:
    if path.is_symlink() or not path.is_file():
        return 0
    try:
        original = path.read_bytes()
    except OSError:
        return 0
    if b"\0" in original:
        return 0
    sanitized = original
    replacements = 0
    for pattern, marker in REPLACEMENTS:
        sanitized, count = pattern.subn(marker, sanitized)
        replacements += count
    if replacements:
        path.write_bytes(sanitized)
    return replacements


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    changed_files = 0
    replacements = 0
    for root in args.paths:
        paths = root.rglob("*") if root.is_dir() else (root,)
        for path in paths:
            count = sanitize_file(path)
            if count:
                changed_files += 1
                replacements += count
    print(f"redacted {replacements} credentials in {changed_files} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
