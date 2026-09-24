#!/bin/sh
set -eu
root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
dest="$root/third_party/SWE-Together"
if [ -d "$dest/.git" ]; then
  echo "SWE-Together already exists at $dest"
  exit 0
fi
git clone https://github.com/Togetherbench/SWE-Together.git "$dest"
git -C "$dest" checkout "$(cat "$root/config/swe_together_commit.txt")"
python3 "$root/scripts/patch_swe_together.py" "$dest"
