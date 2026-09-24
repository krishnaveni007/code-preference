#!/usr/bin/env python3
"""Apply the small, pinned action-agent-only preference hook idempotently."""
from pathlib import Path
import sys

root = Path(sys.argv[1])

def replace(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if new in text:
        return
    if old not in text:
        raise SystemExit(f"expected pinned source fragment missing in {path}")
    path.write_text(text.replace(old, new, 1))

run = root / "src/run_eval.py"
replace(run, '    reasoning_effort: str | None = None,\n) -> TrialConfig:',
             '    reasoning_effort: str | None = None,\n    preference_profile: str = "",\n) -> TrialConfig:')
replace(run, '        user_sim_kwargs.update(_per_task)\n\n    agent_config = AgentConfig(',
             '        user_sim_kwargs.update(_per_task)\n\n    if agent_type in {"codex", "claude-code"}:\n        # Action-agent-only context. The user simulator never receives this.\n        user_sim_kwargs["preference_profile"] = preference_profile\n\n    agent_config = AgentConfig(')
replace(run, '    parser.add_argument("--force-build", action="store_true", help="Force E2B template rebuild (recovers from corrupted template aliases)")\n    args = parser.parse_args()',
             '    parser.add_argument("--force-build", action="store_true", help="Force E2B template rebuild (recovers from corrupted template aliases)")\n    parser.add_argument("--preference-profiles", type=Path, default=None,\n                        help="JSON object mapping task name to action-agent profile text")\n    args = parser.parse_args()\n    preference_profiles = {}\n    if args.preference_profiles:\n        preference_profiles = json.loads(args.preference_profiles.read_text())\n        if not isinstance(preference_profiles, dict):\n            raise SystemExit("--preference-profiles must contain a JSON object")')
replace(run, '        "tasks": task_names,\n    }',
             '        "tasks": task_names,\n        "preference_profiles_file": str(args.preference_profiles) if args.preference_profiles else None,\n    }')
replace(run,
        '    action_model, action_key, _env_var = resolve_model(args.model)\n'
        '    user_model, user_key, _ = resolve_model(args.user_model or args.model)\n',
        '    # Accept Claude Code subscription OAuth without requiring an API key.\n'
        '    oauth_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")\n'
        '    if args.model.startswith("anthropic/") and oauth_token:\n'
        '        action_model = args.model.split("/", 1)[1]\n'
        '        action_key = oauth_token\n'
        '        _env_var = "CLAUDE_CODE_OAUTH_TOKEN"\n'
        '    else:\n'
        '        action_model, action_key, _env_var = resolve_model(args.model)\n'
        '    user_model, user_key, _ = resolve_model(args.user_model or args.model)\n')
replace(run, '            reasoning_effort=args.reasoning_effort,\n        )',
             '            reasoning_effort=args.reasoning_effort,\n            preference_profile=str(preference_profiles.get(task_name, "")),\n        )')

for name in ("user_enabled_codex.py", "user_enabled_claude_code.py"):
    path = root / "src/user_agent/agents" / name
    if name == "user_enabled_codex.py":
        replace(path, '        reasoning_effort: str = "medium",\n        **kwargs,',
                     '        reasoning_effort: str = "medium",\n        preference_profile: str = "",\n        **kwargs,')
        anchor = '            version=codex_version, reasoning_effort=reasoning_effort, **kwargs,\n        )'
    else:
        replace(path, '        call_user_on_completion: bool = True,\n        **kwargs,',
                     '        call_user_on_completion: bool = True,\n        preference_profile: str = "",\n        **kwargs,')
        anchor = '        self._inner = ClaudeCode(logs_dir=logs_dir, model_name=model_name, **kwargs)'
    replace(path, anchor, anchor + '\n        self._preference_profile = preference_profile')
    replace(path, '        self._task_instruction = ""',
                 '        self._task_instruction = ""\n        self._sim_task_instruction = ""')
    replace(path, '            task_description=self._task_instruction,',
                 '            task_description=self._sim_task_instruction,')
    replace(path, '    ) -> None:\n        # Inject repo config files into the instruction',
                 '    ) -> None:\n        self._sim_task_instruction = instruction\n        # Intervention is visible only to the action agent, never the simulator.\n        if self._preference_profile:\n            instruction = f"{instruction}\\n\\n## Developer preferences\\n{self._preference_profile}"\n        # Inject repo config files into the instruction')

claude_wrapper = root / "src/user_agent/agents/user_enabled_claude_code.py"
replace(claude_wrapper,
        '    assert action == "diff" and turn is not None and prev_turn is not None\n',
        '''    if action == "restore":
        per_repo = (
            '  (cd "$d" && git -c safe.directory="$PWD" reset --mixed harbor-base >/dev/null 2>&1) || true\\n'
        )
        return (
            'set +e\\n'
            f'ROOTS="{_DEFAULT_REPO_ROOTS}"\\n'
            'if [ -n "${HARBOR_REPO_PATHS:-}" ]; then ROOTS="$ROOTS $(echo "$HARBOR_REPO_PATHS" | tr ":" " ")"; fi\\n'
            'EXISTING=""\\n'
            'for r in $ROOTS; do [ -e "$r" ] && EXISTING="$EXISTING $r"; done\\n'
            'REPOS=$(find $EXISTING -maxdepth 3 \\( -type d -o -type f \\) -name .git 2>/dev/null | sort -u)\\n'
            'for gitdir in $REPOS; do\\n'
            '  d=$(dirname "$gitdir")\\n'
            f'{per_repo}'
            'done\\n'
        )

    assert action == "diff" and turn is not None and prev_turn is not None
''')
replace(claude_wrapper,
        '        # Post-run: build trajectory from session logs\n',
        '''        # Expose marker-committed changes to Harbor's verifier patch collector.
        try:
            await environment.exec(
                command=_repo_discovery_cmd(action="restore"),
                cwd="/", env={}, timeout_sec=90,
            )
        except Exception as e:
            log.warning("Failed to restore final worktree for verifier: %s", e)

        # Post-run: build trajectory from session logs
''')

e2b = root / "external/harbor/src/harbor/environments/e2b.py"
replace(e2b, '            timeout=5_400,', '            timeout=3_600,')

installer = root / "external/harbor/src/harbor/agents/installed/install-codex.sh.j2"
replace(installer,
        'apt-get update && apt-get install -y curl ripgrep',
        'apt-get update && apt-get install -y curl ripgrep nodejs npm')
replace(installer,
        '    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.2/install.sh | bash\n\n'
        '    export NVM_DIR="$HOME/.nvm"\n'
        '    # Source nvm with || true to handle nvm.sh\'s internal non-zero returns\n'
        '    \\. "$NVM_DIR/nvm.sh" || true\n'
        '    # Verify NVM loaded successfully\n'
        '    command -v nvm &>/dev/null || { echo "Error: NVM failed to load" >&2; exit 1; }\n\n'
        '    nvm install 22\n'
        '    nvm alias default 22\n',
        '    node --version\n')

codex_wrapper = root / "src/user_agent/agents/user_enabled_codex.py"
replace(codex_wrapper,
        '        except Exception as e:\n            log.warning("Failed to populate context post-run: %s", e)\n',
        '        except Exception as e:\n            log.warning("Failed to populate context post-run: %s", e)\n\n'
        '        # Exclude disposable Codex plugin cache from Harbor artifacts.\n'
        '        try:\n'
        '            await environment.exec(command="rm -rf /logs/agent/.tmp/plugins")\n'
        '        except Exception as e:\n'
        '            log.debug("temporary plugin-cache cleanup failed: %s", e)\n')

sentinel = root / "src/eval_infra_sentinel.py"
replace(sentinel,
        '    provider_error_texts: list[str] = field(default_factory=list)\n',
        '    provider_error_texts: list[str] = field(default_factory=list)\n'
        '    # Codex CLI writes fatal JSON events to per-command stdout files.\n'
        '    codex_fatal_texts: list[str] = field(default_factory=list)\n')
replace(sentinel,
        '        sig.empty_transcript_names = [p.name for p in present]\n    return sig\n',
        '        sig.empty_transcript_names = [p.name for p in present]\n'
        '    unsupported = "model is not supported when using Codex with a ChatGPT account"\n'
        '    for stdout in agent_dir.glob("command-*/stdout.txt"):\n'
        '        try:\n'
        '            for line in stdout.read_text(errors="replace").splitlines():\n'
        '                if unsupported.lower() in line.lower():\n'
        '                    sig.codex_fatal_texts.append(line[:500])\n'
        '        except OSError:\n'
        '            continue\n'
        '    return sig\n')
replace(sentinel,
        '\n\ndef _detect_no_agent_progress(sig: TrialSignals)',
        '\n\ndef _detect_codex_unsupported_model(sig: TrialSignals) -> tuple[bool, str, dict[str, Any]]:\n'
        '    if not sig.codex_fatal_texts:\n'
        '        return False, "", {}\n'
        '    return True, (\n'
        '        "Codex host authentication rejected the requested action model; "\n'
        '        "the coding agent never ran"\n'
        '    ), {"hit_count": len(sig.codex_fatal_texts), "sample": sig.codex_fatal_texts[0][:200]}\n'
        '\n\ndef _detect_no_agent_progress(sig: TrialSignals)')
replace(sentinel,
        'DETECTORS: list[tuple[str, Any]] = [\n',
        'DETECTORS: list[tuple[str, Any]] = [\n'
        '    ("codex_unsupported_model", _detect_codex_unsupported_model),\n')
