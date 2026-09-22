<!-- repo: oddessentials/ado-git-repo-insights path: CLAUDE.md
     commit: ae72c1afb9da3b1961c57223d7df56b51ea59d49 date: 2026-04-04T17:20:25+00:00
     attribution: human_only chars: 1578 -->

# ado-git-repo-insights Development Guidelines

## Active Technologies
- Python 3.12+ (backend, scripts, tests)
- TypeScript 6.x (extension UI, tests)
- pandas (aggregation), pytest (Python tests), Jest (TypeScript tests)
- ruff (Python linting), ESLint (TypeScript linting), mypy (type checking)
- esbuild (IIFE bundler), Playwright (smoke tests)
- Python 3.12+ (backend pipeline), TypeScript 6.x (extension UI — no changes needed) + pandas (aggregation), requests (ADO API client), sqlite3 (persistence), pytest (testing) (052-review-time-pipeline)
- SQLite (source of truth per Constitution Principle V), JSON rollup files (aggregation output) (052-review-time-pipeline)

## Project Structure

```text
src/
tests/
```

## Commands

```bash
pytest                                    # Python tests
cd extension && pnpm test                 # Extension tests
python scripts/run_pr_preflight.py        # Authoritative local PR gate
```

## Code Style

Python 3.12+ (backend), TypeScript 6.x (frontend): Follow standard conventions

## Recent Changes
- 052-review-time-pipeline: Added Python 3.12+ (backend pipeline), TypeScript 6.x (extension UI — no changes needed) + pandas (aggregation), requests (ADO API client), sqlite3 (persistence), pytest (testing)
- 049-cross-platform-hardening: Replaced PowerShell ACL check with Python, build-demo.sh with build_demo.py, added SETUP/INFRA/GATE error categories, Node engine enforcement
- 048-https-github-com: mypy strict mode on src/ with disallow_any_generics = true

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
