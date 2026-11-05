# Generic turn-replay sandbox: Python 3.12 + Node 22 + git + Claude Code (pinned to the host version).
# The repo worktree and the CLAUDE_CONFIG_DIR are bind-mounted at run time; nothing task-specific is baked in.
FROM python:3.12-slim
ARG CC_VERSION=2.1.273
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates ripgrep \
 && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && apt-get install -y --no-install-recommends nodejs \
 && npm install -g @anthropic-ai/claude-code@${CC_VERSION} pnpm@9 && pip install --no-cache-dir uv \
 && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 agent && mkdir -p /workspace && chown agent /workspace
USER agent
ENV PATH=/home/agent/.local/bin:$PATH
WORKDIR /workspace
