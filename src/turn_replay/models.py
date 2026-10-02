from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.pilot_study.common import read_json


class ReplayValidationError(ValueError):
    """An instance contains missing or malformed agent-visible inputs."""


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class Conversation:
    history: tuple[Message, ...]
    current_user_message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "history": [
                {"role": message.role, "content": message.content}
                for message in self.history
            ],
            "current_user_message": self.current_user_message,
        }


@dataclass(frozen=True)
class ReplayInstance:
    """Only data that is legal to expose to the coding agent.

    Deliberately absent: evaluation.json and next_user_message. Evaluation data
    has a separate lifecycle and no loader in this package.
    """

    root: Path
    repo: Path
    conversation: Conversation
    metadata: dict[str, Any]
    instance_id: str

    @classmethod
    def load(cls, root: Path) -> "ReplayInstance":
        root = root.resolve()
        repo = root / "repo"
        conversation_path = root / "conversation.json"
        metadata_path = root / "metadata.json"
        missing = [
            str(path)
            for path in (repo, conversation_path, metadata_path)
            if not path.exists()
        ]
        if missing:
            raise ReplayValidationError(
                "missing required instance input(s): " + ", ".join(missing)
            )
        if not repo.is_dir():
            raise ReplayValidationError(f"repo is not a directory: {repo}")

        try:
            raw_conversation = read_json(conversation_path)
        except Exception as exc:
            raise ReplayValidationError(
                f"could not parse {conversation_path}: {exc}"
            ) from exc
        if not isinstance(raw_conversation, dict):
            raise ReplayValidationError("conversation.json must contain an object")
        raw_history = raw_conversation.get("history", [])
        if not isinstance(raw_history, list):
            raise ReplayValidationError("conversation.history must be a list")
        history: list[Message] = []
        for index, item in enumerate(raw_history):
            if not isinstance(item, dict):
                raise ReplayValidationError(
                    f"conversation.history[{index}] must be an object"
                )
            role, content = item.get("role"), item.get("content")
            if role not in {"user", "assistant"}:
                raise ReplayValidationError(
                    f"conversation.history[{index}].role must be user or assistant"
                )
            if not isinstance(content, str):
                raise ReplayValidationError(
                    f"conversation.history[{index}].content must be a string"
                )
            history.append(Message(role=role, content=content))
        current = raw_conversation.get("current_user_message")
        if not isinstance(current, str) or not current.strip():
            raise ReplayValidationError(
                "conversation.current_user_message must be a non-empty string"
            )

        try:
            metadata = read_json(metadata_path)
        except Exception as exc:
            raise ReplayValidationError(f"could not parse {metadata_path}: {exc}") from exc
        if not isinstance(metadata, dict):
            raise ReplayValidationError("metadata.json must contain an object")
        instance_id = metadata.get("instance_id", root.name)
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise ReplayValidationError("metadata.instance_id must be a non-empty string")
        if Path(instance_id).name != instance_id:
            raise ReplayValidationError("metadata.instance_id must be a single path component")

        return cls(
            root=root,
            repo=repo,
            conversation=Conversation(tuple(history), current),
            metadata=metadata,
            instance_id=instance_id,
        )

    def in_repo_working_directory(self, copied_repo: Path) -> Path:
        environment = self.metadata.get("environment") or {}
        if not isinstance(environment, dict):
            raise ReplayValidationError("metadata.environment must be an object when present")
        relative = environment.get("working_directory", ".") or "."
        if not isinstance(relative, str):
            raise ReplayValidationError("environment.working_directory must be a string")
        candidate = (copied_repo / relative).resolve()
        copied_root = copied_repo.resolve()
        if candidate != copied_root and copied_root not in candidate.parents:
            raise ReplayValidationError("environment.working_directory escapes repo/")
        if not candidate.is_dir():
            raise ReplayValidationError(
                f"environment.working_directory does not exist in copied repo: {relative}"
            )
        return candidate
