#!/usr/bin/env python3
"""
ember -- the agent core: a single-module agentic loop.

Kept whole on purpose. stdlib plus the openai package, nothing else, so the
whole control flow of an agent is readable end to end in one sitting.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import difflib
import fnmatch
import hashlib
import json
import logging
import os
import pathlib
import platform
import random
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import openai
from openai import OpenAI

from ember import config as app_config


DEFAULT_API_BASE_URL = app_config.DEFAULT_API_BASE_URL
DEFAULT_API_KEY = app_config.DEFAULT_API_KEY
DEFAULT_MODEL = app_config.DEFAULT_MODEL
DEFAULT_WORKSPACE = app_config.DEFAULT_WORKSPACE
DEFAULT_MAX_STEPS = app_config.DEFAULT_MAX_STEPS
DEFAULT_CONTEXT_WINDOW = app_config.DEFAULT_CONTEXT_WINDOW
DEFAULT_TOOL_TIMEOUT = app_config.DEFAULT_TOOL_TIMEOUT
DEFAULT_COMPACT_THRESHOLD = app_config.DEFAULT_COMPACT_THRESHOLD
DEFAULT_KEEP_FRESH = app_config.DEFAULT_KEEP_FRESH

MEMORY_MAX_FILE_BYTES = app_config.MEMORY_MAX_FILE_BYTES
MAX_FILE_READ_BYTES = app_config.MAX_FILE_READ_BYTES
EXEC_STDOUT_CAP = app_config.EXEC_STDOUT_CAP
EXEC_STDERR_CAP = app_config.EXEC_STDERR_CAP
GREP_RESULTS_CAP = app_config.GREP_RESULTS_CAP
GREP_OUTPUT_CAP = app_config.GREP_OUTPUT_CAP


def load_dotenv(dotenv_path: str | os.PathLike[str] | None = None) -> None:
    if dotenv_path is None:
        configured = Path(app_config.DOTENV_PATH)
        if configured.is_absolute():
            path = configured
        else:
            path = Path(__file__).resolve().parent.parent / configured
    else:
        path = Path(dotenv_path)
    if not path.exists():
        return

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid .env line {line_number}: {raw_line!r}")

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid .env line {line_number}: empty key")

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ.setdefault(key, value)


load_dotenv()


def _env(name: str, default: Any, kind: str = "str") -> Any:
    raw = os.environ.get(name)
    if raw is None:
        return default

    if kind == "str":
        return raw
    if kind == "int":
        return int(raw)
    if kind == "float":
        return float(raw)
    if kind == "bool":
        value = raw.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Invalid boolean env var for {name}: {raw!r}")
    raise ValueError(f"Unsupported env kind: {kind}")


@dataclass
class Config:
    api_base_url: str = DEFAULT_API_BASE_URL
    api_key: str = DEFAULT_API_KEY
    model: str = DEFAULT_MODEL
    workspace: str = DEFAULT_WORKSPACE
    max_steps: int = DEFAULT_MAX_STEPS
    context_window: int = DEFAULT_CONTEXT_WINDOW
    tool_timeout: float = DEFAULT_TOOL_TIMEOUT
    yolo: bool = False
    compact_threshold: float = DEFAULT_COMPACT_THRESHOLD
    compact_keep_fresh: int = DEFAULT_KEEP_FRESH
    llm_max_retries: int = app_config.DEFAULT_LLM_MAX_RETRIES
    llm_retry_base_delay: float = app_config.DEFAULT_LLM_RETRY_BASE_DELAY

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            api_base_url=_env("EMBER_API_BASE_URL", DEFAULT_API_BASE_URL, "str"),
            api_key=_env("EMBER_API_KEY", DEFAULT_API_KEY, "str"),
            model=_env("EMBER_MODEL", DEFAULT_MODEL, "str"),
            workspace=_env("EMBER_WORKSPACE", DEFAULT_WORKSPACE, "str"),
            max_steps=_env("EMBER_MAX_STEPS", DEFAULT_MAX_STEPS, "int"),
            context_window=_env(
                "EMBER_CONTEXT_WINDOW",
                DEFAULT_CONTEXT_WINDOW,
                "int",
            ),
            tool_timeout=_env("EMBER_TOOL_TIMEOUT", DEFAULT_TOOL_TIMEOUT, "float"),
            yolo=_env("EMBER_YOLO", False, "bool"),
            compact_threshold=_env(
                "EMBER_COMPACT_THRESHOLD",
                DEFAULT_COMPACT_THRESHOLD,
                "float",
            ),
            compact_keep_fresh=_env(
                "EMBER_COMPACT_KEEP_FRESH",
                DEFAULT_KEEP_FRESH,
                "int",
            ),
            llm_max_retries=_env(
                "EMBER_LLM_MAX_RETRIES",
                app_config.DEFAULT_LLM_MAX_RETRIES,
                "int",
            ),
            llm_retry_base_delay=_env(
                "EMBER_LLM_RETRY_BASE_DELAY",
                app_config.DEFAULT_LLM_RETRY_BASE_DELAY,
                "float",
            ),
        )


log = logging.getLogger("ember")


def setup_logging(level: str | int = "INFO") -> None:
    if isinstance(level, str):
        normalized = level.upper()
        numeric_level = getattr(logging, normalized, None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Unknown log level: {level}")
    else:
        numeric_level = level

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

    logger = logging.getLogger("ember")
    logger.handlers.clear()
    logger.setLevel(numeric_level)
    logger.addHandler(handler)
    logger.propagate = False


Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None
    timestamp: float = 0.0

    def to_openai(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}

        if self.role == "assistant" and self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                }
                for call in self.tool_calls
            ]

        if self.role == "tool":
            if not self.tool_call_id:
                raise ValueError("tool messages require tool_call_id")
            payload["tool_call_id"] = self.tool_call_id

        return payload

    @classmethod
    def from_openai_response(cls, choice_msg: Any) -> "Message":
        content = choice_msg.content or ""
        parsed_tool_calls = None

        if getattr(choice_msg, "tool_calls", None):
            parsed_tool_calls = []
            for raw_call in choice_msg.tool_calls:
                function_obj = getattr(raw_call, "function", None)
                parsed_tool_calls.append(
                    ToolCall(
                        id=getattr(raw_call, "id", ""),
                        name=getattr(function_obj, "name", ""),
                        arguments=getattr(function_obj, "arguments", "") or "",
                    )
                )

        role = getattr(choice_msg, "role", "assistant")
        if role not in {"system", "user", "assistant", "tool"}:
            role = "assistant"

        return cls(
            role=role,
            content=content,
            tool_calls=parsed_tool_calls,
            timestamp=time.time(),
        )


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    requires_confirmation: bool = False
    confirmation_preview: Optional[Callable[..., str]] = None


@dataclass
class ToolContext:
    confiner: Any
    workspace: str
    memory: Any
    config: Config
    cancel_flag: Any = None


# -----------------------------------------------------------------------------
# §5 Confiner -- workspace path safety
# -----------------------------------------------------------------------------


class Confiner:
    """Canonicalize paths into a workspace and reject escapes."""

    def __init__(self, workspace: str | os.PathLike[str]) -> None:
        root = Path(workspace).resolve(strict=False)
        root.mkdir(parents=True, exist_ok=True)
        self.root = root.resolve(strict=True)

    def resolve(self, path: str | os.PathLike[str]) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            target = candidate.resolve(strict=False)
        else:
            target = (self.root / candidate).resolve(strict=False)

        if not target.exists():
            parent = target.parent
            if not parent.exists():
                raise FileNotFoundError(f"Parent missing: {parent}")
            parent = parent.resolve(strict=True)
            target = parent / target.name

        try:
            target.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError(
                f"Path escape blocked: {path} -> {target}, outside workspace {self.root}"
            ) from exc
        return target

    def relpath(self, p: str | os.PathLike[str]) -> str:
        path = Path(p)
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)


# -----------------------------------------------------------------------------
# §6 Token estimation + truncation utilities
# -----------------------------------------------------------------------------


_CODE_HINTS = ("{", "[", "def ", "function ", "class ", "import ", "=>", "};", "):")


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    sample = text[:200]
    divisor = 3 if any(hint in sample for hint in _CODE_HINTS) else 4
    return max(1, len(text) // divisor)


def estimate_messages_tokens(messages: list[Message]) -> int:
    total = 0
    for m in messages:
        total += estimate_tokens(m.content or "") + 4
        if m.tool_calls:
            for tc in m.tool_calls:
                total += estimate_tokens((tc.name or "") + (tc.arguments or "")) + 4
    return total


def _tool_truncation_budget(context_window: int) -> int:
    budget = int(context_window * 0.3 * 4)
    return max(2000, min(400_000, budget))


def truncate_oversized_tool_results(
    messages: list[Message], context_window: int
) -> None:
    max_chars = _tool_truncation_budget(context_window)
    for m in messages:
        if m.role != "tool" or not m.content:
            continue
        if len(m.content) <= max_chars:
            continue
        original_len = len(m.content)
        kept = m.content[:max_chars]
        newline_idx = kept.rfind("\n")
        if newline_idx > max_chars // 2:
            kept = kept[:newline_idx]
        m.content = (
            kept
            + f"\n\n[...truncated, original was {original_len} chars...]"
        )


def repair_orphaned_tool_calls(messages: list[Message]) -> None:
    n = len(messages)
    for i, m in enumerate(messages):
        if m.role != "assistant" or not m.tool_calls:
            continue
        needed_ids = {tc.id for tc in m.tool_calls}
        found_ids: set[str] = set()
        for j in range(i + 1, n):
            other = messages[j]
            if other.role == "tool" and other.tool_call_id in needed_ids:
                found_ids.add(other.tool_call_id)
        if needed_ids == found_ids:
            continue
        kept_calls = [tc for tc in m.tool_calls if tc.id in found_ids]
        m.tool_calls = kept_calls or None
        if not m.tool_calls and not (m.content or "").strip():
            m.content = "[truncated tool calls]"


def repair_orphaned_tool_results(messages: list[Message]) -> None:
    valid_ids: set[str] = set()
    kept: list[Message] = []
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            for tc in m.tool_calls:
                valid_ids.add(tc.id)
            kept.append(m)
        elif m.role == "tool":
            if m.tool_call_id and m.tool_call_id in valid_ids:
                kept.append(m)
            else:
                log.warning(
                    "dropping orphaned tool result: tool_call_id=%s",
                    m.tool_call_id,
                )
        else:
            kept.append(m)
    if len(kept) != len(messages):
        messages.clear()
        messages.extend(kept)


def truncate_context_file(text: str, max_chars: int = 30_000) -> str:
    if not text or len(text) <= max_chars:
        return text
    head_chars = int(max_chars * 0.7)
    tail_chars = int(max_chars * 0.2)
    head = text[:head_chars]
    tail = text[-tail_chars:]
    dropped = len(text) - head_chars - tail_chars
    return head + f"\n\n[...truncated {dropped} chars...]\n\n" + tail


# -----------------------------------------------------------------------------
# §7 Tool registry -- decorator-based
# -----------------------------------------------------------------------------


TOOLS: dict[str, Tool] = {}


def tool(
    *,
    name: str,
    description: str,
    parameters: dict[str, Any],
    requires_confirmation: bool = False,
    confirmation_preview: Optional[Callable[..., str]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in TOOLS:
            raise ValueError(f"duplicate tool: {name}")
        TOOLS[name] = Tool(
            name=name,
            description=description,
            parameters=parameters,
            handler=fn,
            requires_confirmation=requires_confirmation,
            confirmation_preview=confirmation_preview,
        )
        return fn

    return decorator


class ToolRegistry:
    def __init__(self, ctx: ToolContext) -> None:
        self.ctx = ctx

    def openai_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in TOOLS.values()
        ]

    def dispatch(self, name: str, arguments_json: str) -> ToolResult:
        tool_def = TOOLS.get(name)
        if tool_def is None:
            return ToolResult(
                f"Unknown tool: {name}. Available: {', '.join(sorted(TOOLS))}",
                is_error=True,
            )
        try:
            if arguments_json and arguments_json.strip():
                args = json.loads(arguments_json)
            else:
                args = {}
        except json.JSONDecodeError as exc:
            return ToolResult(f"Invalid JSON arguments: {exc}", is_error=True)
        if not isinstance(args, dict):
            return ToolResult(
                f"Tool arguments must be a JSON object, got {type(args).__name__}",
                is_error=True,
            )
        try:
            raw = tool_def.handler(args, self.ctx)
        except Exception as exc:  # noqa: BLE001
            log.exception("tool %s raised", name)
            return ToolResult(
                f"Tool error: {type(exc).__name__}: {exc}",
                is_error=True,
            )
        if isinstance(raw, ToolResult):
            return raw
        return ToolResult(str(raw))


# -----------------------------------------------------------------------------
# §8 SQLite helpers -- connect, schema, jittered write retry
# -----------------------------------------------------------------------------


SCHEMA_VERSION = 1

_SCHEMA_STATEMENTS = [
    "CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)",
    """CREATE TABLE IF NOT EXISTS memory_files (
        path TEXT PRIMARY KEY,
        hash TEXT NOT NULL,
        mtime REAL NOT NULL,
        size INTEGER NOT NULL,
        indexed_at REAL NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS memory_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT NOT NULL REFERENCES memory_files(path) ON DELETE CASCADE,
        start_line INTEGER NOT NULL,
        end_line INTEGER NOT NULL,
        text TEXT NOT NULL,
        hash TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_chunks_path ON memory_chunks(path)",
    """CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
        text,
        content=memory_chunks,
        content_rowid=id,
        tokenize='porter unicode61'
    )""",
    """CREATE TRIGGER IF NOT EXISTS memory_chunks_ai AFTER INSERT ON memory_chunks BEGIN
        INSERT INTO memory_fts(rowid, text) VALUES (new.id, new.text);
    END""",
    """CREATE TRIGGER IF NOT EXISTS memory_chunks_ad AFTER DELETE ON memory_chunks BEGIN
        INSERT INTO memory_fts(memory_fts, rowid, text) VALUES ('delete', old.id, old.text);
    END""",
    """CREATE TRIGGER IF NOT EXISTS memory_chunks_au AFTER UPDATE ON memory_chunks BEGIN
        INSERT INTO memory_fts(memory_fts, rowid, text) VALUES ('delete', old.id, old.text);
        INSERT INTO memory_fts(rowid, text) VALUES (new.id, new.text);
    END""",
]


def db_connect(path: str | os.PathLike[str]) -> sqlite3.Connection:
    conn = sqlite3.connect(
        str(path),
        timeout=1.0,
        isolation_level=None,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def _is_busy_error(exc: sqlite3.OperationalError) -> bool:
    msg = str(exc).lower()
    return "locked" in msg or "busy" in msg


def _retry_sleep() -> None:
    time.sleep((20 + random.random() * 130) / 1000.0)


def _execute_write(
    conn: sqlite3.Connection,
    sql: str,
    params: tuple | list = (),
    max_attempts: int = 15,
) -> sqlite3.Cursor:
    last_exc: Optional[sqlite3.OperationalError] = None
    for _ in range(max_attempts):
        try:
            return conn.execute(sql, params)
        except sqlite3.OperationalError as exc:
            if not _is_busy_error(exc):
                raise
            last_exc = exc
            _retry_sleep()
    raise RuntimeError(
        f"SQLite write failed after {max_attempts} attempts: {last_exc}"
    )


def executemany_write(
    conn: sqlite3.Connection,
    sql: str,
    rows: list | tuple,
    max_attempts: int = 15,
) -> sqlite3.Cursor:
    last_exc: Optional[sqlite3.OperationalError] = None
    for _ in range(max_attempts):
        try:
            return conn.executemany(sql, rows)
        except sqlite3.OperationalError as exc:
            if not _is_busy_error(exc):
                raise
            last_exc = exc
            _retry_sleep()
    raise RuntimeError(
        f"SQLite executemany failed after {max_attempts} attempts: {last_exc}"
    )


def execute_tx(
    conn: sqlite3.Connection,
    fn: Callable[[sqlite3.Connection], Any],
    max_attempts: int = 15,
) -> Any:
    last_exc: Optional[sqlite3.OperationalError] = None
    for _ in range(max_attempts):
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            if not _is_busy_error(exc):
                raise
            last_exc = exc
            _retry_sleep()
            continue
        try:
            result = fn(conn)
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        try:
            conn.execute("COMMIT")
        except sqlite3.OperationalError as exc:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            if not _is_busy_error(exc):
                raise
            last_exc = exc
            _retry_sleep()
            continue
        return result
    raise RuntimeError(
        f"SQLite transaction failed after {max_attempts} attempts: {last_exc}"
    )


def init_schema(conn: sqlite3.Connection) -> None:
    for stmt in _SCHEMA_STATEMENTS:
        conn.execute(stmt)
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    if row is None:
        _execute_write(
            conn,
            "INSERT INTO schema_version(version) VALUES (?)",
            (SCHEMA_VERSION,),
        )


def dbschema_version_get(conn: sqlite3.Connection) -> Optional[int]:
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    return int(row[0]) if row else None


def dbschema_version_set(conn: sqlite3.Connection, n: int) -> None:
    _execute_write(conn, "UPDATE schema_version SET version = ?", (n,))


# -----------------------------------------------------------------------------
# §9 MemoryStore -- markdown chunking, FTS5 search, append-only write
# -----------------------------------------------------------------------------


_FTS_SPECIALS = set('^:*"()')
_FTS_KEYWORDS = {"NOT", "OR", "AND", "NEAR"}


def sanitize_fts_query(q: str) -> str:
    stripped = q.strip()
    if not stripped:
        return '""'
    tokens = stripped.split()
    has_keyword = any(tok in _FTS_KEYWORDS for tok in tokens)
    has_special = any(ch in _FTS_SPECIALS for ch in stripped)
    if not (has_keyword or has_special):
        return stripped
    terms = [f'"{t}"' for t in re.findall(r"\w+", stripped) if t]
    if not terms:
        return '""'
    return " OR ".join(terms)


class MemoryStore:
    """Markdown-backed memory store indexed into SQLite FTS5."""

    def __init__(
        self,
        workspace: str | os.PathLike[str],
        db_path: Optional[str | os.PathLike[str]] = None,
    ) -> None:
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.memory_dir = self.workspace / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_md_path = self.workspace / "MEMORY.md"
        self.agents_md_path = self.workspace / "AGENTS.md"
        self.db_path = Path(db_path) if db_path else self.workspace / ".ember.db"
        self.conn = db_connect(str(self.db_path))
        init_schema(self.conn)
        self._last_sync_check: float = 0.0

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:  # noqa: BLE001
            pass

    # --- chunking -----------------------------------------------------------

    def _chunk_by_window(
        self, lines: list[str], start: int, end: int
    ) -> list[tuple[int, int, str]]:
        out: list[tuple[int, int, str]] = []
        window = 50
        overlap = 5
        i = start
        if end <= start:
            return out
        while i < end:
            win_end = min(i + window, end)
            text = "\n".join(lines[i:win_end])
            out.append((i + 1, win_end, text))
            if win_end >= end:
                break
            i = max(i + 1, win_end - overlap)
        return out

    def _chunk_markdown(self, text: str) -> list[tuple[int, int, str]]:
        lines = text.splitlines()
        if not lines:
            return []
        heading_indices = [i for i, ln in enumerate(lines) if ln.startswith("## ")]
        if not heading_indices:
            return self._chunk_by_window(lines, 0, len(lines))

        section_starts = sorted(set([0] + heading_indices))
        chunks: list[tuple[int, int, str]] = []
        for idx, start in enumerate(section_starts):
            end = (
                section_starts[idx + 1]
                if idx + 1 < len(section_starts)
                else len(lines)
            )
            if end <= start:
                continue
            section_text = "\n".join(lines[start:end])
            if (end - start) > 50 or len(section_text) > 2000:
                chunks.extend(self._chunk_by_window(lines, start, end))
            else:
                chunks.append((start + 1, end, section_text))
        return chunks

    # --- file sync ----------------------------------------------------------

    def _file_fingerprint(self, path: Path) -> tuple[str, float, int]:
        st = path.stat()
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                block = f.read(65536)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()[:16], st.st_mtime, st.st_size

    def _file_relpath(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.workspace)).replace(os.sep, "/")

    def _sync_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            try:
                rel = self._file_relpath(path)
            except ValueError:
                return
            _execute_write(
                self.conn,
                "DELETE FROM memory_files WHERE path = ?",
                (rel,),
            )
            return

        size = path.stat().st_size
        if size > MEMORY_MAX_FILE_BYTES:
            log.warning(
                "memory: skipping %s (%.1fMB > %.1fMB cap)",
                path,
                size / 1e6,
                MEMORY_MAX_FILE_BYTES / 1e6,
            )
            return

        rel = self._file_relpath(path)
        file_hash, mtime, size = self._file_fingerprint(path)

        row = self.conn.execute(
            "SELECT hash FROM memory_files WHERE path = ?", (rel,)
        ).fetchone()
        if row is not None and row["hash"] == file_hash:
            return

        text = path.read_text(encoding="utf-8", errors="replace")
        chunks = self._chunk_markdown(text)

        def _do(c: sqlite3.Connection) -> None:
            c.execute("DELETE FROM memory_files WHERE path = ?", (rel,))
            c.execute(
                "INSERT INTO memory_files(path, hash, mtime, size, indexed_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (rel, file_hash, mtime, size, time.time()),
            )
            for (start_line, end_line, chunk_text) in chunks:
                chunk_hash = hashlib.sha256(
                    chunk_text.encode("utf-8")
                ).hexdigest()[:16]
                c.execute(
                    "INSERT INTO memory_chunks(path, start_line, end_line, text, hash) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (rel, start_line, end_line, chunk_text, chunk_hash),
                )

        execute_tx(self.conn, _do)

    def _collect_tracked_paths(self) -> list[Path]:
        paths: list[Path] = []
        if self.memory_md_path.exists():
            paths.append(self.memory_md_path)
        if self.memory_dir.exists():
            for p in sorted(self.memory_dir.rglob("*.md")):
                paths.append(p)
        return paths

    def sync_all(self) -> None:
        paths = self._collect_tracked_paths()
        for p in paths:
            try:
                self._sync_file(p)
            except Exception:  # noqa: BLE001
                log.exception("memory sync failed for %s", p)
        indexed = {
            row["path"]
            for row in self.conn.execute("SELECT path FROM memory_files")
        }
        current = {self._file_relpath(p) for p in paths}
        for gone in indexed - current:
            _execute_write(
                self.conn,
                "DELETE FROM memory_files WHERE path = ?",
                (gone,),
            )
        self._last_sync_check = time.time()

    def _sync_if_stale(self, max_age: float = 5.0) -> None:
        now = time.time()
        if now - self._last_sync_check < max_age:
            return
        self.sync_all()

    # --- search / append ----------------------------------------------------

    def search(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        self._sync_if_stale()
        safe_q = sanitize_fts_query(query)
        try:
            rows = self.conn.execute(
                "SELECT c.path AS path, c.start_line AS start_line, "
                "c.end_line AS end_line, c.text AS text, "
                "bm25(memory_fts) AS rank "
                "FROM memory_fts JOIN memory_chunks c ON c.id = memory_fts.rowid "
                "WHERE memory_fts MATCH ? ORDER BY rank LIMIT ?",
                (safe_q, max_results),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            log.warning("fts search failed: %s (query=%r)", exc, safe_q)
            return []
        return [
            {
                "path": r["path"],
                "start_line": r["start_line"],
                "end_line": r["end_line"],
                "text": r["text"],
                "score": r["rank"],
            }
            for r in rows
        ]

    def format_search_results(self, hits: list[dict[str, Any]]) -> str:
        if not hits:
            return "No memory results."
        blocks: list[str] = []
        for h in hits:
            header = (
                f"--- Source: {h['path']}#L{h['start_line']}-L{h['end_line']} "
                f"(score: {h['score']:.3f}) ---"
            )
            blocks.append(f"{header}\n{h['text']}")
        return "\n\n".join(blocks)

    def append(self, path: str, content: str) -> None:
        rel = str(path).replace("\\", "/").lstrip("./")
        is_memory_md = rel == "MEMORY.md"
        is_memory_dir_child = False
        if rel.startswith("memory/"):
            tail = rel[len("memory/"):]
            is_memory_dir_child = (
                "/" not in tail
                and tail.endswith(".md")
                and tail not in (".md", "")
            )
        if not (is_memory_md or is_memory_dir_child):
            raise ValueError(
                f"memory append rejects path {rel!r}; "
                "only MEMORY.md or memory/<name>.md allowed"
            )
        target = (self.workspace / rel).resolve()
        try:
            target.relative_to(self.workspace)
        except ValueError as exc:
            raise PermissionError(f"memory path escape: {rel}") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        body = content if content.endswith("\n") else content + "\n"
        with open(target, "a", encoding="utf-8") as f:
            f.write(body)
            f.flush()
            os.fsync(f.fileno())
        self._sync_file(target)

    def flush_conversation(self, messages: list[Message], llm: Any) -> None:
        if not messages:
            return
        relevant = [
            m for m in messages if m.role in ("user", "assistant") and (m.content or "").strip()
        ]
        if not relevant:
            return
        text_block = "\n\n".join(
            f"[{m.role}] {(m.content or '')[:2000]}" for m in relevant
        )
        prompt = [
            {
                "role": "system",
                "content": "You extract durable memory from conversations.",
            },
            {
                "role": "user",
                "content": (
                    "Extract from this conversation: decisions, facts, "
                    "preferences, open TODOs, personal info. Output as "
                    "markdown bullets. If nothing worth keeping, output "
                    "exactly NO_DURABLE_MEMORY.\n\n" + text_block
                ),
            },
        ]
        try:
            summary = (llm.complete_plain(prompt) or "").strip()
        except Exception:  # noqa: BLE001
            log.exception("memory flush LLM call failed")
            return
        if not summary or summary.upper() == "NO_DURABLE_MEMORY":
            return
        today = datetime.date.today().isoformat()
        body = f"## {today}\n\n{summary}\n\n"
        self.append(f"memory/{today}.md", body)

    # --- context file readers ----------------------------------------------

    def read_memory_md(self, max_chars: int = 30_000) -> str:
        if not self.memory_md_path.exists():
            return ""
        try:
            text = self.memory_md_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return truncate_context_file(text, max_chars)

    def read_agents_md(self, max_chars: int = 30_000) -> str:
        if not self.agents_md_path.exists():
            return ""
        try:
            text = self.agents_md_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return truncate_context_file(text, max_chars)


# -----------------------------------------------------------------------------
# §10 SessionStore -- JSONL append, crash-safe load, atomic rewrite
# -----------------------------------------------------------------------------


_SESSION_ID_RE = re.compile(r"^\d{8}_\d{6}_[0-9a-f]{6}$")


def _message_to_dict(msg: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": msg.role,
        "content": msg.content,
        "timestamp": msg.timestamp,
    }
    if msg.tool_call_id:
        payload["tool_call_id"] = msg.tool_call_id
    if msg.tool_calls:
        payload["tool_calls"] = [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
            for tc in msg.tool_calls
        ]
    return payload


def _dict_to_message(d: dict[str, Any]) -> Message:
    tool_calls = None
    raw_calls = d.get("tool_calls")
    if raw_calls:
        tool_calls = [
            ToolCall(
                id=str(raw.get("id", "")),
                name=str(raw.get("name", "")),
                arguments=str(raw.get("arguments", "")),
            )
            for raw in raw_calls
        ]
    role = d.get("role", "user")
    if role not in {"system", "user", "assistant", "tool"}:
        role = "user"
    return Message(
        role=role,
        content=d.get("content", "") or "",
        tool_call_id=d.get("tool_call_id"),
        tool_calls=tool_calls,
        timestamp=float(d.get("timestamp", 0.0) or 0.0),
    )


class SessionStore:
    """JSONL session log with crash-safe load and atomic rewrite."""

    def __init__(self, workspace: str | os.PathLike[str]) -> None:
        self.workspace = Path(workspace).resolve()
        self.sessions_dir = self.workspace / ".sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def new_session_id(self) -> str:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = hashlib.sha256(os.urandom(16)).hexdigest()[:6]
        return f"{ts}_{suffix}"

    def _path(self, session_id: str) -> Path:
        if not _SESSION_ID_RE.match(session_id):
            raise ValueError(f"invalid session id: {session_id!r}")
        return self.sessions_dir / f"{session_id}.jsonl"

    def append(self, session_id: str, msg: Message) -> None:
        path = self._path(session_id)
        line = json.dumps(_message_to_dict(msg), ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def load(self, session_id: str) -> list[Message]:
        path = self._path(session_id)
        if not path.exists():
            return []
        raw = path.read_text(encoding="utf-8", errors="replace")
        lines = raw.split("\n")
        messages: list[Message] = []
        non_empty = [(i, ln) for i, ln in enumerate(lines) if ln.strip()]
        if not non_empty:
            return []
        last_idx = non_empty[-1][0]
        for i, ln in enumerate(lines):
            if not ln.strip():
                continue
            try:
                obj = json.loads(ln)
            except json.JSONDecodeError as exc:
                if i == last_idx:
                    log.warning(
                        "dropping malformed last line of session %s: %s",
                        session_id,
                        exc,
                    )
                    continue
                raise RuntimeError(
                    f"session corruption in {path} at line {i + 1}: {exc}"
                ) from exc
            messages.append(_dict_to_message(obj))
        return messages

    def rewrite(self, session_id: str, messages: list[Message]) -> None:
        path = self._path(session_id)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for msg in messages:
                f.write(
                    json.dumps(_message_to_dict(msg), ensure_ascii=False) + "\n"
                )
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def list_recent(self, n: int = 10) -> list[tuple[str, float, int]]:
        entries: list[tuple[str, float, int]] = []
        for path in self.sessions_dir.glob("*.jsonl"):
            sid = path.stem
            if not _SESSION_ID_RE.match(sid):
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            count = 0
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        if line.strip():
                            count += 1
            except OSError:
                continue
            entries.append((sid, mtime, count))
        entries.sort(key=lambda e: e[1], reverse=True)
        return entries[:n]

    def delete(self, session_id: str) -> None:
        path = self._path(session_id)
        try:
            path.unlink()
        except FileNotFoundError:
            pass


# -----------------------------------------------------------------------------
# §13 Confirmation UI + color helpers
# -----------------------------------------------------------------------------


C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_RED = "\033[31m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_BLUE = "\033[34m"
C_MAGENTA = "\033[35m"
C_CYAN = "\033[36m"


def _colors_supported() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    try:
        return sys.stdout.isatty()
    except Exception:  # noqa: BLE001
        return False


def colorize(text: str, color: str) -> str:
    if not _colors_supported():
        return text
    return f"{color}{text}{C_RESET}"


def confirm(
    tool_name: str,
    preview: str,
    yolo: bool,
    logger: logging.Logger,
) -> bool:
    if yolo:
        logger.warning("--yolo: auto-approving %s", tool_name)
        return True
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        logger.warning("no tty; denying %s", tool_name)
        return False
    header = colorize(f"\n-- Confirmation: {tool_name} --", C_YELLOW + C_BOLD)
    rule = colorize("-" * 48, C_YELLOW)
    print(header)
    print(preview)
    print(rule)
    try:
        reply = input("  Proceed? [y/N]: ")
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return reply.strip().lower() in ("y", "yes")


def format_diff_preview(old: str, new: str, path: str) -> str:
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff_iter = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=3,
    )
    out: list[str] = []
    for line in diff_iter:
        stripped = line.rstrip("\n")
        if line.startswith("+++") or line.startswith("---"):
            out.append(colorize(stripped, C_BOLD))
        elif line.startswith("@@"):
            out.append(colorize(stripped, C_CYAN))
        elif line.startswith("+"):
            out.append(colorize(stripped, C_GREEN))
        elif line.startswith("-"):
            out.append(colorize(stripped, C_RED))
        else:
            out.append(stripped)
    return "\n".join(out) if out else "(no textual diff)"


# -----------------------------------------------------------------------------
# §12 LLMClient -- OpenAI wrapper with streaming + retries
# -----------------------------------------------------------------------------


class ContextOverflowError(Exception):
    """Raised when the LLM signals a context-length overflow."""


_OVERFLOW_KEYWORDS = (
    "context_length_exceeded",
    "maximum context",
    "context window",
    "context length",
    "too many tokens",
)


def friendly_connection_error(exc: Exception, base_url: str) -> str:
    return (
        f"Cannot reach LLM at {base_url}. Is Ollama running? "
        "Try `ollama serve` or check EMBER_API_BASE_URL. "
        f"(underlying error: {type(exc).__name__}: {exc})"
    )


class LLMClient:
    """OpenAI SDK wrapper with streaming, retries, overflow detection."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key or "placeholder",
            base_url=config.api_base_url,
            timeout=120.0,
            max_retries=0,
        )
        self._last_usage: Any = None

    @property
    def last_usage(self) -> Any:
        return self._last_usage

    # --- streaming ----------------------------------------------------------

    def _call_streaming(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
    ) -> Message:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools

        try:
            stream = self.client.chat.completions.create(**kwargs)
        except TypeError:
            kwargs.pop("stream_options", None)
            stream = self.client.chat.completions.create(**kwargs)

        content_parts: list[str] = []
        tool_calls_acc: dict[int, dict[str, str]] = {}
        usage: Any = None
        any_text_printed = False

        for chunk in stream:
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage is not None:
                usage = chunk_usage
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue

            delta_content = getattr(delta, "content", None)
            if delta_content:
                if delta_content.strip() or any_text_printed:
                    sys.stdout.write(delta_content)
                    sys.stdout.flush()
                    any_text_printed = True
                content_parts.append(delta_content)

            delta_tool_calls = getattr(delta, "tool_calls", None) or []
            for tc_delta in delta_tool_calls:
                idx = getattr(tc_delta, "index", 0) or 0
                slot = tool_calls_acc.setdefault(
                    idx, {"id": "", "name": "", "arguments": ""}
                )
                if getattr(tc_delta, "id", None):
                    slot["id"] = tc_delta.id
                fn_obj = getattr(tc_delta, "function", None)
                if fn_obj is not None:
                    if getattr(fn_obj, "name", None):
                        slot["name"] = fn_obj.name
                    if getattr(fn_obj, "arguments", None):
                        slot["arguments"] += fn_obj.arguments

        if any_text_printed:
            sys.stdout.write("\n")
            sys.stdout.flush()

        self._last_usage = usage
        content = "".join(content_parts)
        tool_calls: Optional[list[ToolCall]] = None
        if tool_calls_acc:
            tool_calls = [
                ToolCall(
                    id=tool_calls_acc[k]["id"],
                    name=tool_calls_acc[k]["name"],
                    arguments=tool_calls_acc[k]["arguments"],
                )
                for k in sorted(tool_calls_acc)
            ]
        return Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
            timestamp=time.time(),
        )

    def _call_plain(self, messages: list[dict[str, Any]]) -> str:
        resp = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            stream=False,
        )
        self._last_usage = getattr(resp, "usage", None)
        choice = resp.choices[0]
        return choice.message.content or ""

    # --- public API with retry ---------------------------------------------

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        stream: bool = True,
    ) -> Message:
        return self._with_retry(
            lambda: self._call_streaming(messages, tools)
            if stream
            else Message(
                role="assistant",
                content=self._call_plain(messages),
                timestamp=time.time(),
            )
        )

    def complete_plain(self, messages: list[dict[str, Any]]) -> str:
        return self._with_retry(lambda: self._call_plain(messages))

    def _with_retry(self, fn: Callable[[], Any]) -> Any:
        max_retries = max(1, self.config.llm_max_retries)
        base_delay = self.config.llm_retry_base_delay
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                return fn()
            except openai.BadRequestError as exc:
                msg = str(exc).lower()
                if any(k in msg for k in _OVERFLOW_KEYWORDS):
                    raise ContextOverflowError(str(exc)) from exc
                raise
            except openai.APIConnectionError as exc:
                last_exc = exc
                if attempt == max_retries - 1:
                    raise ConnectionError(
                        friendly_connection_error(exc, self.config.api_base_url)
                    ) from exc
                delay = base_delay * (2 ** attempt) + random.random() * 0.5
                log.warning(
                    "LLM connection error %s; retry in %.1fs",
                    type(exc).__name__,
                    delay,
                )
                time.sleep(delay)
            except (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.InternalServerError,
            ) as exc:
                last_exc = exc
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt) + random.random() * 0.5
                log.warning(
                    "LLM transient error %s; retry in %.1fs",
                    type(exc).__name__,
                    delay,
                )
                time.sleep(delay)
            except openai.APIStatusError as exc:
                last_exc = exc
                status = getattr(exc, "status_code", 0) or 0
                if 500 <= status < 600 and attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLM retry loop exited without result")


# -----------------------------------------------------------------------------
# §11 Compaction -- LLM summary + memory flush + session rewrite
# -----------------------------------------------------------------------------


_COMPACTION_HEADER = "## [Compacted conversation history]"
_COMPACTION_FOOTER = "## [End compacted history -- resume below]"


def _format_summary_block(messages_to_summarize: list[Message]) -> str:
    return "\n\n".join(
        f"[{m.role}] {(m.content or '')[:1000]}"
        for m in messages_to_summarize
        if (m.content or "").strip() or m.role != "tool"
    )


def _summary_prompt(text_block: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are a conversation summarizer.",
        },
        {
            "role": "user",
            "content": (
                "Summarize concisely, preserving: decisions, key facts, "
                "TODOs, open questions, and tool outputs future turns will "
                "need. Keep it dense.\n\n" + text_block
            ),
        },
    ]


def _compaction_message(summary_text: str) -> Message:
    body = f"{_COMPACTION_HEADER}\n\n{summary_text}\n\n{_COMPACTION_FOOTER}"
    return Message(role="user", content=body, timestamp=time.time())


def compact_history(
    messages: list[Message],
    llm: LLMClient,
    memory: MemoryStore,
    keep_fresh: int,
    session_store: SessionStore,
    session_id: str,
) -> tuple[list[Message], int]:
    if len(messages) < 4:
        return messages, 0

    try:
        memory.flush_conversation(messages, llm)
    except Exception:  # noqa: BLE001
        log.exception("memory flush during compaction failed")

    keep_last = min(max(1, keep_fresh), len(messages))
    summarize_end = len(messages) - keep_last
    if summarize_end <= 0:
        return messages, 0

    to_summarize = messages[:summarize_end]
    keep = messages[summarize_end:]

    text_block = _format_summary_block(to_summarize)
    try:
        summary = llm.complete_plain(_summary_prompt(text_block)).strip()
    except Exception:  # noqa: BLE001
        log.exception("compaction summary failed; falling back to head+tail trim")
        new_messages = [messages[0]] + keep
        try:
            session_store.rewrite(session_id, new_messages)
        except Exception:  # noqa: BLE001
            log.exception("session rewrite failed during fallback compaction")
        return new_messages, summarize_end

    if not summary:
        summary = "(no summary produced)"

    compaction_msg = _compaction_message(summary)
    new_messages = [compaction_msg] + keep
    try:
        session_store.rewrite(session_id, new_messages)
    except Exception:  # noqa: BLE001
        log.exception("session rewrite failed during compaction")
    return new_messages, summarize_end


# -----------------------------------------------------------------------------
# §15 System prompt builder
# -----------------------------------------------------------------------------


SYSTEM_PROMPT_TEMPLATE = """\
You are ember, a minimalist learning-oriented coding agent.

# Identity
- Operate entirely within the workspace: {workspace}
- Use tools to inspect and modify files. Never assume file contents.
- When unsure, read the file before editing it.
- After learning durable info (decisions, preferences, plans), use `memory` action=write to persist under `memory/{today}.md`.

# Runtime
- OS: {os} ({arch})
- Shell: {shell}
- Model: {model}
- Workspace: {workspace}
- Session: {session_id}
- Date: {today}

# Memory Recall
Before answering about prior work, decisions, or personal context:
1. Call `memory` action=search to find relevant snippets.
2. Read full source lines via `file` action=read when you need context.
3. Cite as `Source: path#Lstart-Lend`.

NOTE: MEMORY.md contents are sent to the LLM on every turn. Do not store secrets there.

# MEMORY.md
{memory_md_section}

# Workspace Instructions (AGENTS.md)
{agents_md_section}

# Tools
Prefer `grep`/`file list` before `exec`. They are faster and confined.

{tool_list}

# Safety
- Tool calls modifying state (edit, file write/append, exec, memory write) require user confirmation.
- Keep tool outputs short. If a read would exceed 400KB, use offset/limit.
- If you hit a dead end, ask the user via `clarify` rather than spinning.
"""


def _runtime_info() -> dict[str, str]:
    return {
        "os": platform.system() or "unknown",
        "arch": platform.machine() or "unknown",
        "shell": os.environ.get("SHELL", "") or "unknown",
        "today": datetime.date.today().isoformat(),
    }


def _format_tool_list(tools: dict[str, Tool]) -> str:
    if not tools:
        return "(no tools registered)"
    lines = [f"- {name} -- {tools[name].description}" for name in sorted(tools)]
    return "\n".join(lines)


def build_system_prompt(
    config: Config,
    session_id: str,
    memory_store: MemoryStore,
) -> str:
    info = _runtime_info()
    memory_md = memory_store.read_memory_md()
    agents_md = memory_store.read_agents_md()
    return SYSTEM_PROMPT_TEMPLATE.format(
        workspace=config.workspace,
        os=info["os"],
        arch=info["arch"],
        shell=info["shell"],
        model=config.model,
        session_id=session_id or "(no session)",
        today=info["today"],
        memory_md_section=memory_md.strip() or "(empty)",
        agents_md_section=agents_md.strip() or "(empty)",
        tool_list=_format_tool_list(TOOLS),
    )


# -----------------------------------------------------------------------------
# §14 Tool implementations
# -----------------------------------------------------------------------------

_IGNORE_DIRS = {
    ".git",
    ".sessions",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".idea",
    ".vscode",
}
_IGNORE_FILES = {
    ".ember.db",
    ".ember.db-wal",
    ".ember.db-shm",
    ".DS_Store",
}

_FILE_LIST_CAP = 500


def _should_ignore_dir(name: str) -> bool:
    return name in _IGNORE_DIRS


def _should_ignore_file(name: str) -> bool:
    return name in _IGNORE_FILES


# --- file tool ---------------------------------------------------------------


def _file_read(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = args.get("path")
    if not path:
        return ToolResult("file read requires 'path'", is_error=True)
    offset = int(args.get("offset", 0) or 0)
    limit_raw = args.get("limit")
    limit = int(limit_raw) if limit_raw is not None else None

    try:
        target = ctx.confiner.resolve(path)
    except (PermissionError, FileNotFoundError) as exc:
        return ToolResult(f"file read error: {exc}", is_error=True)
    if not target.exists():
        return ToolResult(f"file not found: {path}", is_error=True)
    if not target.is_file():
        return ToolResult(f"not a file: {path}", is_error=True)

    size = target.stat().st_size
    if size > MAX_FILE_READ_BYTES and limit is None:
        limit = 1000
    try:
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as exc:
        return ToolResult(f"file read failed: {exc}", is_error=True)

    total = len(lines)
    start = max(0, offset)
    end = total if limit is None else min(total, start + limit)
    sliced = lines[start:end]

    rendered: list[str] = []
    bytes_used = 0
    truncated = False
    for i, ln in enumerate(sliced, start=start + 1):
        formatted = f"{i:6d}  {ln.rstrip(chr(10))}"
        bytes_used += len(formatted) + 1
        if bytes_used > MAX_FILE_READ_BYTES:
            truncated = True
            break
        rendered.append(formatted)

    header = f"[file:read {path} lines {start + 1}-{start + len(rendered)} / {total}]"
    body = "\n".join(rendered) if rendered else "(empty selection)"
    suffix = ""
    if end < total:
        suffix = f"\n[...{total - end} more lines; use offset/limit to continue]"
    if truncated:
        suffix += "\n[...output truncated at MAX_FILE_READ_BYTES]"
    return ToolResult(f"{header}\n{body}{suffix}")


def _resolve_for_write(ctx: ToolContext, path: str) -> Path:
    """Resolve path for write/append, ensuring containment BEFORE any mkdir."""
    candidate = Path(path)
    if candidate.is_absolute():
        resolved = candidate.resolve(strict=False)
    else:
        resolved = (ctx.confiner.root / candidate).resolve(strict=False)
    try:
        resolved.relative_to(ctx.confiner.root)
    except ValueError as exc:
        raise PermissionError(
            f"Path escape blocked: {path} -> {resolved}, "
            f"outside workspace {ctx.confiner.root}"
        ) from exc
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _file_write(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = args.get("path")
    content = args.get("content", "")
    if not path:
        return ToolResult("file write requires 'path'", is_error=True)
    if not isinstance(content, str):
        return ToolResult("file write 'content' must be a string", is_error=True)
    try:
        target = _resolve_for_write(ctx, path)
    except PermissionError as exc:
        return ToolResult(f"file write error: {exc}", is_error=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    except OSError as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        return ToolResult(f"file write failed: {exc}", is_error=True)
    return ToolResult(f"[file:write] wrote {len(content)} bytes to {path}")


def _file_append(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = args.get("path")
    content = args.get("content", "")
    if not path:
        return ToolResult("file append requires 'path'", is_error=True)
    if not isinstance(content, str):
        return ToolResult("file append 'content' must be a string", is_error=True)
    try:
        target = _resolve_for_write(ctx, path)
    except PermissionError as exc:
        return ToolResult(f"file append error: {exc}", is_error=True)
    try:
        with open(target, "a", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
    except OSError as exc:
        return ToolResult(f"file append failed: {exc}", is_error=True)
    return ToolResult(f"[file:append] appended {len(content)} bytes to {path}")


def _file_list(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = args.get("path", ".")
    try:
        target = ctx.confiner.resolve(path)
    except (PermissionError, FileNotFoundError) as exc:
        return ToolResult(f"file list error: {exc}", is_error=True)
    if not target.exists():
        return ToolResult(f"path not found: {path}", is_error=True)

    entries: list[str] = []
    truncated = False
    if target.is_file():
        return ToolResult(str(ctx.confiner.relpath(target)))

    for root, dirs, files in os.walk(target):
        dirs[:] = sorted([d for d in dirs if not _should_ignore_dir(d)])
        files = sorted([f for f in files if not _should_ignore_file(f)])
        for name in files:
            full = Path(root) / name
            rel = ctx.confiner.relpath(full)
            entries.append(rel)
            if len(entries) >= _FILE_LIST_CAP:
                truncated = True
                break
        if truncated:
            break

    header = f"[file:list {path}] {len(entries)} entries"
    if truncated:
        header += f" (capped at {_FILE_LIST_CAP})"
    body = "\n".join(entries) if entries else "(empty)"
    return ToolResult(f"{header}\n{body}")


def _file_preview(args: dict[str, Any], ctx: ToolContext) -> str:
    action = args.get("action", "?")
    path = args.get("path", "")
    content = args.get("content", "") or ""
    preview = content[:500]
    if len(content) > 500:
        preview += f"\n[...+{len(content) - 500} more chars]"
    return f"[file:{action}] path={path}\n{preview}"


@tool(
    name="file",
    description=(
        "Read, write, append or list files within the workspace. "
        "Actions: read (offset/limit), write (atomic), append, list (recursive)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read", "write", "append", "list"],
            },
            "path": {"type": "string"},
            "content": {"type": "string"},
            "offset": {"type": "integer"},
            "limit": {"type": "integer"},
        },
        "required": ["action"],
    },
    requires_confirmation=False,
    confirmation_preview=_file_preview,
)
def file_tool(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    action = (args.get("action") or "").lower()
    if action == "read":
        return _file_read(args, ctx)
    if action == "write":
        return _file_write(args, ctx)
    if action == "append":
        return _file_append(args, ctx)
    if action == "list":
        return _file_list(args, ctx)
    return ToolResult(
        f"file: unknown action {action!r}. Use read/write/append/list.",
        is_error=True,
    )


# --- edit tool ---------------------------------------------------------------


def _edit_preview(args: dict[str, Any], ctx: ToolContext) -> str:
    path = args.get("path", "")
    old_text = args.get("old_text", "") or ""
    new_text = args.get("new_text", "") or ""
    count = int(args.get("count", 0) or 0)
    try:
        target = ctx.confiner.resolve(path)
        old = target.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return f"[edit] cannot preview {path}: {exc}"
    occurrences = old.count(old_text) if old_text else 0
    if occurrences == 0:
        return f"[edit] {path}: old_text not found"
    if count == 0:
        new = old.replace(old_text, new_text)
    else:
        new = old.replace(old_text, new_text, count)
    diff = format_diff_preview(old, new, str(path))
    header = f"[edit] {path} ({occurrences} occurrence(s); replacing {count or 'all'})"
    return f"{header}\n{diff}"


@tool(
    name="edit",
    description=(
        "Replace old_text with new_text in a file. "
        "Fails if old_text not found or ambiguous (unless count specifies how many)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_text": {"type": "string"},
            "new_text": {"type": "string"},
            "count": {
                "type": "integer",
                "description": "Max replacements (0 or omitted = all).",
            },
        },
        "required": ["path", "old_text", "new_text"],
    },
    requires_confirmation=True,
    confirmation_preview=_edit_preview,
)
def edit_tool(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = args.get("path")
    old_text = args.get("old_text")
    new_text = args.get("new_text")
    count = int(args.get("count", 0) or 0)
    if not path or old_text is None or new_text is None:
        return ToolResult(
            "edit requires 'path', 'old_text', 'new_text'",
            is_error=True,
        )
    if old_text == "":
        return ToolResult("edit 'old_text' cannot be empty", is_error=True)
    try:
        target = ctx.confiner.resolve(path)
    except (PermissionError, FileNotFoundError) as exc:
        return ToolResult(f"edit error: {exc}", is_error=True)
    if not target.exists() or not target.is_file():
        return ToolResult(f"edit: file not found: {path}", is_error=True)

    try:
        old = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return ToolResult(f"edit read failed: {exc}", is_error=True)

    occurrences = old.count(old_text)
    if occurrences == 0:
        return ToolResult(
            f"edit: old_text not found in {path}",
            is_error=True,
        )
    if count == 0 and occurrences > 1:
        return ToolResult(
            f"edit: old_text appears {occurrences} times in {path}; "
            "specify 'count' or provide a more unique snippet",
            is_error=True,
        )
    if count != 0 and occurrences > count:
        return ToolResult(
            f"edit: ambiguous -- old_text appears {occurrences} times in "
            f"{path} but count={count}; widen count or use a more unique snippet",
            is_error=True,
        )

    if count == 0:
        new = old.replace(old_text, new_text)
        replaced = occurrences
    else:
        new = old.replace(old_text, new_text, count)
        replaced = min(count, occurrences)

    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(new)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    except OSError as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        return ToolResult(f"edit write failed: {exc}", is_error=True)
    return ToolResult(
        f"[edit] {path}: replaced {replaced} occurrence(s)"
    )


# --- exec tool ---------------------------------------------------------------

BLOCKED_COMMAND_PATTERNS = [
    re.compile(r"\brm\s+-[a-zA-Z]*r[a-zA-Z]*f?\s+/(\s|$|\*)"),
    re.compile(r"\brm\s+-[a-zA-Z]*f[a-zA-Z]*r?\s+/(\s|$|\*)"),
    re.compile(r"\bdd\s+.*\bif=/dev/(zero|random|urandom)"),
    re.compile(r":\(\)\s*\{\s*:\|:&\s*\}\s*;\s*:"),
    re.compile(r"\bmkfs\.[a-z0-9]+\b"),
    re.compile(r">\s*/dev/(sd[a-z]|nvme\d+)"),
    re.compile(r"\b(shutdown|reboot|halt|poweroff)\b"),
    re.compile(r"\bchmod\s+(-R\s+)?777\s+/(\s|$)"),
    re.compile(r"\bchown\s+(-R\s+)?\w+\s+/(\s|$)"),
    re.compile(r"\bcurl\s.+\|\s*sh\b"),
    re.compile(r"\bwget\s.+\|\s*sh\b"),
]


def _check_blocked(command: str) -> Optional[str]:
    for pat in BLOCKED_COMMAND_PATTERNS:
        m = pat.search(command)
        if m:
            return m.group(0)
    return None


def _exec_preview(args: dict[str, Any], ctx: ToolContext) -> str:
    command = args.get("command", "")
    timeout = args.get("timeout_secs", ctx.config.tool_timeout)
    hit = _check_blocked(command)
    banner = f"[exec] {command}\ntimeout: {timeout}s"
    if hit:
        banner += f"\n{colorize(f'WARNING: matches blocked pattern: {hit}', C_RED)}"
    return banner


@tool(
    name="exec",
    description=(
        "Run a shell command inside the workspace. "
        "Blocked patterns (rm -rf /, fork bomb, etc.) are rejected. "
        "Output truncated at EXEC_STDOUT_CAP/EXEC_STDERR_CAP."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout_secs": {"type": "number"},
        },
        "required": ["command"],
    },
    requires_confirmation=True,
    confirmation_preview=_exec_preview,
)
def exec_tool(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    command = args.get("command")
    if not command or not isinstance(command, str):
        return ToolResult("exec requires a non-empty 'command' string", is_error=True)
    timeout = float(args.get("timeout_secs", ctx.config.tool_timeout) or ctx.config.tool_timeout)
    hit = _check_blocked(command)
    if hit:
        return ToolResult(
            f"exec blocked: command matches dangerous pattern {hit!r}",
            is_error=True,
        )

    cancel_flag = getattr(ctx, "cancel_flag", None)
    start = time.time()
    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=str(ctx.confiner.root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        return ToolResult(f"exec spawn failed: {exc}", is_error=True)

    stdout_buf: list[str] = []
    stderr_buf: list[str] = []

    def _drain_stream(stream: Any, buf: list[str]) -> None:
        try:
            for line in iter(stream.readline, ""):
                buf.append(line)
        finally:
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass

    t_out = threading.Thread(
        target=_drain_stream, args=(proc.stdout, stdout_buf), daemon=True
    )
    t_err = threading.Thread(
        target=_drain_stream, args=(proc.stderr, stderr_buf), daemon=True
    )
    t_out.start()
    t_err.start()

    timed_out = False
    cancelled = False
    while True:
        rc = proc.poll()
        if rc is not None:
            break
        elapsed = time.time() - start
        if elapsed > timeout:
            timed_out = True
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
            break
        if cancel_flag is not None and cancel_flag.is_set():
            cancelled = True
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
            break
        time.sleep(0.1)

    t_out.join(timeout=1.0)
    t_err.join(timeout=1.0)
    exit_code = proc.returncode if proc.returncode is not None else -1

    stdout_text = "".join(stdout_buf)
    stderr_text = "".join(stderr_buf)
    if len(stdout_text) > EXEC_STDOUT_CAP:
        stdout_text = (
            stdout_text[:EXEC_STDOUT_CAP]
            + f"\n[...stdout truncated, original {len(stdout_text)} bytes...]"
        )
    if len(stderr_text) > EXEC_STDERR_CAP:
        stderr_text = (
            stderr_text[:EXEC_STDERR_CAP]
            + f"\n[...stderr truncated, original {len(stderr_text)} bytes...]"
        )

    elapsed = time.time() - start
    status = "ok"
    if timed_out:
        status = f"timed out after {timeout:.1f}s"
    elif cancelled:
        status = "cancelled by user"

    body = (
        f"Exit code: {exit_code} ({status}) in {elapsed:.2f}s\n"
        f"--- stdout ---\n{stdout_text}\n"
        f"--- stderr ---\n{stderr_text}"
    )
    is_err = timed_out or cancelled or (exit_code != 0)
    return ToolResult(body, is_error=is_err)


# --- grep tool ---------------------------------------------------------------


def _grep_walk(root: Path) -> list[Path]:
    out: list[Path] = []
    for cur_root, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not _should_ignore_dir(d)]
        for fn in files:
            if _should_ignore_file(fn):
                continue
            out.append(Path(cur_root) / fn)
    return out


@tool(
    name="grep",
    description=(
        "Search file contents for a pattern (regex by default). "
        "Walks recursively from path, honors include glob. "
        f"Capped at {GREP_RESULTS_CAP} hits / {GREP_OUTPUT_CAP} chars."
    ),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "description": "Relative path (default '.')"},
            "is_regex": {"type": "boolean"},
            "case_insensitive": {"type": "boolean"},
            "include": {
                "type": "string",
                "description": "fnmatch glob filter, e.g. '*.py'",
            },
        },
        "required": ["pattern"],
    },
    requires_confirmation=False,
)
def grep_tool(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    pattern = args.get("pattern")
    if not pattern or not isinstance(pattern, str):
        return ToolResult("grep requires 'pattern' string", is_error=True)
    path = args.get("path", ".")
    is_regex = bool(args.get("is_regex", True))
    case_insensitive = bool(args.get("case_insensitive", False))
    include = args.get("include")

    try:
        base = ctx.confiner.resolve(path)
    except (PermissionError, FileNotFoundError) as exc:
        return ToolResult(f"grep error: {exc}", is_error=True)
    if not base.exists():
        return ToolResult(f"grep: path not found: {path}", is_error=True)

    try:
        if is_regex:
            flags = re.IGNORECASE if case_insensitive else 0
            regex = re.compile(pattern, flags)
        else:
            regex = None
    except re.error as exc:
        return ToolResult(f"grep invalid regex: {exc}", is_error=True)

    candidates: list[Path]
    if base.is_file():
        candidates = [base]
    else:
        candidates = _grep_walk(base)
        if include:
            candidates = [p for p in candidates if fnmatch.fnmatch(p.name, include)]

    hits: list[str] = []
    total_chars = 0
    truncated = False
    for fp in candidates:
        try:
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                for lineno, line in enumerate(f, start=1):
                    line_stripped = line.rstrip("\n")
                    matched = False
                    if regex is not None:
                        matched = regex.search(line_stripped) is not None
                    else:
                        needle = pattern
                        hay = line_stripped
                        if case_insensitive:
                            needle = needle.lower()
                            hay = hay.lower()
                        matched = needle in hay
                    if matched:
                        rel = ctx.confiner.relpath(fp)
                        display = f"{rel}:{lineno}:{line_stripped[:400]}"
                        total_chars += len(display) + 1
                        if (
                            len(hits) >= GREP_RESULTS_CAP
                            or total_chars > GREP_OUTPUT_CAP
                        ):
                            truncated = True
                            break
                        hits.append(display)
        except (OSError, UnicodeDecodeError):
            continue
        if truncated:
            break

    header = f"[grep] pattern={pattern!r} {len(hits)} hit(s)"
    if truncated:
        header += " (capped)"
    body = "\n".join(hits) if hits else "(no matches)"
    return ToolResult(f"{header}\n{body}")


# --- memory tool -------------------------------------------------------------


def _memory_preview(args: dict[str, Any], ctx: ToolContext) -> str:
    action = (args.get("action") or "").lower()
    path = args.get("path", "")
    content = args.get("content", "") or ""
    preview = content[:500]
    if len(content) > 500:
        preview += f"\n[...+{len(content) - 500} more chars]"
    return f"[memory:{action}] path={path}\n{preview}"


@tool(
    name="memory",
    description=(
        "Search or append durable memory. "
        "search: FTS5 across MEMORY.md + memory/*.md. "
        "write: append-only to MEMORY.md or memory/<YYYY-MM-DD>.md."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["search", "write"]},
            "query": {"type": "string"},
            "path": {"type": "string"},
            "content": {"type": "string"},
            "max_results": {"type": "integer"},
        },
        "required": ["action"],
    },
    requires_confirmation=False,
    confirmation_preview=_memory_preview,
)
def memory_tool(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    action = (args.get("action") or "").lower()
    if action == "search":
        query = args.get("query", "") or ""
        if not query.strip():
            return ToolResult("memory search requires 'query'", is_error=True)
        max_results = int(args.get("max_results", 10) or 10)
        hits = ctx.memory.search(query, max_results=max_results)
        header = f"[memory:search {query!r}] {len(hits)} hit(s)"
        return ToolResult(f"{header}\n{ctx.memory.format_search_results(hits)}")
    if action == "write":
        path = args.get("path") or f"memory/{datetime.date.today().isoformat()}.md"
        content = args.get("content")
        if not content:
            return ToolResult(
                "memory write requires 'content'",
                is_error=True,
            )
        try:
            ctx.memory.append(path, content)
        except (ValueError, PermissionError) as exc:
            return ToolResult(f"memory write rejected: {exc}", is_error=True)
        return ToolResult(f"[memory:write] appended {len(content)} bytes to {path}")
    return ToolResult(
        f"memory: unknown action {action!r}. Use search or write.",
        is_error=True,
    )


# --- clarify tool ------------------------------------------------------------


@tool(
    name="clarify",
    description=(
        "Ask the user a question when input is needed. "
        "Optional 'choices' lets you list options numerically. "
        "Returns the user's typed answer."
    ),
    parameters={
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "choices": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["question"],
    },
    requires_confirmation=False,
)
def clarify_tool(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
    question = args.get("question", "")
    choices = args.get("choices") or []
    if not question:
        return ToolResult("clarify requires 'question'", is_error=True)
    print(colorize(f"\n[clarify] {question}", C_YELLOW + C_BOLD))
    if isinstance(choices, list) and choices:
        for i, c in enumerate(choices, start=1):
            print(colorize(f"  [{i}] {c}", C_YELLOW))
    try:
        reply = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ToolResult("[user cancelled]")
    if not reply:
        return ToolResult("[user provided no answer]")
    return ToolResult(f"user_answer: {reply}")


# -----------------------------------------------------------------------------
# §16 Agent loop -- dispatch, truncation, compaction, observability
# -----------------------------------------------------------------------------


MODEL_PRICING: dict[str, tuple[float, float]] = {
    # (per-1k prompt USD, per-1k completion USD)
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4.1": (0.002, 0.008),
    "gpt-4.1-mini": (0.0004, 0.0016),
    "claude-3-5-sonnet": (0.003, 0.015),
    "claude-3-5-haiku": (0.0008, 0.004),
    "claude-3-opus": (0.015, 0.075),
    "claude-sonnet-4-6": (0.003, 0.015),
    "claude-opus-4-7": (0.015, 0.075),
}


def _model_price(model: str) -> tuple[float, float]:
    key = model.lower()
    if key in MODEL_PRICING:
        return MODEL_PRICING[key]
    for known, price in MODEL_PRICING.items():
        if known in key:
            return price
    return (0.0, 0.0)


_DYNAMIC_CONFIRM = {
    ("file", "write"),
    ("file", "append"),
    ("memory", "write"),
}


def _needs_confirmation(tool_def: Tool, args: dict[str, Any]) -> bool:
    if tool_def.requires_confirmation:
        return True
    key = (tool_def.name, (args.get("action") or "").lower())
    return key in _DYNAMIC_CONFIRM


def _truncate_args_for_display(arguments_json: str, limit: int = 200) -> str:
    if not arguments_json:
        return ""
    if len(arguments_json) <= limit:
        return arguments_json
    return arguments_json[:limit] + f"...(+{len(arguments_json) - limit})"


class Agent:
    def __init__(
        self,
        config: Config,
        confiner: Confiner,
        memory: MemoryStore,
        session_store: SessionStore,
        llm: LLMClient,
        registry: ToolRegistry,
    ) -> None:
        self.config = config
        self.confiner = confiner
        self.memory = memory
        self.session_store = session_store
        self.llm = llm
        self.registry = registry
        self.messages: list[Message] = []
        self.session_id: str = ""
        self.compaction_count: int = 0
        self.cancel_flag: threading.Event = threading.Event()
        # share the same event with tool context
        registry.ctx.cancel_flag = self.cancel_flag

    # --- session lifecycle --------------------------------------------------

    def start_session(self, session_id: Optional[str] = None) -> str:
        if session_id:
            loaded = self.session_store.load(session_id)
            self.session_id = session_id
            self.messages = loaded
            log.info(
                "resumed session %s with %d messages",
                session_id,
                len(loaded),
            )
        else:
            self.session_id = self.session_store.new_session_id()
            self.messages = []
            log.info("started new session %s", self.session_id)
        return self.session_id

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def token_estimate(self) -> int:
        return estimate_messages_tokens(self.messages)

    # --- cancellation -------------------------------------------------------

    def cancel(self) -> None:
        self.cancel_flag.set()

    # --- tool dispatch ------------------------------------------------------

    def _announce_tool_call(self, tc: ToolCall) -> None:
        args_display = _truncate_args_for_display(tc.arguments)
        line = f"> {tc.name}({args_display})"
        print(colorize(line, C_CYAN))

    def _dispatch_tool_call(self, tc: ToolCall) -> ToolResult:
        self._announce_tool_call(tc)
        tool_def = TOOLS.get(tc.name)
        if tool_def is None:
            return self.registry.dispatch(tc.name, tc.arguments)

        try:
            parsed_args = (
                json.loads(tc.arguments) if tc.arguments.strip() else {}
            )
        except json.JSONDecodeError as exc:
            return ToolResult(f"Invalid JSON arguments: {exc}", is_error=True)
        if not isinstance(parsed_args, dict):
            return ToolResult(
                "Tool arguments must be a JSON object.",
                is_error=True,
            )

        if _needs_confirmation(tool_def, parsed_args):
            preview: str
            try:
                if tool_def.confirmation_preview:
                    preview = tool_def.confirmation_preview(
                        parsed_args, self.registry.ctx
                    )
                else:
                    preview = f"{tc.name}({_truncate_args_for_display(tc.arguments, 500)})"
            except Exception as exc:  # noqa: BLE001
                preview = f"{tc.name}({_truncate_args_for_display(tc.arguments, 500)})\n[preview error: {exc}]"
            if not confirm(tc.name, preview, self.config.yolo, log):
                print(colorize("  -> denied by user", C_YELLOW))
                return ToolResult("User denied tool execution.", is_error=True)

        t0 = time.time()
        result = self.registry.dispatch(tc.name, tc.arguments)
        elapsed = time.time() - t0
        marker = colorize(f"  [{tc.name}: {elapsed:.2f}s]", C_DIM)
        print(marker)
        return result

    # --- compaction ---------------------------------------------------------

    def compact(self) -> int:
        before = len(self.messages)
        new_msgs, summarized = compact_history(
            self.messages,
            self.llm,
            self.memory,
            keep_fresh=self.config.compact_keep_fresh,
            session_store=self.session_store,
            session_id=self.session_id,
        )
        if summarized > 0:
            self.messages = new_msgs
            self.compaction_count += 1
            log.info(
                "compacted: %d -> %d messages (summarized %d)",
                before,
                len(new_msgs),
                summarized,
            )
        return summarized

    # --- cost display -------------------------------------------------------

    def _format_token_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        elapsed: float,
    ) -> str:
        p_rate, c_rate = _model_price(self.config.model)
        cost = (prompt_tokens / 1000.0) * p_rate + (
            completion_tokens / 1000.0
        ) * c_rate
        return (
            f"[turn done * {elapsed:.1f}s * in {prompt_tokens} out "
            f"{completion_tokens} tokens * ${cost:.4f}]"
        )

    # --- main loop ----------------------------------------------------------

    def run_turn(self, user_input: str) -> str:
        text = (user_input or "").strip()
        if not text:
            return ""

        self.cancel_flag.clear()
        if not self.session_id:
            self.start_session()

        user_msg = Message(role="user", content=text, timestamp=time.time())
        self.messages.append(user_msg)
        try:
            self.session_store.append(self.session_id, user_msg)
        except Exception:  # noqa: BLE001
            log.exception("failed to append user message to session log")

        soft_limit = self.config.context_window * self.config.compact_threshold
        if estimate_messages_tokens(self.messages) > soft_limit:
            log.info("soft compaction threshold reached; compacting")
            self.compact()

        step = 0
        prompt_tokens_total = 0
        completion_tokens_total = 0
        turn_start = time.time()

        while True:
            if self.cancel_flag.is_set():
                print(colorize("\n[cancelled]", C_YELLOW))
                return "[cancelled]"

            step += 1
            if step > self.config.max_steps:
                msg = f"[max_steps={self.config.max_steps} reached]"
                print(colorize(msg, C_YELLOW))
                return msg

            truncate_oversized_tool_results(
                self.messages, self.config.context_window
            )
            repair_orphaned_tool_calls(self.messages)
            repair_orphaned_tool_results(self.messages)

            sys_prompt = build_system_prompt(
                self.config, self.session_id, self.memory
            )
            api_msgs: list[dict[str, Any]] = [
                {"role": "system", "content": sys_prompt}
            ]
            for m in self.messages:
                api_msgs.append(m.to_openai())

            try:
                assistant_msg = self.llm.complete(
                    api_msgs,
                    tools=self.registry.openai_definitions(),
                    stream=True,
                )
            except ContextOverflowError:
                log.warning(
                    "context overflow mid-turn; running emergency compaction"
                )
                summarized = self.compact()
                if summarized == 0:
                    return colorize(
                        "[cannot recover from context overflow]",
                        C_RED,
                    )
                continue
            except ConnectionError as exc:
                err = str(exc)
                print(colorize(f"\n{err}", C_RED))
                return f"[llm connection error: {err}]"

            usage = self.llm.last_usage
            if usage is not None:
                prompt_tokens_total += int(
                    getattr(usage, "prompt_tokens", 0) or 0
                )
                completion_tokens_total += int(
                    getattr(usage, "completion_tokens", 0) or 0
                )

            if not assistant_msg.content and not assistant_msg.tool_calls:
                log.warning("empty response from model")
                self.messages.append(assistant_msg)
                try:
                    self.session_store.append(self.session_id, assistant_msg)
                except Exception:  # noqa: BLE001
                    log.exception("session append failed")
                return "[empty response from model]"

            self.messages.append(assistant_msg)
            try:
                self.session_store.append(self.session_id, assistant_msg)
            except Exception:  # noqa: BLE001
                log.exception("session append failed")

            if assistant_msg.tool_calls:
                for tc in assistant_msg.tool_calls:
                    if self.cancel_flag.is_set():
                        print(colorize("\n[cancelled]", C_YELLOW))
                        return "[cancelled]"
                    result = self._dispatch_tool_call(tc)
                    tool_msg = Message(
                        role="tool",
                        content=result.content,
                        tool_call_id=tc.id,
                        timestamp=time.time(),
                    )
                    self.messages.append(tool_msg)
                    try:
                        self.session_store.append(self.session_id, tool_msg)
                    except Exception:  # noqa: BLE001
                        log.exception("session append failed")
                continue

            elapsed = time.time() - turn_start
            summary_line = self._format_token_cost(
                prompt_tokens_total, completion_tokens_total, elapsed
            )
            print(colorize(summary_line, C_DIM))
            return assistant_msg.content or ""


# -----------------------------------------------------------------------------
# §17 REPL -- slash commands, signals, banner, streaming colors
# -----------------------------------------------------------------------------


_BANNER_ART = r"""
                  _
  ___ _ __ ___   | |__   ___ _ __
 / _ \ '_ ` _ \  | '_ \ / _ \ '__|
|  __/ | | | | | | |_) |  __/ |
 \___|_| |_| |_| |_.__/ \___|_|
"""


def _try_import_readline() -> None:
    try:
        import readline  # noqa: F401
    except Exception:  # noqa: BLE001
        pass


def _short_sid(session_id: str) -> str:
    if not session_id:
        return "no-session"
    return session_id[-13:] if len(session_id) > 13 else session_id


def _format_prompt(model: str, session_id: str) -> str:
    return f"ember ({model}) [{_short_sid(session_id)}] > "


def _print_banner(config: Config, agent: "Agent", tool_count: int) -> None:
    art = colorize(_BANNER_ART, C_CYAN)
    yolo_badge = (
        colorize(" [YOLO]", C_RED + C_BOLD) if config.yolo else ""
    )
    print(art)
    print(colorize(f"  model      : {config.model}", C_BOLD) + yolo_badge)
    print(f"  workspace  : {config.workspace}")
    print(f"  session    : {agent.session_id or '(none)'}")
    print(f"  base_url   : {config.api_base_url}")
    print(f"  tools      : {tool_count} ({', '.join(sorted(TOOLS))})")
    print(f"  context    : {config.context_window} tokens "
          f"(compact at {int(config.compact_threshold*100)}%)")
    print(colorize("  Tip: type /help for slash commands.", C_DIM))
    print(colorize("  Tip: one instance per workspace.", C_DIM))
    print()


def _pick_session(session_store: SessionStore) -> Optional[str]:
    recent = session_store.list_recent(n=10)
    if not recent:
        return None
    print(colorize("-- Recent sessions --", C_BOLD))
    for i, (sid, mtime, count) in enumerate(recent):
        ts = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"  [{i}] {sid}  msgs={count}  {ts}")
    print("  [N] new session (default)")
    try:
        reply = input("Select: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    if reply in ("", "n", "new"):
        return None
    try:
        idx = int(reply)
    except ValueError:
        return None
    if 0 <= idx < len(recent):
        return recent[idx][0]
    return None


# --- slash command handlers --------------------------------------------------


SLASH_COMMANDS: dict[str, str] = {
    "/help": "Print this list of slash commands.",
    "/quit": "Exit the REPL cleanly.",
    "/reset": "Confirm, then clear messages and start a fresh session.",
    "/memory": "Print the first 200 lines of MEMORY.md.",
    "/compact": "Force immediate compaction.",
    "/session": "Print id, message count, est tokens, compaction count.",
    "/history": "Print the last 20 messages (role + first 100 chars).",
    "/sessions": "List recent sessions and switch.",
    "/tools": "Print registered tools with descriptions.",
    "/model": "Switch the model mid-session (usage: /model <name>).",
    "/export": "Dump current session as markdown (usage: /export [path]).",
}


def _cmd_help(args: str, agent: "Agent", session_store: SessionStore, config: Config) -> bool:
    print(colorize("-- Slash commands --", C_BOLD))
    for name in sorted(SLASH_COMMANDS):
        print(f"  {name:<11} {SLASH_COMMANDS[name]}")
    return True


def _cmd_quit(args: str, agent: "Agent", session_store: SessionStore, config: Config) -> bool:
    print(colorize("  goodbye.", C_DIM))
    raise SystemExit(0)


def _cmd_reset(args: str, agent: "Agent", session_store: SessionStore, config: Config) -> bool:
    if not confirm(
        "reset",
        "Clear current messages and start a fresh session (existing session log preserved on disk).",
        config.yolo,
        log,
    ):
        print(colorize("  reset cancelled.", C_YELLOW))
        return True
    old_sid = agent.session_id
    agent.messages = []
    agent.compaction_count = 0
    agent.start_session()
    print(colorize(
        f"  reset: {old_sid} -> {agent.session_id}",
        C_GREEN,
    ))
    return True


def _cmd_memory(args: str, agent: "Agent", session_store: SessionStore, config: Config) -> bool:
    text = agent.memory.read_memory_md(max_chars=60_000)
    if not text.strip():
        print(colorize("  (MEMORY.md is empty)", C_DIM))
        return True
    lines = text.splitlines()[:200]
    print(colorize(f"-- MEMORY.md ({len(lines)} lines shown) --", C_BOLD))
    print("\n".join(lines))
    return True


def _cmd_compact(args: str, agent: "Agent", session_store: SessionStore, config: Config) -> bool:
    summarized = agent.compact()
    if summarized:
        print(colorize(
            f"  compacted: summarized {summarized} messages, kept {len(agent.messages)}",
            C_GREEN,
        ))
    else:
        print(colorize("  nothing to compact.", C_DIM))
    return True


def _cmd_session(args: str, agent: "Agent", session_store: SessionStore, config: Config) -> bool:
    print(colorize("-- Session --", C_BOLD))
    print(f"  id            : {agent.session_id}")
    print(f"  messages      : {agent.message_count}")
    print(f"  est tokens    : {agent.token_estimate}")
    print(f"  compactions   : {agent.compaction_count}")
    print(f"  model         : {config.model}")
    print(f"  workspace     : {config.workspace}")
    return True


def _cmd_history(args: str, agent: "Agent", session_store: SessionStore, config: Config) -> bool:
    tail = agent.messages[-20:]
    if not tail:
        print(colorize("  (no messages yet)", C_DIM))
        return True
    print(colorize(f"-- last {len(tail)} messages --", C_BOLD))
    for m in tail:
        content = (m.content or "").replace("\n", " ")[:100]
        marker = ""
        if m.tool_calls:
            marker = f" [+{len(m.tool_calls)} tool_calls]"
        print(f"  [{m.role:>9}] {content}{marker}")
    return True


def _cmd_sessions(args: str, agent: "Agent", session_store: SessionStore, config: Config) -> bool:
    sid = _pick_session(session_store)
    if sid is None:
        print(colorize("  stayed on current session.", C_DIM))
        return True
    if sid == agent.session_id:
        print(colorize("  already on that session.", C_DIM))
        return True
    agent.start_session(sid)
    print(colorize(f"  switched to {sid}", C_GREEN))
    return True


def _cmd_tools(args: str, agent: "Agent", session_store: SessionStore, config: Config) -> bool:
    print(colorize(f"-- {len(TOOLS)} tools --", C_BOLD))
    for name in sorted(TOOLS):
        t = TOOLS[name]
        confirm_str = " (confirm)" if t.requires_confirmation else ""
        print(f"  {name}{confirm_str}: {t.description}")
    return True


def _cmd_model(args: str, agent: "Agent", session_store: SessionStore, config: Config) -> bool:
    new_model = args.strip()
    if not new_model:
        print(f"  current model: {config.model}")
        return True
    old = config.model
    config.model = new_model
    print(colorize(f"  model: {old} -> {new_model}", C_GREEN))
    return True


def _cmd_export(args: str, agent: "Agent", session_store: SessionStore, config: Config) -> bool:
    target_arg = args.strip() or f"session_{agent.session_id}.md"
    try:
        target = agent.confiner.resolve(target_arg)
    except (PermissionError, FileNotFoundError) as exc:
        print(colorize(f"  export failed: {exc}", C_RED))
        return True
    lines: list[str] = [f"# Session {agent.session_id}\n"]
    for m in agent.messages:
        lines.append(f"\n## {m.role}\n")
        if m.tool_calls:
            for tc in m.tool_calls:
                lines.append(f"- tool_call: **{tc.name}** `{tc.arguments}`\n")
        if m.content:
            lines.append(m.content)
            lines.append("\n")
    try:
        with open(target, "w", encoding="utf-8") as f:
            f.write("".join(lines))
    except OSError as exc:
        print(colorize(f"  export failed: {exc}", C_RED))
        return True
    print(colorize(f"  exported to {target_arg}", C_GREEN))
    return True


_SLASH_HANDLERS: dict[str, Callable[..., bool]] = {
    "/help": _cmd_help,
    "/quit": _cmd_quit,
    "/exit": _cmd_quit,
    "/reset": _cmd_reset,
    "/memory": _cmd_memory,
    "/compact": _cmd_compact,
    "/session": _cmd_session,
    "/history": _cmd_history,
    "/sessions": _cmd_sessions,
    "/tools": _cmd_tools,
    "/model": _cmd_model,
    "/export": _cmd_export,
}


def _handle_slash_command(
    line: str,
    agent: "Agent",
    session_store: SessionStore,
    config: Config,
) -> bool:
    parts = line.split(None, 1)
    cmd = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""
    handler = _SLASH_HANDLERS.get(cmd)
    if handler is None:
        print(colorize(f"  unknown command: {cmd}. Try /help.", C_YELLOW))
        return True
    return handler(rest, agent, session_store, config)


# --- signals + main loop -----------------------------------------------------


_REPL_STATE = {"agent": None, "last_interrupt": 0.0}


def _sigint_handler(signum: int, frame: Any) -> None:
    now = time.time()
    agent = _REPL_STATE.get("agent")
    if agent is not None:
        agent.cancel()
    last = _REPL_STATE.get("last_interrupt") or 0.0
    if now - last < 2.0:
        print(colorize("\n  double Ctrl-C -- exiting.", C_YELLOW))
        sys.exit(130)
    _REPL_STATE["last_interrupt"] = now
    print(colorize("\n  [interrupted] press Ctrl-C again within 2s to exit.", C_YELLOW))


def _sigterm_handler(signum: int, frame: Any) -> None:
    log.info("SIGTERM received; exiting")
    sys.exit(0)


def repl(agent: "Agent", session_store: SessionStore, config: Config) -> None:
    _REPL_STATE["agent"] = agent
    _try_import_readline()
    try:
        signal.signal(signal.SIGINT, _sigint_handler)
    except (ValueError, OSError):
        pass
    try:
        signal.signal(signal.SIGTERM, _sigterm_handler)
    except (ValueError, OSError):
        pass

    while True:
        prompt = _format_prompt(config.model, agent.session_id)
        try:
            line = input(prompt)
        except EOFError:
            print()
            print(colorize("  goodbye.", C_DIM))
            return
        except KeyboardInterrupt:
            # handled by signal handler which sets last_interrupt; loop
            continue

        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("/"):
            try:
                _handle_slash_command(stripped, agent, session_store, config)
            except SystemExit:
                raise
            except Exception:  # noqa: BLE001
                log.exception("slash command error")
            continue

        try:
            agent.run_turn(stripped)
        except ConnectionError as exc:
            print(colorize(f"  connection error: {exc}", C_RED))
        except Exception:  # noqa: BLE001
            log.exception("agent turn crashed")


# -----------------------------------------------------------------------------
# §18 CLI entry + bootstrap
# -----------------------------------------------------------------------------


_MEMORY_MD_TEMPLATE = """\
# MEMORY.md

This file is the agent's durable, hand-editable long-term memory.
It is injected into the system prompt on every turn, so keep it concise.

## How to use
- Add short bullet entries that future sessions should know about.
- Use `## <date>` sections for entries the agent appends via the memory tool.
- Do NOT store secrets, API keys, or other credentials here.

## Examples
- Preferred stack: Python 3.11+, stdlib where possible.
- Coding style: small, explicit, no hidden globals.
- Open question: how much logging is too much?
"""

_AGENTS_MD_TEMPLATE = """\
# AGENTS.md

Workspace-wide instructions injected into the system prompt each turn.
Use it to describe the project goals, conventions, and any ground rules
the agent must follow.

## Examples
- This workspace is a learning sandbox -- prefer verbose, readable code.
- Always run tests after editing core modules.
- Do not touch files under `third_party/` without asking.
"""


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ember",
        description="ember -- minimal single-module agent loop.",
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        default=None,
        help="Auto-approve every confirmation (dangerous).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the LLM model name.",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Override the workspace directory.",
    )
    parser.add_argument(
        "--api-base-url",
        default=None,
        help="Override the OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Override the API key.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable DEBUG logging.",
    )
    return parser.parse_args(argv)


def apply_cli_overrides(config: Config, args: argparse.Namespace) -> None:
    if args.model:
        config.model = args.model
    if args.workspace:
        config.workspace = args.workspace
    if args.api_base_url:
        config.api_base_url = args.api_base_url
    if args.api_key:
        config.api_key = args.api_key
    if args.yolo:
        config.yolo = True


def _ensure_starter_files(workspace: str | os.PathLike[str]) -> None:
    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=True)
    mem_md = root / "MEMORY.md"
    agents_md = root / "AGENTS.md"
    if not mem_md.exists():
        mem_md.write_text(_MEMORY_MD_TEMPLATE, encoding="utf-8")
    if not agents_md.exists():
        agents_md.write_text(_AGENTS_MD_TEMPLATE, encoding="utf-8")
    (root / "memory").mkdir(parents=True, exist_ok=True)


def bootstrap(config: Config) -> tuple[Agent, SessionStore]:
    _ensure_starter_files(config.workspace)
    confiner = Confiner(config.workspace)
    config.workspace = str(confiner.root)
    memory_store = MemoryStore(str(confiner.root))
    memory_store.sync_all()
    session_store = SessionStore(str(confiner.root))
    llm = LLMClient(config)
    ctx = ToolContext(
        confiner=confiner,
        workspace=str(confiner.root),
        memory=memory_store,
        config=config,
    )
    registry = ToolRegistry(ctx)
    agent = Agent(
        config,
        confiner,
        memory_store,
        session_store,
        llm,
        registry,
    )
    return agent, session_store


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    config = Config.from_env()
    apply_cli_overrides(config, args)
    setup_logging("DEBUG" if args.debug else os.environ.get("EMBER_LOG_LEVEL", "INFO"))

    try:
        agent, session_store = bootstrap(config)
    except openai.APIConnectionError as exc:
        print(colorize(friendly_connection_error(exc, config.api_base_url), C_RED))
        return 1
    except Exception as exc:  # noqa: BLE001
        log.exception("bootstrap failed")
        print(colorize(f"bootstrap error: {exc}", C_RED))
        return 1

    chosen = _pick_session(session_store)
    agent.start_session(chosen)
    _print_banner(config, agent, len(TOOLS))
    try:
        repl(agent, session_store, config)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print(colorize("\n  interrupted; exiting.", C_YELLOW))
    finally:
        try:
            agent.memory.close()
        except Exception:  # noqa: BLE001
            pass
    return 0


# -----------------------------------------------------------------------------
# §19 __main__ guard
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    sys.exit(main())
