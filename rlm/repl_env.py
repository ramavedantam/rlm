"""Sandboxed REPL execution environment with LLM sub-calls and filesystem access."""

from __future__ import annotations

import asyncio
import io
import os
import sys
import threading
import traceback
from dataclasses import dataclass, field

from .client import LLMClient
from .utils import truncate

# Avoid circular import — RLM_REPL is imported lazily in _rlm_query_sync
TYPE_CHECKING = False
if TYPE_CHECKING:
    from .rlm_repl import RLM_REPL


@dataclass
class REPLResult:
    stdout: str
    stderr: str
    execution_time: float = 0.0


class REPLEnv:
    """Sandboxed Python execution environment with llm_query() and file access.

    Code runs via exec() in a curated globals namespace. Variables persist
    across calls within the same REPLEnv instance.

    The async/sync bridge:
    - execute() is async (called from the RLM loop)
    - exec() runs in a thread via asyncio.to_thread
    - llm_query() inside exec uses run_coroutine_threadsafe to call the async client
    """

    def __init__(
        self,
        client: LLMClient,
        sub_model: str | None = None,
        allowed_dirs: list[str] | None = None,
        max_output_len: int = 100_000,
        initial_vars: dict | None = None,
        rlm_factory: callable | None = None,
    ):
        self.client = client
        self.sub_model = sub_model
        self.allowed_dirs = [os.path.realpath(d) for d in (allowed_dirs or [])]
        self.max_output_len = max_output_len
        self._rlm_factory = rlm_factory  # callable that creates a child RLM_REPL

        # Token accounting for sub-LM calls
        self.sub_input_tokens = 0
        self.sub_output_tokens = 0
        self.sub_call_count = 0
        self._token_lock = threading.Lock()

        # The answer, set by FINAL()
        self.final_answer: str | None = None

        # Names that cannot be overwritten by exec'd code
        self._protected_names = {"FINAL", "llm_query", "rlm_query", "read_file", "list_files"}

        # Persistent namespace for exec'd code
        self._globals: dict = self._build_globals()
        self._locals: dict = {}

        # Inject initial variables (e.g., pre-loaded context)
        if initial_vars:
            for k, v in initial_vars.items():
                if k not in self._protected_names:
                    self._locals[k] = v

    # ── Public API ────────────────────────────────────────────────────

    async def execute(self, code: str) -> REPLResult:
        """Execute Python code in the sandbox. Returns captured output."""
        loop = asyncio.get_running_loop()

        def _run():
            return self._exec_sync(code, loop)

        return await asyncio.to_thread(_run)

    # ── Sync execution (runs in thread) ───────────────────────────────

    def _exec_sync(self, code: str, loop: asyncio.AbstractEventLoop) -> REPLResult:
        import time

        # Inject the event loop so llm_query can schedule coroutines
        self._loop = loop

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        start = time.time()

        # Override print to capture output
        self._globals["print"] = lambda *args, **kwargs: print(
            *args, **kwargs, file=stdout_buf
        )

        try:
            # Warn if code tries to redefine protected functions
            for name in self._protected_names:
                if f"def {name}(" in code or f"def {name} (" in code:
                    print(
                        f"WARNING: {name}() is a built-in REPL function and cannot be redefined. "
                        f"Just call {name}(...) directly.",
                        file=stdout_buf,
                    )

            # Split imports from other code so imports go into globals
            lines = code.split("\n")
            import_lines = []
            other_lines = []
            for line in lines:
                stripped = line.lstrip()
                if stripped.startswith(("import ", "from ")) and not stripped.startswith("#"):
                    import_lines.append(line)
                else:
                    other_lines.append(line)

            if import_lines:
                exec("\n".join(import_lines), self._globals, self._globals)

            if other_lines:
                combined = {**self._globals, **self._locals}
                # Ensure injected functions can't be shadowed
                for k in self._protected_names:
                    combined[k] = self._globals[k]
                exec("\n".join(other_lines), combined, combined)
                # Capture new variables into _locals (skip protected names)
                for k, v in combined.items():
                    if k not in self._globals and k not in self._protected_names:
                        self._locals[k] = v

        except Exception:
            stderr_buf.write(traceback.format_exc())

        elapsed = time.time() - start
        return REPLResult(
            stdout=truncate(stdout_buf.getvalue(), self.max_output_len),
            stderr=truncate(stderr_buf.getvalue(), self.max_output_len),
            execution_time=elapsed,
        )

    # ── Globals namespace ─────────────────────────────────────────────

    def _build_globals(self) -> dict:
        import json
        import math
        import re
        import collections
        import statistics

        safe_builtins = {
            "abs": abs, "all": all, "any": any, "bool": bool, "bytes": bytes,
            "callable": callable, "chr": chr, "dict": dict, "dir": dir,
            "divmod": divmod, "enumerate": enumerate, "filter": filter,
            "float": float, "format": format, "frozenset": frozenset,
            "getattr": getattr, "hasattr": hasattr, "hash": hash, "hex": hex,
            "int": int, "isinstance": isinstance, "issubclass": issubclass,
            "iter": iter, "len": len, "list": list, "map": map, "max": max,
            "min": min, "next": next, "oct": oct, "ord": ord, "pow": pow,
            "print": print,  # overridden per-execution
            "range": range, "repr": repr, "reversed": reversed, "round": round,
            "set": set, "slice": slice, "sorted": sorted, "str": str,
            "sum": sum, "tuple": tuple, "type": type, "zip": zip,
            # Exceptions
            "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
            "KeyError": KeyError, "IndexError": IndexError, "AttributeError": AttributeError,
            "FileNotFoundError": FileNotFoundError, "RuntimeError": RuntimeError,
            "StopIteration": StopIteration,
            # Allow imports
            "__import__": __import__,
        }

        return {
            "__builtins__": safe_builtins,
            # Pre-imported modules
            "json": json,
            "re": re,
            "math": math,
            "collections": collections,
            "statistics": statistics,
            # Injected functions
            "llm_query": self._llm_query_sync,
            "rlm_query": self._rlm_query_sync,
            "read_file": self._read_file,
            "list_files": self._list_files,
            "FINAL": self._final,
        }

    # ── Injected functions ────────────────────────────────────────────

    def _llm_query_sync(self, prompt: str, system: str = "") -> str:
        """Synchronous wrapper around the async LLM client.

        Called from inside exec'd code (running in a thread).
        Schedules the async call on the main event loop and blocks for the result.
        """
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})

        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        elif isinstance(prompt, list):
            messages = prompt
        else:
            messages.append({"role": "user", "content": str(prompt)})

        try:
            future = asyncio.run_coroutine_threadsafe(
                self.client.complete(messages, model=self.sub_model),
                self._loop,
            )
            response = future.result(timeout=300)

            with self._token_lock:
                self.sub_input_tokens += response.input_tokens
                self.sub_output_tokens += response.output_tokens
                self.sub_call_count += 1

            return response.content
        except Exception as e:
            return f"Error in llm_query: {e}"

    def _rlm_query_sync(
        self,
        query: str,
        context: str | dict | list | None = None,
        context_dir: str | None = None,
    ) -> str:
        """Spawn a full child RLM with its own REPL loop.

        Unlike llm_query (single-shot), the child gets its own sandbox,
        can write code, call sub-LMs, and iterate until FINAL().

        Args:
            query: The question for the child RLM to answer.
            context: Data to pre-load as a `context` variable in the child's
                     namespace. Can be a string, dict, or list.
            context_dir: Directory path for child's file access (added to
                         its allowed_dirs). Also generates a file listing
                         in the child's initial prompt.
        """
        if self._rlm_factory is None:
            return "Error: rlm_query is not available (no RLM factory configured)"

        try:
            child_rlm = self._rlm_factory()

            # Build context string for the initial prompt (file listing)
            context_str = ""
            if context_dir:
                real_dir = os.path.realpath(context_dir)
                try:
                    files = sorted(os.listdir(real_dir))
                    sample_file = files[0] if files else "example.json"
                    context_str = (
                        f"DATA DIRECTORY: {real_dir}\n"
                        f"Contains {len(files)} files.\n\n"
                        f"To list files, use exactly:\n"
                        f'  list_files("{real_dir}")\n\n'
                        f"To read a file, use exactly:\n"
                        f'  read_file("{real_dir}/{sample_file}")\n\n'
                        f"File listing:\n"
                        + "\n".join(f"  {f}" for f in files[:100])
                    )
                except OSError as e:
                    context_str = f"Error listing {context_dir}: {e}"

            # Build initial_vars for the child
            child_vars = {}
            if context is not None:
                child_vars["context"] = context

            # Run the child RLM
            future = asyncio.run_coroutine_threadsafe(
                child_rlm.run(
                    query=query,
                    context=context_str,
                    initial_vars=child_vars if child_vars else None,
                    allowed_dirs=[os.path.realpath(context_dir)] if context_dir else None,
                ),
                self._loop,
            )
            result = future.result(timeout=600)

            # Roll up child's token usage
            with self._token_lock:
                self.sub_input_tokens += result.total_input_tokens
                self.sub_output_tokens += result.total_output_tokens

            return result.answer
        except Exception as e:
            return f"Error in rlm_query: {e}"

    def _read_file(self, path: str) -> str:
        """Read a file, enforcing allowed_dirs."""
        real = self._resolve_path(path)
        if real is None:
            return f"Error: access denied — {path} is not in allowed directories"
        try:
            with open(real) as f:
                return f.read()
        except Exception as e:
            return f"Error reading {path}: {e}"

    def _list_files(self, directory: str) -> list[str]:
        """List files in a directory, enforcing allowed_dirs."""
        real = self._resolve_path(directory)
        if real is None:
            return [f"Error: access denied — {directory} is not in allowed directories"]
        try:
            return sorted(os.listdir(real))
        except Exception as e:
            return [f"Error listing {directory}: {e}"]

    def _final(self, answer) -> None:
        """Set the final answer. Called from REPL code.

        Handles common LM mistakes:
        - Passing a dict/list instead of a JSON string → auto-serializes
        - Passing double-encoded JSON (json.dumps on a JSON string) → unwraps
        """
        import json

        # Auto-serialize dicts and lists to JSON instead of Python repr
        if isinstance(answer, (dict, list)):
            s = json.dumps(answer, ensure_ascii=False)
        else:
            s = str(answer)

        # Detect double-encoded JSON: a JSON string whose parsed value is itself a string
        # that looks like JSON (e.g., '"{\\"title\\":\\"Great Expectations\\"}"')
        if s.startswith('"') or s.startswith("'{"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, str) and parsed.lstrip().startswith(("{", "[")):
                    # Verify the inner string is valid JSON before unwrapping
                    json.loads(parsed)
                    s = parsed
            except (json.JSONDecodeError, ValueError):
                pass

        self.final_answer = s

    def _resolve_path(self, path: str) -> str | None:
        """Resolve a path, trying both absolute and relative to each allowed_dir.

        Returns the resolved real path if allowed, None if denied.
        """
        if not self.allowed_dirs:
            return os.path.realpath(path)

        # First try the path as-is (absolute or relative to cwd)
        real = os.path.realpath(path)
        if any(real.startswith(d) for d in self.allowed_dirs):
            return real

        # Then try resolving relative to each allowed directory
        for allowed in self.allowed_dirs:
            candidate = os.path.realpath(os.path.join(allowed, path))
            if any(candidate.startswith(d) for d in self.allowed_dirs):
                if os.path.exists(candidate):
                    return candidate

        return None
