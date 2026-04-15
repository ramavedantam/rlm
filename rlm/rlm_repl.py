"""Main RLM loop: iterate root LM, extract code, execute in REPL, repeat."""

from __future__ import annotations

import sys
from dataclasses import dataclass

from .client import LLMClient
from .repl_env import REPLEnv
from .prompts import (
    DEFAULT_SYSTEM_PROMPT,
    INITIAL_USER_PROMPT,
    NEXT_ACTION_PROMPT,
    FORCE_FINAL_PROMPT,
)
from .utils import extract_code_blocks, extract_final, truncate


@dataclass
class CodeLogEntry:
    depth: int
    iteration: int
    code: str
    stdout: str
    stderr: str


@dataclass
class RLMResult:
    answer: str
    iterations: int
    total_input_tokens: int
    total_output_tokens: int
    messages: list[dict]
    code_log: list[CodeLogEntry] | None = None


class RLM_REPL:
    """Recursive Language Model with REPL environment.

    The root LM writes Python code in ```repl``` blocks. Code executes in a
    sandbox that provides llm_query() for sub-LM calls and read_file() for
    filesystem access. The root LM sees execution output and iterates until
    it calls FINAL(answer).
    """

    def __init__(
        self,
        client: LLMClient,
        model: str | None = None,
        sub_client: LLMClient | None = None,
        sub_model: str | None = None,
        system_prompt: str | None = None,
        max_iterations: int = 15,
        allowed_dirs: list[str] | None = None,
        verbose: bool = False,
        depth: int = 0,
        code_log: list[CodeLogEntry] | None = None,
    ):
        self.client = client
        self.model = model
        self.sub_client = sub_client or client
        self.sub_model = sub_model
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.max_iterations = max_iterations
        self.allowed_dirs = allowed_dirs
        self.verbose = verbose
        self.depth = depth
        self.code_log = code_log if code_log is not None else []

        # Token accounting for root LM calls
        self._root_input_tokens = 0
        self._root_output_tokens = 0

    def _make_child_rlm(self) -> "RLM_REPL":
        """Factory for creating child RLM instances (used by rlm_query)."""
        return RLM_REPL(
            client=self.client,
            model=self.model,
            sub_client=self.sub_client,
            sub_model=self.sub_model,
            system_prompt=self.system_prompt,
            max_iterations=self.max_iterations,
            verbose=self.verbose,
            depth=self.depth + 1,
            code_log=self.code_log,  # shared across tree
            # Don't pass allowed_dirs — child's run() gets its own
        )

    _SUPPORTED_MODEL = "gpt-5.4"

    async def run(
        self,
        query: str,
        context: str = "",
        initial_vars: dict | None = None,
        allowed_dirs: list[str] | None = None,
    ) -> RLMResult:
        """Run the RLM loop on a query with optional context.

        Args:
            query: The question to answer.
            context: Optional context string (e.g., file listing, inline data).
            initial_vars: Variables to pre-load into the REPL namespace
                          (e.g., {"context": {...}} for pre-loaded data).
            allowed_dirs: Override allowed_dirs for this run (used by child RLMs).

        Returns:
            RLMResult with the answer and token usage.

        Raises:
            ValueError: If model is not gpt-5.4.
        """
        if self.model and self.model != self._SUPPORTED_MODEL:
            raise ValueError(
                f"RLM requires model '{self._SUPPORTED_MODEL}', got '{self.model}'. "
                f"Unbounded recursion depth is enabled (rlm_query children can spawn "
                f"indefinitely) and has only been tested with {self._SUPPORTED_MODEL}."
            )
        dirs = allowed_dirs or self.allowed_dirs
        repl = REPLEnv(
            client=self.sub_client,
            sub_model=self.sub_model,
            allowed_dirs=dirs,
            initial_vars=initial_vars,
            rlm_factory=self._make_child_rlm,
        )

        # Build initial messages
        context_section = ""
        if context:
            context_section = f"Context:\n{context}\n\n"

        vars_section = ""
        if initial_vars:
            var_descriptions = []
            for k, v in initial_vars.items():
                if isinstance(v, str):
                    var_descriptions.append(f"- `{k}` — string, {len(v):,} chars")
                elif isinstance(v, dict):
                    var_descriptions.append(f"- `{k}` — dict with {len(v)} keys: {list(v.keys())[:5]}")
                elif isinstance(v, list):
                    var_descriptions.append(f"- `{k}` — list with {len(v)} items")
                else:
                    var_descriptions.append(f"- `{k}` — {type(v).__name__}")
            vars_section = (
                "Pre-loaded variables (already in your namespace):\n"
                + "\n".join(var_descriptions)
                + "\n\n"
            )

        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": INITIAL_USER_PROMPT.format(
                    query=query,
                    context_section=context_section,
                    vars_section=vars_section,
                ),
            },
        ]

        self._root_input_tokens = 0
        self._root_output_tokens = 0

        for iteration in range(self.max_iterations):
            # Call root LM
            response = await self.client.complete(
                messages, model=self.model, temperature=0.7
            )
            self._root_input_tokens += response.input_tokens
            self._root_output_tokens += response.output_tokens
            assistant_text = response.content

            if self.verbose:
                self._log(f"\n--- Iteration {iteration + 1} ---")
                self._log(f"Root LM: {assistant_text[:300]}...")

            messages.append({"role": "assistant", "content": assistant_text})

            # Check for FINAL() in text (outside code blocks)
            text_final = extract_final(assistant_text)
            if text_final:
                return self._build_result(text_final, iteration + 1, repl, messages)

            # Extract and execute code blocks
            code_blocks = extract_code_blocks(assistant_text)

            if code_blocks:
                for code in code_blocks:
                    result = await repl.execute(code)

                    self.code_log.append(CodeLogEntry(
                        depth=self.depth,
                        iteration=iteration + 1,
                        code=code,
                        stdout=result.stdout,
                        stderr=result.stderr,
                    ))

                    if self.verbose:
                        self._log(f"REPL stdout: {result.stdout[:200]}")
                        if result.stderr:
                            self._log(f"REPL stderr: {result.stderr[:200]}")

                    # Build output message
                    output_parts = []
                    if result.stdout:
                        output_parts.append(result.stdout)
                    if result.stderr:
                        output_parts.append(f"Error:\n{result.stderr}")
                    if not output_parts:
                        output_parts.append("(no output)")

                    output_text = "\n".join(output_parts)
                    messages.append({
                        "role": "user",
                        "content": f"Code executed:\n```python\n{code}\n```\n\nOutput:\n{truncate(output_text, 100_000)}",
                    })

                    # Check if FINAL() was called from within code
                    if repl.final_answer is not None:
                        return self._build_result(
                            repl.final_answer, iteration + 1, repl, messages
                        )
            else:
                # No code blocks — nudge the root LM
                messages.append({"role": "user", "content": NEXT_ACTION_PROMPT})

        # Max iterations reached — force a final answer
        if self.verbose:
            self._log("Max iterations reached, forcing final answer")

        messages.append({"role": "user", "content": FORCE_FINAL_PROMPT})
        response = await self.client.complete(
            messages, model=self.model, temperature=0.7
        )
        self._root_input_tokens += response.input_tokens
        self._root_output_tokens += response.output_tokens
        final_text = response.content

        # Try to extract FINAL() from the forced response
        forced_final = extract_final(final_text)
        answer = forced_final or final_text

        return self._build_result(answer, self.max_iterations, repl, messages)

    def _build_result(
        self,
        answer: str,
        iterations: int,
        repl: REPLEnv,
        messages: list[dict],
    ) -> RLMResult:
        return RLMResult(
            answer=answer,
            iterations=iterations,
            total_input_tokens=self._root_input_tokens + repl.sub_input_tokens,
            total_output_tokens=self._root_output_tokens + repl.sub_output_tokens,
            messages=messages,
            code_log=self.code_log,
        )

    def _log(self, msg: str) -> None:
        print(msg, file=sys.stderr)
