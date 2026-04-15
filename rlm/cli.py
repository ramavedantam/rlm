#!/usr/bin/env python3
"""CLI entry point for the RLM module.

Usage:
    rlm -q "Why do users leave?" --context-dir runs/
    rlm -q "Summarize" --context-file data.txt --model gpt-5.4
    rlm -q "Find the needle" --context-inline "..." --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from rlm.client import OpenAIClient, AnthropicClient
from rlm.rlm_repl import RLM_REPL


def build_context(args: argparse.Namespace) -> tuple[str, list[str]]:
    """Build context string and allowed_dirs from CLI arguments.

    Returns (context_string, allowed_dirs).
    """
    context_parts = []
    allowed_dirs = []

    if args.context_dir:
        real_dir = os.path.realpath(args.context_dir)
        allowed_dirs.append(real_dir)
        try:
            files = sorted(os.listdir(real_dir))
            sample_file = files[0] if files else "example.json"
            context_parts.append(
                f"DATA DIRECTORY: {real_dir}\n"
                f"Contains {len(files)} files.\n\n"
                f"To list files, use exactly:\n"
                f'  list_files("{real_dir}")\n\n'
                f"To read a file, use exactly:\n"
                f'  read_file("{real_dir}/{sample_file}")\n\n'
                f"File listing:\n"
                + "\n".join(f"  {f}" for f in files[:100])
            )
            if len(files) > 100:
                context_parts.append(f"  ... and {len(files) - 100} more")
        except OSError as e:
            context_parts.append(f"Error listing {args.context_dir}: {e}")

    if args.context_file:
        real_file = os.path.realpath(args.context_file)
        allowed_dirs.append(os.path.dirname(real_file))
        try:
            with open(real_file) as f:
                content = f.read()
            if len(content) > 50_000:
                context_parts.append(
                    f"File {args.context_file} ({len(content)} chars) — too large to inline.\n"
                    f"Use read_file(\"{args.context_file}\") to access it."
                )
            else:
                context_parts.append(f"Contents of {args.context_file}:\n{content}")
        except OSError as e:
            context_parts.append(f"Error reading {args.context_file}: {e}")

    if args.context_inline:
        context_parts.append(args.context_inline)

    return "\n\n".join(context_parts), allowed_dirs


def make_client(provider: str, model: str):
    """Create an LLM client for the given provider."""
    if provider == "openai":
        return OpenAIClient(default_model=model)
    elif provider == "anthropic":
        return AnthropicClient(default_model=model)
    else:
        raise ValueError(f"Unknown provider: {provider}")


async def main_async(args: argparse.Namespace) -> None:
    context, allowed_dirs = build_context(args)

    # Build clients
    root_client = make_client(args.provider, args.model)
    if args.sub_provider or args.sub_model:
        sub_provider = args.sub_provider or args.provider
        sub_model = args.sub_model or args.model
        sub_client = make_client(sub_provider, sub_model)
    else:
        sub_client = root_client

    # Load custom system prompt
    system_prompt = None
    if args.system_prompt_file:
        with open(args.system_prompt_file) as f:
            system_prompt = f.read()

    rlm = RLM_REPL(
        client=root_client,
        model=args.model,
        sub_client=sub_client,
        sub_model=args.sub_model,
        system_prompt=system_prompt,
        max_iterations=args.max_iterations,
        allowed_dirs=allowed_dirs if allowed_dirs else None,
        verbose=args.verbose,
    )

    result = await rlm.run(query=args.query, context=context)

    if args.save_transcript:
        import datetime
        from dataclasses import asdict
        transcript = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "config": {
                "model": args.model,
                "sub_model": args.sub_model or args.model,
                "provider": args.provider,
                "sub_provider": args.sub_provider or args.provider,
                "max_iterations": args.max_iterations,
                "query": args.query,
            },
            "result": {
                "answer": result.answer,
                "iterations": result.iterations,
                "total_input_tokens": result.total_input_tokens,
                "total_output_tokens": result.total_output_tokens,
            },
            "messages": result.messages,
            "code_log": [asdict(e) for e in (result.code_log or [])],
        }
        with open(args.save_transcript, "w") as f:
            json.dump(transcript, f, indent=2, ensure_ascii=False)
        print(f"Transcript saved to {args.save_transcript}", file=sys.stderr)

        # Summary of code log
        if result.code_log:
            depths = set(e.depth for e in result.code_log)
            print(
                f"Code log: {len(result.code_log)} blocks across depths {sorted(depths)}",
                file=sys.stderr,
            )

    if args.json:
        output = {
            "answer": result.answer,
            "iterations": result.iterations,
            "total_input_tokens": result.total_input_tokens,
            "total_output_tokens": result.total_output_tokens,
        }
        print(json.dumps(output, indent=2))
    else:
        print(result.answer)

    if args.verbose:
        print(
            f"\n--- Stats ---\n"
            f"Iterations: {result.iterations}\n"
            f"Input tokens: {result.total_input_tokens:,}\n"
            f"Output tokens: {result.total_output_tokens:,}\n"
            f"Total tokens: {result.total_input_tokens + result.total_output_tokens:,}",
            file=sys.stderr,
        )


def main():
    parser = argparse.ArgumentParser(description="RLM — Recursive Language Model CLI")
    parser.add_argument("-q", "--query", required=True, help="The query to answer")
    parser.add_argument("--context-dir", help="Directory of files to analyze")
    parser.add_argument("--context-file", help="Single file to use as context")
    parser.add_argument("--context-inline", help="Inline context string")
    parser.add_argument("--model", default="gpt-5.4", help="Root LM model (default: gpt-5.4)")
    parser.add_argument("--sub-model", default=None, help="Sub-LM model (default: same as --model)")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic"], help="LLM provider")
    parser.add_argument("--sub-provider", default=None, choices=["openai", "anthropic"], help="Sub-LM provider")
    parser.add_argument("--max-iterations", type=int, default=15, help="Max REPL iterations (default: 15)")
    parser.add_argument("--system-prompt-file", help="Override system prompt with file contents")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print iteration details to stderr")
    parser.add_argument("--json", action="store_true", help="Output result as JSON")
    parser.add_argument("--save-transcript", metavar="PATH", help="Save full transcript (config, result, messages) to a JSON file")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
