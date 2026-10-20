#!/usr/bin/env python3
"""ESA Telemanom agent — CLI entrypoint.

Tools (implemented in tools.py):
  1) retrieve_knowledge
  2) score_telemetry
  3) compare_alarms_to_labels

LLM loop lives in llm.py. Paths / .env live in config.py.

Examples (from agent/ folder, conda env esa-agent):

  # RAG only
  python agent.py "What is the window size?"

  # Score + compare to labels
  python agent.py "Score the slice and compare alarms to ground-truth labels" \\
    --csv ../../data/preprocessed/multivariate/ESA-Mission1-semi-supervised/84_months.test.csv \\
    --skip-rows 33064 --max-rows 3000
"""
from __future__ import annotations

import argparse

# config first: sys.path + .env
import config  # noqa: F401
from llm import run_agent


def print_agent_result(result: dict, question: str, csv_hint: str | None) -> None:
    """Print provider, tool steps, and final answer to the terminal."""
    print("=== Mode: LLM agent (3 tools) ===")
    print(f"Provider: {result['provider']}")
    print(f"Model:    {result['model']}")
    if csv_hint:
        print(f"CSV hint: {csv_hint}")
    print(f"\nQuestion: {question}\n")

    for step in result["tool_steps"]:
        print(f"Thought/Action: `{step['tool']}` {step['args']}")
        obs = step["observation"]
        # Keep terminal output readable for long score/compare dumps.
        if len(obs) <= 1800:
            preview = obs
        else:
            preview = obs[:1800] + "…"
        print("Observation:\n", preview, "\n", sep="")

    print("Final answer:")
    print(result["final_answer"])


def main() -> None:
    """Parse CLI args and run the agent."""
    parser = argparse.ArgumentParser(
        description="ESA Telemanom agent: retrieve + score + compare-to-labels (CLI)"
    )
    parser.add_argument("question", help="User question")
    parser.add_argument("-k", type=int, default=5, help="Default RAG top-k (max 8)")
    parser.add_argument("--model", default=None, help="Optional LLM model override")
    parser.add_argument("--csv", type=str, default=None, help="Default telemetry CSV path")
    parser.add_argument("--max-rows", type=int, default=3000)
    parser.add_argument("--skip-rows", type=int, default=0)
    args = parser.parse_args()

    result = run_agent(
        args.question,
        k=args.k,
        model=args.model,
        csv_hint=args.csv,
        default_max_rows=args.max_rows,
        default_skip_rows=args.skip_rows,
    )
    print_agent_result(result, question=args.question, csv_hint=args.csv)


if __name__ == "__main__":
    main()
