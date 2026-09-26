"""Summarise a Harbor job of rig-coder trials and record it in the ledger.

    python3 bench/ledger.py summary runs/<job>
    python3 bench/ledger.py record runs/<job> --dataset quixbugs@1.0 --rung 3 [--note ...]
    python3 bench/ledger.py compare runs/<before> runs/<after>

Cost uses the ladder's accounting rates for gemini-3.8-flash: $0.75 per million
input tokens (cached included) and $3.75 per million output tokens. A trial
without recorded usage has unknown cost, which `summary` reports separately;
it is never counted as zero.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "ledger.jsonl"
INPUT_RATE = 0.75 / 1_000_000
OUTPUT_RATE = 3.75 / 1_000_000


def wilson(passes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = passes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def trial_facts(trial: Path) -> dict:
    result = json.loads((trial / "result.json").read_text())
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    reward = rewards.get("reward")
    agent = result.get("agent_result") or {}
    meta = agent.get("metadata") or {}
    exception = result.get("exception_info")
    input_tokens = agent.get("n_input_tokens")
    output_tokens = agent.get("n_output_tokens")
    cost = None
    if isinstance(input_tokens, int) and isinstance(output_tokens, int):
        cost = input_tokens * INPUT_RATE + output_tokens * OUTPUT_RATE
    return {
        "task": result.get("task_name"),
        "trial": trial.name,
        "reward": reward,
        "passed": isinstance(reward, (int, float)) and reward >= 1.0,
        "exception": (exception or {}).get("exception_type") if exception else None,
        "ending": meta.get("ending"),
        "error": meta.get("error"),
        "turns": meta.get("turns"),
        "tool_calls": meta.get("tool_calls"),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_tokens": agent.get("n_cache_tokens"),
        "cost": cost,
    }


def job_trials(job: Path) -> list[dict]:
    trials = [p for p in sorted(job.iterdir()) if (p / "result.json").is_file() and p.is_dir()]
    return [trial_facts(t) for t in trials]


def aggregate(trials: list[dict]) -> dict:
    n = len(trials)
    passes = sum(1 for t in trials if t["passed"])
    known = [t["cost"] for t in trials if t["cost"] is not None]
    unknown = n - len(known)
    low, high = wilson(passes, n)
    cost = sum(known)
    return {
        "trials": n,
        "passed": passes,
        "mean": passes / n if n else 0.0,
        "ci_low": low,
        "ci_high": high,
        "cost_known": round(cost, 4),
        "cost_unknown_trials": unknown,
        "cost_per_resolved": round(cost / passes, 4) if passes else None,
        "errored": sum(1 for t in trials if t["exception"]),
    }


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=HERE, capture_output=True, text=True, check=False
    ).stdout.strip()


def cmd_summary(args: argparse.Namespace) -> int:
    trials = job_trials(Path(args.job))
    print(f"{'task':40} {'rew':>4} {'turns':>5} {'calls':>5} {'cost':>7}  ending / exception")
    for t in sorted(trials, key=lambda t: (t["passed"], t["task"] or "")):
        cost = f"{t['cost']:.3f}" if t["cost"] is not None else "?"
        reward = "-" if t["reward"] is None else f"{t['reward']:.0f}"
        detail = t["exception"] or t["ending"] or ""
        if t["error"]:
            detail += f": {str(t['error'])[:60]}"
        print(
            f"{(t['task'] or '?')[:40]:40} {reward:>4} {str(t['turns'] or '-'):>5} "
            f"{str(t['tool_calls'] or '-'):>5} {cost:>7}  {detail}"
        )
    print(json.dumps(aggregate(trials)))
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    job = Path(args.job).resolve()
    trials = job_trials(job)
    receipt = next(HERE.glob("bin/*.build.json"), None)
    row = {
        "time": int(time.time()),
        "job": job.name,
        "dataset": args.dataset,
        "rung": args.rung,
        "model": args.model,
        "commit": git("rev-parse", "--short=12", "HEAD"),
        "binary": json.loads(receipt.read_text()) if receipt else None,
        "note": args.note,
        **aggregate(trials),
        "rewards": {t["task"]: t["reward"] for t in trials},
    }
    with LEDGER.open("a") as ledger:
        ledger.write(json.dumps(row) + "\n")
    print(json.dumps({k: v for k, v in row.items() if k != "rewards"}))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    before = {t["task"]: t for t in job_trials(Path(args.before))}
    after = {t["task"]: t for t in job_trials(Path(args.after))}
    for task in sorted(set(before) | set(after)):
        b, a = before.get(task), after.get(task)
        rb = "-" if not b else int(b["passed"])
        ra = "-" if not a else int(a["passed"])
        cb = f"{b['cost']:.3f}" if b and b["cost"] is not None else "?"
        ca = f"{a['cost']:.3f}" if a and a["cost"] is not None else "?"
        flag = "" if rb == ra else ("  GAIN" if ra == 1 else "  LOSS")
        print(f"{task[:44]:44} {rb} -> {ra}   ${cb} -> ${ca}{flag}")
    print("before", json.dumps(aggregate(list(before.values()))))
    print("after ", json.dumps(aggregate(list(after.values()))))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    summary = sub.add_parser("summary")
    summary.add_argument("job")
    summary.set_defaults(run=cmd_summary)
    record = sub.add_parser("record")
    record.add_argument("job")
    record.add_argument("--dataset", required=True)
    record.add_argument("--rung", type=int)
    record.add_argument("--model", default="gemini/gemini-3.8-flash")
    record.add_argument("--note")
    record.set_defaults(run=cmd_record)
    compare = sub.add_parser("compare")
    compare.add_argument("before")
    compare.add_argument("after")
    compare.set_defaults(run=cmd_compare)
    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
