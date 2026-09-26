"""Download a Harbor dataset into a job-owned snapshot and restrict the agent
phase's network to the model provider.

    python3 test-support/rig-coder/bench/harbor_policy.py quixbugs@1.0

writes `runs/datasets/<name>@<version>/` with every task's `[agent]` table set
to `network_mode = "allowlist"` and the provider host allowed. The registry
copy is never edited, and the verifier phase keeps the task's own policy.
A task that already declares an agent network mode is left alone and reported.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASETS = HERE.parent / "runs" / "datasets"
DEFAULT_HOSTS = ["generativelanguage.googleapis.com"]


def apply(task_toml: Path, hosts: list[str]) -> str:
    text = task_toml.read_text()
    config = tomllib.loads(text)
    agent = config.get("agent", {})
    if "network_mode" in agent:
        return f"kept task's own agent network_mode={agent['network_mode']!r}"
    policy = f'network_mode = "allowlist"\nallowed_hosts = {hosts!r}\n'.replace("'", '"')
    if re.search(r"^\[agent\]\s*$", text, flags=re.M):
        text = re.sub(r"^\[agent\]\s*$\n?", "[agent]\n" + policy, text, count=1, flags=re.M)
    else:
        text = text.rstrip("\n") + "\n\n[agent]\n" + policy
    parsed = tomllib.loads(text).get("agent", {})
    if parsed.get("network_mode") != "allowlist" or parsed.get("allowed_hosts") != hosts:
        raise ValueError(f"{task_toml}: policy did not apply")
    task_toml.write_text(text)
    return "allowlist"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset", help="name@version")
    parser.add_argument("--host", action="append", help="allowed host (repeatable)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    hosts = args.host or DEFAULT_HOSTS
    target = DATASETS / args.dataset
    if target.exists():
        if not args.overwrite:
            print(f"{target} exists; pass --overwrite to download again", file=sys.stderr)
            return 1
        shutil.rmtree(target)
    target.mkdir(parents=True)
    subprocess.run(
        ["harbor", "datasets", "download", args.dataset, "-o", str(target)], check=True
    )
    tasks = sorted(target.rglob("task.toml"))
    if not tasks:
        print(f"no task.toml under {target}", file=sys.stderr)
        return 1
    kept = []
    for task_toml in tasks:
        if apply(task_toml, hosts) != "allowlist":
            kept.append(task_toml.parent.name)
    roots = {task.parent.parent for task in tasks}
    print(f"{len(tasks)} tasks under {', '.join(str(r) for r in sorted(roots))}")
    if kept:
        print(f"{len(kept)} tasks declare their own agent network mode: {', '.join(kept)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
