#!/usr/bin/env python3
"""Downgrade every direct dependency to its declared floor and make sure the
workspace still builds.

A `[workspace.dependencies]` requirement is a floor, not a pin: `tokio = "1.49"`
means "any 1.x from 1.49.0 on". Downstream users resolve inside that range,
usually to whatever their lockfile already holds, so rig must actually compile
against the *lowest* version it declares — otherwise the floor is a lie that
surfaces as a build break in someone else's tree (#2195). Cargo has no stable
switch for this: `-Zdirect-minimal-versions` is nightly-only and dead-ends in
this workspace's transitive graph (lancedb/datafusion), so this script does the
one thing the flag would do that matters here — for each direct dependency,
`cargo update --precise` the lockfile entry to the lowest version that
satisfies the declared requirement — and then runs `cargo check`.

Run from the repository root:

    python3 scripts/check-dependency-floors.py            # check
    python3 scripts/check-dependency-floors.py --keep     # leave the lowered Cargo.lock in place

The lockfile is restored afterwards unless `--keep` is given.

When a floor is unreachable — some other dependency in this tree requires a
newer version than rig declares — the script bisects to the *lowest version the
tree admits* and checks against that, reporting the gap. Rig's declared floor is
still rig's own honest requirement (a downstream with older transitives can
resolve below what this tree can), but it can only be verified from the lowest
reachable version up; the report says exactly which floors carry that caveat.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "Cargo.lock"
INDEX = "https://index.crates.io"


def index_path(name: str) -> str:
    n = len(name)
    if n == 1:
        return f"1/{name}"
    if n == 2:
        return f"2/{name}"
    if n == 3:
        return f"3/{name[0]}/{name}"
    return f"{name[:2]}/{name[2:4]}/{name}"


def parse_version(v: str) -> tuple[int, ...]:
    core = v.split("+", 1)[0].split("-", 1)[0]
    return tuple(int(x) for x in core.split(".")[:3])


def satisfies(version: str, req: str) -> bool:
    """Minimal caret/exact semver matching, enough for the requirement forms
    this workspace uses (`"1"`, `"0.4"`, `"1.2.3"`, `"=1.2.3"`)."""
    if "-" in version or "+" in version:
        return False
    if req.startswith("="):
        return version == req[1:]
    req = req.lstrip("^")
    floor = parse_version(req)
    v = parse_version(version)
    if v < floor:
        return False
    # caret: same leading non-zero component
    rp = req.split(".")
    if rp[0] != "0":
        return v[0] == floor[0]
    if len(rp) == 1:
        return v[0] == 0
    if rp[1] != "0":
        return v[0] == 0 and v[1] == floor[1]
    if len(rp) == 2:
        return v[0] == 0 and v[1] == 0
    return v[:3] == floor[:3]


def matching_versions(name: str, req: str) -> list[str]:
    """Every published, non-yanked version satisfying `req`, ascending."""
    with urllib.request.urlopen(f"{INDEX}/{index_path(name)}") as resp:
        lines = resp.read().decode().splitlines()
    candidates = []
    for line in lines:
        entry = json.loads(line)
        if entry.get("yanked"):
            continue
        if satisfies(entry["vers"], req):
            candidates.append(entry["vers"])
    candidates.sort(key=parse_version)
    return candidates


def try_precise(name: str, locked: str, version: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["cargo", "update", "-p", f"{name}@{locked}", "--precise", version],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def lowest_reachable(name: str, locked: str, versions: list[str]) -> str | None:
    """Bisect `versions` (ascending, all below `locked`) for the lowest one
    the resolver accepts. Assumes monotonicity: if a version is admitted, every
    later one is too — true for the "a transitive needs at least X" case this
    handles."""
    lo, hi, found = 0, len(versions) - 1, None
    while lo <= hi:
        mid = (lo + hi) // 2
        if try_precise(name, locked, versions[mid]).returncode == 0:
            found = versions[mid]
            hi = mid - 1
            # the lockfile now holds `found`; later probes update from there
            locked = found
        else:
            lo = mid + 1
    return found


def workspace_direct_deps() -> dict[str, tuple[str, str]]:
    """name -> (requirement, locked version) for every crates.io dependency
    that some workspace member depends on directly."""
    meta = json.loads(
        subprocess.check_output(
            ["cargo", "metadata", "--format-version", "1", "--all-features"], cwd=ROOT
        )
    )
    members = set(meta["workspace_members"])
    by_id = {p["id"]: p for p in meta["packages"]}
    resolve = {n["id"]: n for n in meta["resolve"]["nodes"]}
    deps: dict[str, tuple[str, str]] = {}
    for member in members:
        pkg = by_id[member]
        declared = {}
        for d in pkg["dependencies"]:
            if d.get("source") is None or "crates.io" not in d["source"]:
                continue
            declared[d["name"]] = d["req"]
        for dep in resolve[member]["deps"]:
            dep_pkg = by_id[dep["pkg"]]
            name = dep_pkg["name"]
            if name in declared and "crates.io" in (dep_pkg.get("source") or ""):
                req = declared[name]
                # keep the tightest requirement seen across members
                prev = deps.get(name)
                if prev is None or parse_version(req.lstrip("^=")) > parse_version(prev[0].lstrip("^=")):
                    deps[name] = (req, dep_pkg["version"])
    return deps


def main() -> int:
    keep = "--keep" in sys.argv
    backup = LOCK.read_bytes()
    downgraded, skipped, unreachable = [], [], []
    try:
        for name, (req, locked) in sorted(workspace_direct_deps().items()):
            versions = matching_versions(name, req)
            if not versions:
                print(f"?? {name}: no published version satisfies {req}")
                continue
            lowest = versions[0]
            if parse_version(lowest) >= parse_version(locked):
                skipped.append(name)
                continue
            result = try_precise(name, locked, lowest)
            if result.returncode == 0:
                downgraded.append((name, locked, lowest))
                print(f"↓  {name}: {locked} -> {lowest}  (floor {req})")
                continue
            lines = result.stderr.strip().splitlines()
            reason = next(
                (l.strip() for l in lines if l.strip().startswith("required by package")),
                lines[-1].strip() if lines else "?",
            )
            below = [v for v in versions if parse_version(v) < parse_version(locked)]
            reachable = lowest_reachable(name, locked, below[1:]) if len(below) > 1 else None
            if reachable:
                downgraded.append((name, locked, reachable))
                unreachable.append((name, req, reachable))
                print(
                    f"↓· {name}: {locked} -> {reachable}  (floor {req} unreachable in this "
                    f"tree — {reason}; verified from {reachable} up)"
                )
            else:
                unreachable.append((name, req, locked))
                print(f"·  {name}: floor {req} unreachable in this tree ({reason}); stays at {locked}")
        print(
            f"\n{len(downgraded)} downgraded, {len(skipped)} already at floor, "
            f"{len(unreachable)} floors below what this tree admits (verified from the lowest "
            f"admitted version up); checking…\n"
        )
        check = subprocess.run(
            ["cargo", "check", "--workspace", "--all-features", "--all-targets"], cwd=ROOT
        )
        if check.returncode != 0:
            print("\nFAILED: the workspace does not build against its declared floors.")
            print("Raise the offending floor(s) in [workspace.dependencies] to the first version that has the API in use.")
            return check.returncode
        print("\nOK: the workspace builds against its declared dependency floors.")
        return 0
    finally:
        if not keep:
            LOCK.write_bytes(backup)


if __name__ == "__main__":
    sys.exit(main())
