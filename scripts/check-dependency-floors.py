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
    python3 scripts/check-dependency-floors.py --keep     # retain the temporary floor workspace

Resolution and compilation run in a temporary copy of the current workspace.
The caller's Cargo.lock is never written, including on failure or interruption.
--keep retains that copy for inspection; it never replaces the caller's lockfile.

When a floor is unreachable — some other dependency in this tree requires a
newer version than rig declares — the script bisects to the *lowest version the
tree admits* and checks against that, reporting the gap. Rig's declared floor is
still rig's own honest requirement (a downstream with older transitives can
resolve below what this tree can), but it can only be verified from the lowest
reachable version up; the report says exactly which floors carry that caveat.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "Cargo.lock"
INDEX = "https://index.crates.io"


def phase(message: str) -> None:
    print(message, flush=True)


def isolated_workspace(root: Path, target: Path) -> Path:
    """Copy working bytes, including uncommitted/ignored inputs, not Git HEAD.

    Only Git metadata and the known build-output directories are omitted.
    Refuse symlinks rather than allowing a lowered lockfile/build to write back
    through one. Cargo artifacts use a stable target outside this source copy.
    """
    destination = Path(tempfile.mkdtemp(prefix=f".{root.name}-dependency-floors-", dir=root.parent))
    omitted = {root / ".git", root / "target", target}

    def ignore(directory: str, names: list[str]) -> list[str]:
        paths = [Path(directory) / name for name in names]
        for path in paths:
            if path not in omitted and path.is_symlink():
                raise RuntimeError(f"unsupported symlink in floor inputs: {path}")
        return [path.name for path in paths if path in omitted]

    try:
        shutil.copytree(root, destination, dirs_exist_ok=True, ignore=ignore)
    except BaseException:
        shutil.rmtree(destination)
        raise
    return destination



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
    # Build metadata does not affect SemVer precedence or requirement matching.
    # e.g. toml publishes 1.1.2+spec-1.1.0 for the requirement ^1.1.2.
    version = version.split("+", 1)[0]
    if "-" in version:
        return False
    if req.startswith("="):
        return version == req[1:].split("+", 1)[0]
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
    phase(f"RESOLVE cargo update -p {name}@{locked} --precise {version}")
    started = time.monotonic()
    result = subprocess.run(
        ["cargo", "update", "-p", f"{name}@{locked}", "--precise", version],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    phase(f"RESOLVE {name}@{version}: exit {result.returncode}, {time.monotonic() - started:.3f}s measured")
    return result


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


def check_floors() -> int:
    downgraded, skipped, unreachable = [], [], []
    resolution_start = time.monotonic()
    phase("RESOLVE workspace metadata (fresh all-feature resolution)")
    try:
        for name, (req, locked) in sorted(workspace_direct_deps().items()):
            versions = matching_versions(name, req)
            if not versions:
                phase(f"FAILED {name}: no published version satisfies {req}; floor is unverified")
                return 1
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
        phase(f"RESOLUTION complete: {time.monotonic() - resolution_start:.3f}s measured")
        phase("COMPILE cargo check --workspace --all-features --all-targets")
        compile_start = time.monotonic()
        check = subprocess.run(
            ["cargo", "check", "--workspace", "--all-features", "--all-targets"], cwd=ROOT
        )
        phase(f"COMPILE complete: {time.monotonic() - compile_start:.3f}s measured; exit {check.returncode}")
        if check.returncode != 0:
            print("\nFAILED: the workspace does not build against its declared floors.")
            print("Raise the offending floor(s) in [workspace.dependencies] to the first version that has the API in use.")
            return check.returncode
        print("\nOK: the workspace builds against its declared dependency floors.")
        return 0
    finally:
        phase(f"FLOORS total: {time.monotonic() - resolution_start:.3f}s measured")


def guard_source_paths(caller: Path, metadata: dict) -> None:
    """Reject relocation-unsafe paths, including patches hidden by --no-deps."""
    def paths(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "path" and isinstance(item, str):
                    yield item
                yield from paths(item)
        elif isinstance(value, list):
            for item in value:
                yield from paths(item)

    manifests = {caller / "Cargo.toml"}
    excluded = {caller / ".git", caller / "target"}
    if "target_directory" in metadata:
        excluded.add(Path(metadata["target_directory"]))
    for directory, names, files in os.walk(caller):
        names[:] = [name for name in names if Path(directory) / name not in excluded]
        if "Cargo.toml" in files:
            manifests.add(Path(directory) / "Cargo.toml")
    for manifest in manifests:
        if not manifest.exists():
            continue
        for path in paths(tomllib.loads(manifest.read_text())):
            if Path(path).is_absolute() or not (manifest.parent / path).resolve().is_relative_to(caller.resolve()):
                raise RuntimeError(f"path cannot be relocated for floor isolation: {manifest}: {path}")
    config_dirs = [parent / ".cargo" for parent in (caller, *caller.parents)]
    config_dirs.append(Path(os.environ.get("CARGO_HOME", str(Path.home() / ".cargo"))))
    for directory in config_dirs:
        for name in ("config", "config.toml"):
            config = directory / name
            if config.exists():
                value = tomllib.loads(config.read_text())
                if any(key in value for key in ("paths", "patch", "replace", "source")):
                    raise RuntimeError(f"Cargo source/path override cannot be isolated safely: {config}")
    if any(key.startswith(("CARGO_SOURCE_", "CARGO_PATCH_")) for key in os.environ):
        raise RuntimeError("Cargo source/patch environment override cannot be isolated safely")


def main() -> int:
    global ROOT, LOCK
    if any(arg != "--keep" for arg in sys.argv[1:]):
        raise ValueError("usage: check-dependency-floors.py [--keep]")
    caller = ROOT
    phase("PREFLIGHT cargo metadata --locked --no-deps; determine stable target and path dependencies")
    metadata = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"], cwd=caller
    ))
    guard_source_paths(caller, metadata)
    for package in metadata["packages"]:
        for dependency in package["dependencies"]:
            if dependency.get("path") and not Path(dependency["path"]).resolve().is_relative_to(caller):
                raise RuntimeError(f"external path dependency cannot be isolated: {dependency['path']}")
    target = Path(metadata["target_directory"]).resolve()
    if caller.is_relative_to(target):
        raise RuntimeError("Cargo target directory must not contain the source workspace")
    phase("COPY current working inputs for floor-only lockfile isolation")
    copy = isolated_workspace(caller, target)
    previous_target = os.environ.get("CARGO_TARGET_DIR")
    os.environ["CARGO_TARGET_DIR"] = str(target / "dependency-floors")
    ROOT, LOCK = copy, copy / "Cargo.lock"
    phase(f"FLOORS source: {copy}; caller lockfile untouched; target: {os.environ['CARGO_TARGET_DIR']}")
    try:
        return check_floors()
    finally:
        ROOT, LOCK = caller, caller / "Cargo.lock"
        if previous_target is None:
            os.environ.pop("CARGO_TARGET_DIR", None)
        else:
            os.environ["CARGO_TARGET_DIR"] = previous_target
        if "--keep" in sys.argv:
            phase(f"Retained floor workspace: {copy}")
        else:
            shutil.rmtree(copy)


if __name__ == "__main__":
    sys.exit(main())
