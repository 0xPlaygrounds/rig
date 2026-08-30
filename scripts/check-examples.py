#!/usr/bin/env python3
"""Build every Rig example as an independent Cargo consumer.

The commands are intentionally one-target-per-process. Cargo feature
unification is scoped to each invocation, so an example cannot borrow a
provider feature from another workspace member.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import subprocess
import sys
from collections.abc import Sequence


ROOT = pathlib.Path(__file__).resolve().parents[1]


@dataclasses.dataclass(frozen=True)
class Build:
    identifier: str
    command: tuple[str, ...]


def cargo_metadata() -> dict[str, object]:
    completed = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps", "--locked"],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Cargo metadata failed with exit code {completed.returncode}")
    return json.loads(completed.stdout)


def packages(metadata: dict[str, object]) -> list[dict[str, object]]:
    value = metadata.get("packages")
    if not isinstance(value, list):
        raise RuntimeError("Cargo metadata did not contain a packages array")
    return value


def relative_manifest(package: dict[str, object]) -> pathlib.Path:
    raw = package.get("manifest_path")
    if not isinstance(raw, str):
        raise RuntimeError("Cargo package is missing manifest_path")
    manifest = pathlib.Path(raw).resolve()
    try:
        return manifest.relative_to(ROOT)
    except ValueError as error:
        raise RuntimeError(f"Cargo reported a package outside the repository: {manifest}") from error


def discover() -> list[Build]:
    workspace = cargo_metadata()
    workspace_packages = packages(workspace)

    physical_manifests = {
        path.relative_to(ROOT)
        for path in (ROOT / "examples").glob("*/Cargo.toml")
    }
    workspace_example_packages: list[dict[str, object]] = []
    metadata_manifests: set[pathlib.Path] = set()
    for package in workspace_packages:
        manifest = relative_manifest(package)
        if len(manifest.parts) == 3 and manifest.parts[0] == "examples":
            workspace_example_packages.append(package)
            metadata_manifests.add(manifest)

    missing = sorted(physical_manifests - metadata_manifests)
    stale = sorted(metadata_manifests - physical_manifests)
    if missing or stale:
        details = []
        if missing:
            details.append("not represented by Cargo metadata: " + ", ".join(map(str, missing)))
        if stale:
            details.append("metadata entries without a manifest: " + ", ".join(map(str, stale)))
        raise RuntimeError("Unaccounted standalone example manifests: " + "; ".join(details))

    builds: list[Build] = []
    for package in workspace_example_packages:
        name = package.get("name")
        if not isinstance(name, str):
            raise RuntimeError("Cargo package is missing its name")
        manifest = relative_manifest(package)
        builds.append(
            Build(
                identifier=f"standalone:{manifest.parent.name}",
                command=(
                    "cargo",
                    "build",
                    "--locked",
                    "--manifest-path",
                    str(manifest),
                    "--package",
                    name,
                ),
            )
        )

    for package in workspace_packages:
        manifest = relative_manifest(package)
        if manifest in metadata_manifests:
            continue
        package_name = package.get("name")
        targets = package.get("targets")
        if not isinstance(package_name, str) or not isinstance(targets, list):
            raise RuntimeError(f"Malformed package metadata for {manifest}")
        for target in targets:
            if not isinstance(target, dict):
                raise RuntimeError(f"Malformed target metadata for {package_name}")
            kinds = target.get("kind")
            if not isinstance(kinds, list) or "example" not in kinds:
                continue
            target_name = target.get("name")
            required = target.get("required-features", [])
            if not isinstance(target_name, str) or not isinstance(required, list) or not all(
                isinstance(feature, str) for feature in required
            ):
                raise RuntimeError(f"Malformed example target metadata for {package_name}")
            command = [
                "cargo",
                "build",
                "--locked",
                "--package",
                package_name,
                "--example",
                target_name,
            ]
            if required:
                command.extend(["--features", ",".join(sorted(required))])
            builds.append(
                Build(
                    identifier=f"crate:{package_name}/{target_name}",
                    command=tuple(command),
                )
            )

    builds.sort(key=lambda build: build.identifier)
    identifiers = [build.identifier for build in builds]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Example discovery produced duplicate identifiers")
    for build in builds:
        if "--workspace" in build.command or "--all-features" in build.command:
            raise RuntimeError(f"Non-independent example command generated for {build.identifier}")
    return builds


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list discovered builds without running them")
    parser.add_argument("--shard-index", type=int, default=0, help="zero-based shard index")
    parser.add_argument("--shard-count", type=int, default=1, help="total deterministic shard count")
    args = parser.parse_args(argv)
    if args.shard_count < 1:
        parser.error("--shard-count must be at least 1")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")
    return args


def main(argv: Sequence[str] = ()) -> int:
    args = parse_args(argv)
    try:
        all_builds = discover()
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    selected = [
        build
        for position, build in enumerate(all_builds)
        if position % args.shard_count == args.shard_index
    ]
    print(
        f"Discovered {len(all_builds)} independent example builds; "
        f"shard {args.shard_index + 1}/{args.shard_count} contains {len(selected)}."
    )
    if args.list:
        for build in selected:
            print(f"{build.identifier}: {' '.join(build.command)}")
        return 0

    failures: list[tuple[Build, int, str]] = []
    for position, build in enumerate(selected, start=1):
        print(f"\n[{position}/{len(selected)}] {build.identifier}", flush=True)
        completed = subprocess.run(
            build.command,
            cwd=ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if completed.returncode != 0:
            failures.append((build, completed.returncode, completed.stdout))

    if failures:
        print(f"\n{len(failures)} example build(s) failed:", file=sys.stderr)
        for build, returncode, output in failures:
            print(
                f"  - {build.identifier} (exit {returncode}): {' '.join(build.command)}",
                file=sys.stderr,
            )
            print(output.rstrip(), file=sys.stderr)
        return 1

    print(f"\nAll {len(selected)} selected example builds passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
