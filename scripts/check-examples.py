#!/usr/bin/env python3
"""Type-check every Rig example as an independent Cargo consumer.

Each standalone example package gets its own process: Cargo feature
unification is scoped to one invocation, so an example cannot borrow a
provider feature from another workspace member — the property this script
exists to enforce. Crate-local `[[example]]` targets that share a package
*and* a `required-features` set resolve identically, so they are batched
into one invocation; that is a smaller process count, not a weaker check.

`cargo check` rather than `cargo build`: these examples are compile-only
(nothing here runs them), and skipping codegen and linking cuts roughly a
third of the CPU.

Builds are ordered by feature set so that a contiguous shard recompiles
`rig`/`rig-core` once per distinct set rather than once per example.
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
    # Groups builds that resolve to the same dependency graph, so sharding can
    # keep them together instead of paying for the same rebuild in two shards.
    feature_key: str = ""


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


def rig_feature_key(package: dict[str, object]) -> str:
    """The example's `rig` dependency features, as a sort key.

    Two examples with the same key resolve `rig`/`rig-core` identically, so
    running them back to back reuses the compilation instead of evicting it.
    """
    dependencies = package.get("dependencies")
    if not isinstance(dependencies, list):
        return ""
    for dependency in dependencies:
        if not isinstance(dependency, dict) or dependency.get("name") != "rig":
            continue
        features = dependency.get("features")
        if not isinstance(features, list):
            return ""
        return ",".join(sorted(str(feature) for feature in features))
    return ""


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
                    "check",
                    "--locked",
                    "--manifest-path",
                    str(manifest),
                    "--package",
                    name,
                ),
                feature_key=rig_feature_key(package),
            )
        )

    grouped: dict[tuple[str, str], list[str]] = {}
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
            grouped.setdefault(
                (package_name, ",".join(sorted(required))), []
            ).append(target_name)

    groups_per_package: dict[str, int] = {}
    for package_name, _ in grouped:
        groups_per_package[package_name] = groups_per_package.get(package_name, 0) + 1
    for (package_name, features), target_names in grouped.items():
        command = ["cargo", "check", "--locked", "--package", package_name]
        for target_name in sorted(target_names):
            command.extend(["--example", target_name])
        if features:
            command.extend(["--features", features])
        # One identifier per invocation; the feature set disambiguates a
        # package whose targets fall into more than one group.
        identifier = f"crate:{package_name}"
        if groups_per_package[package_name] > 1:
            identifier += f"[{features or 'default'}]"
        builds.append(
            Build(
                identifier=identifier,
                command=tuple(command),
                feature_key=f"{package_name}::{features}",
            )
        )

    builds.sort(key=lambda build: (build.feature_key, build.identifier))
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

    # Contiguous slices, not round robin: `all_builds` is ordered by feature
    # set, so a slice recompiles each distinct `rig` configuration once.
    # Striding would scatter one feature set across every shard and pay for it
    # in each.
    total = len(all_builds)
    start = total * args.shard_index // args.shard_count
    end = total * (args.shard_index + 1) // args.shard_count
    selected = all_builds[start:end]
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
