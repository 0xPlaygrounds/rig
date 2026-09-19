//! Dependency-graph invariants for the runtime/transport-agnostic split.
//!
//! The crate boundaries that keep rig usable from non-tokio hosts are enforced
//! here rather than by convention: rig-core and rig-agent carry no runtime or
//! transport, rig-effect-log and rig-rmcp are rig-core-only leaves, rig-ecs is
//! rig-core plus the Bevy crates it installs into, and the facade with only
//! `agent` + `derive` pulls in none of tokio / reqwest / rmcp. Every boundary
//! is one row of `GRAPHS`: Cargo resolves that package's normal (non-dev,
//! non-build) dependency graph under the row's features, the forbidden names
//! must be absent from it and the required names present.

use std::process::Command;

use super::verification_checks;

/// `(package, `cargo tree` feature arguments, forbidden dependency names,
/// required dependency names)`. The three lists are space-separated; empty
/// features select the package's defaults.
type Graph = (&'static str, &'static str, &'static str, &'static str);

/// Neither agent runtime nor the facade: shared cassette test support stays
/// usable outside both.
const AGENT_RUNTIMES: &str = "rig rig-agent rig-ecs";
/// The recorder is a handler-side seam (`rig_core::serve::Recorder`) and the
/// serving policy a serve-side type (`rig_core::serve::ServingPolicy`), so the
/// log crate needs no runtime at all: registering a log's replayers on a driver
/// is each runtime's own (`rig_agent::bus::replay`, rig-ecs's `Replay`).
const LOG_LEAF: &str = "rig-agent rig-ecs tokio reqwest bevy_ecs";
/// rig-ecs's driver is a system, not a client of rig-agent's bus, and its agent
/// half is a rewrite held to rig-agent's bytes by the corpus alone: never
/// rig-agent, never the `bevy` facade, no runtime, no transport, no MCP.
const ECS_LEAF: &str = "rig-agent rig-rmcp rmcp bevy tokio reqwest";

const GRAPHS: &[Graph] = &[
    ("rig-cassette", "", AGENT_RUNTIMES, ""),
    ("rig-cassette", "--all-features", AGENT_RUNTIMES, ""),
    (
        "rig-cassette",
        "--no-default-features",
        "aws-smithy-eventstream aws-smithy-types",
        "",
    ),
    // rig-core carries no runtime or transport under any feature, and the
    // dependency runs one way: it knows nothing of the crates that drive its
    // handlers.
    ("rig-core", "", "tokio reqwest", ""),
    (
        "rig-core",
        "--all-features",
        "tokio reqwest rig-agent rig-effect-log rig-ecs",
        "",
    ),
    // The frozen runtime is runtime- and MCP-free with default features on
    // (tokio is optional, under `test-utils`) and with them off — the shape a
    // host that steps `AgentRun` itself depends on.
    ("rig-agent", "", "tokio reqwest rmcp", ""),
    (
        "rig-agent",
        "--no-default-features",
        "tokio rmcp reqwest",
        "",
    ),
    ("rig-effect-log", "", LOG_LEAF, "rig-core"),
    ("rig-effect-log", "--all-features", LOG_LEAF, ""),
    // rig-ecs is a `bevy_app` plugin over `bevy_ecs` and `bevy_tasks`, with
    // reflection (`bevy_reflect`, the checkpoint), the clock (`bevy_time`,
    // backoff) and diagnostics (`bevy_diagnostic`) unconditional. `bevy_asset`
    // joins only with `assets` — rig-ecs's one feature — and nothing else
    // joins either way: never the `bevy` facade, no runtime, no transport.
    (
        "rig-ecs",
        "",
        "rig-agent rig-rmcp rmcp bevy tokio reqwest bevy_asset",
        "rig-core rig-effect-log bevy_ecs bevy_tasks bevy_reflect bevy_app bevy_time bevy_diagnostic",
    ),
    (
        "rig-ecs",
        "--all-features",
        ECS_LEAF,
        "bevy_reflect bevy_asset bevy_app",
    ),
    ("rig-rmcp", "", "rig-agent", ""),
    (
        "rig",
        "--no-default-features --features agent,derive",
        "tokio reqwest rmcp",
        "",
    ),
];

/// `cargo` with `args`, run from the workspace root; stdout on success.
fn cargo_stdout(args: &[&str]) -> String {
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let output = Command::new(cargo)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args(args)
        .output()
        .expect("cargo runs");
    assert!(
        output.status.success(),
        "cargo {} failed:\n{}",
        args.join(" "),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("cargo output is utf-8")
}

/// `cargo tree -e normal --prefix none` for `package` under `features`, as the
/// package names in the resolved graph.
fn normal_dependency_names(package: &str, features: &str) -> Vec<String> {
    let mut args = vec![
        "tree", "--locked", "-p", package, "-e", "normal", "--prefix", "none",
    ];
    args.extend(features.split_whitespace());
    cargo_stdout(&args)
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .map(str::to_owned)
        .collect()
}

#[test]
fn crate_boundaries_hold_in_the_resolved_dependency_graph() {
    for (package, features, forbidden, required) in GRAPHS {
        let names = normal_dependency_names(package, features);
        let selection = if features.is_empty() {
            "default features"
        } else {
            features
        };
        for crate_name in forbidden.split_whitespace() {
            assert!(
                !names.iter().any(|name| name == crate_name),
                "`{package}` ({selection}) must not depend on `{crate_name}` through its normal dependencies"
            );
        }
        for crate_name in required.split_whitespace() {
            assert!(
                names.iter().any(|name| name == crate_name),
                "`{package}` ({selection}) must depend on `{crate_name}`"
            );
        }
    }

    // Optional and target-specific declarations, and manifest renames, show up
    // only in the metadata: rig-ecs directly owns its handler tasks
    // (`bevy_tasks`), its bounded private delivery queue (`futures`) and the
    // explicit web runtime selection that browser task driving needs
    // (`bevy_platform`) — and declares nothing else.
    let metadata: serde_json::Value = serde_json::from_str(&cargo_stdout(&[
        "metadata",
        "--locked",
        "--no-deps",
        "--format-version",
        "1",
    ]))
    .expect("metadata JSON");
    let package = metadata["packages"]
        .as_array()
        .expect("packages")
        .iter()
        .find(|package| package["name"] == "rig-ecs")
        .expect("rig-ecs package");
    let direct: Vec<_> = package["dependencies"]
        .as_array()
        .expect("dependencies")
        .iter()
        .filter(|dependency| dependency["kind"].is_null())
        .map(|dependency| dependency["name"].as_str().expect("dependency name"))
        .collect();
    for forbidden in ["tracing", "schemars", "futures-channel", "async-channel"] {
        assert!(
            !direct.contains(&forbidden),
            "rig-ecs must not depend directly on {forbidden}"
        );
    }
    for required in ["futures", "bevy_platform", "bevy_tasks"] {
        assert!(
            direct.contains(&required),
            "rig-ecs must depend directly on {required}"
        );
    }
}

#[test]
fn standalone_verification_preserves_minimal_json_features() {
    let checks = verification_checks::all();
    let mut packages = std::collections::BTreeSet::new();
    for id in ["bus-verification", "ecs-parity"] {
        let check = checks
            .iter()
            .find(|check| check.id == id)
            .expect("CI check");
        let standalone = check
            .steps
            .iter()
            .find(|step| step.args.windows(2).any(|pair| pair == ["--retries", "0"]))
            .expect("standalone verification execution");
        packages.insert(
            standalone
                .args
                .windows(2)
                .find(|pair| pair[0] == "-p")
                .expect("standalone package selection")[1]
                .as_str(),
        );
    }
    for package in packages {
        let graph = cargo_stdout(&[
            "tree",
            "--locked",
            "--offline",
            "-p",
            package,
            "--all-features",
            "-e",
            "normal,dev",
            "--invert",
            "serde_json",
            "--depth",
            "0",
            "--prefix",
            "none",
            "--format",
            "{p} {f}",
        ]);
        assert!(graph.lines().any(|line| line.starts_with("serde_json ")));
        for feature in ["preserve_order", "float_roundtrip"] {
            assert!(
                !graph.contains(feature),
                "{package} must replay goldens without serde_json/{feature}:\n{graph}"
            );
        }
    }
}
