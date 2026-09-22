//! Dependency-graph invariants for the runtime/transport-agnostic split.
//!
//! Runtime crates retain only execution mechanisms and the shared vocabulary.
//! Concrete recording and replay live in rig-cassette, whose runtime adapters
//! and native HTTP engine are independently selectable. Workspace graph checks
//! cover runtime back-edges; separate downstream manifests prove cassette
//! feature isolation without workspace dev-dependency unification.

use std::process::Command;

use super::verification_checks;

/// `(package, `cargo tree` feature arguments, forbidden dependency names,
/// required dependency names)`. The three lists are space-separated; empty
/// features select the package's defaults.
type Graph = (&'static str, &'static str, &'static str, &'static str);

/// The ECS runtime owns its tasks and Bevy schedules, never a concrete recorder,
/// another agent runtime, or a provider transport.
const ECS_LEAF: &str = "rig-agent rig-cassette rig-effect-log rig-rmcp rmcp bevy tokio reqwest";

const GRAPHS: &[Graph] = &[
    ("rig-core", "", "tokio reqwest rig-cassette", ""),
    (
        "rig-core",
        "--all-features",
        "tokio reqwest rig-agent rig-cassette rig-effect-log rig-ecs",
        "",
    ),
    (
        "rig-agent",
        "",
        "tokio reqwest rmcp rig-cassette rig-effect-log rig-ecs",
        "rig-core",
    ),
    (
        "rig-agent",
        "--no-default-features",
        "tokio reqwest rmcp rig-cassette rig-effect-log rig-ecs",
        "rig-core",
    ),
    (
        "rig-agent",
        "--all-features",
        "reqwest rmcp rig-cassette rig-effect-log rig-ecs",
        "rig-core",
    ),
    (
        "rig-ecs",
        "",
        "rig-agent rig-cassette rig-effect-log rig-rmcp rmcp bevy tokio reqwest bevy_asset",
        "rig-core bevy_ecs bevy_tasks bevy_reflect bevy_app bevy_time bevy_diagnostic",
    ),
    (
        "rig-ecs",
        "--all-features",
        ECS_LEAF,
        "bevy_reflect bevy_asset bevy_app",
    ),
    ("rig-rmcp", "", "rig-agent rig-cassette", ""),
    (
        "rig",
        "--no-default-features --features agent,derive",
        "tokio reqwest rmcp rig-ecs",
        "rig-core rig-agent rig-cassette",
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
        "tree", "--locked", "-p", package, "-e", "normal", "--target", "all", "--prefix", "none",
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
    for runtime in ["rig-agent", "rig-ecs"] {
        let package = metadata["packages"]
            .as_array()
            .expect("packages")
            .iter()
            .find(|package| package["name"] == runtime)
            .expect("runtime package");
        for dependency in package["dependencies"].as_array().expect("dependencies") {
            if dependency["kind"].is_null() {
                assert!(
                    !matches!(
                        dependency["name"].as_str(),
                        Some("rig-cassette" | "rig-effect-log")
                    ),
                    "{runtime} must not declare a concrete recording dependency, even optional or target-specific: {dependency}"
                );
            }
        }
    }
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
fn cassette_features_are_isolated_for_downstream_consumers() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let path = serde_json::to_string(&root.join("crates/rig-cassette")).expect("manifest path");
    for (features, forbidden, required) in [
        (
            "",
            "rig rig-agent rig-ecs bevy_app bevy_ecs bevy_tasks tokio reqwest rig-reqwest axum httpmock aws-smithy-eventstream aws-smithy-types",
            "rig-core",
        ),
        (
            "agent",
            "rig rig-ecs bevy_app bevy_ecs bevy_tasks tokio reqwest rig-reqwest axum httpmock aws-smithy-eventstream aws-smithy-types",
            "rig-core rig-agent",
        ),
        (
            "ecs",
            "rig rig-agent tokio reqwest rig-reqwest axum httpmock aws-smithy-eventstream aws-smithy-types",
            "rig-core rig-ecs bevy_app bevy_ecs",
        ),
        (
            "agent,ecs",
            "rig tokio reqwest rig-reqwest axum httpmock aws-smithy-eventstream aws-smithy-types",
            "rig-core rig-agent rig-ecs",
        ),
        (
            "http",
            "rig rig-agent rig-ecs bevy_app bevy_ecs aws-smithy-eventstream aws-smithy-types",
            "rig-core rig-reqwest reqwest tokio axum httpmock",
        ),
        (
            "bedrock",
            "rig rig-agent rig-ecs bevy_app bevy_ecs",
            "rig-core rig-reqwest aws-smithy-eventstream aws-smithy-types",
        ),
    ] {
        let scratch = assert_fs::TempDir::new().expect("isolated downstream package");
        std::fs::create_dir(scratch.path().join("src")).expect("source directory");
        std::fs::write(
            scratch.path().join("src/lib.rs"),
            "pub use rig_cassette::effect_log::EffectLog;\n",
        )
        .expect("consumer source");
        let selected: Vec<_> = features
            .split(',')
            .filter(|feature| !feature.is_empty())
            .collect();
        let selected = serde_json::to_string(&selected).expect("feature list");
        std::fs::write(
            scratch.path().join("Cargo.toml"),
            format!(
                "[package]\nname = \"cassette-feature-probe\"\nversion = \"0.0.0\"\nedition = \"2024\"\n[workspace]\n[dependencies]\nrig-cassette = {{ path = {path}, features = {selected} }}\n"
            ),
        )
        .expect("consumer manifest");
        std::fs::copy(root.join("Cargo.lock"), scratch.path().join("Cargo.lock"))
            .expect("seed locked dependency versions");
        let output = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
            .current_dir(scratch.path())
            .args([
                "tree",
                // Neither `--offline` nor `--locked`: `--target all` resolves
                // the platform-gated crates (Apple's `block2`, the Windows
                // bindings) that a host build never downloads, so an offline
                // probe fails on a cache warmed only by this workspace's own
                // targets; and the seeded lockfile has no entry for the probe
                // package itself, which `--locked` refuses to add. The seeded
                // lockfile still pins every version the graph shares with the
                // workspace.
                "-e", "normal", "--target", "all", "--prefix", "none", "--format", "{p} {f}",
            ])
            .output()
            .expect("resolve isolated downstream");
        assert!(
            output.status.success(),
            "{features}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let graph = String::from_utf8(output.stdout).expect("graph UTF-8");
        let names: Vec<_> = graph
            .lines()
            .filter_map(|line| line.split_whitespace().next())
            .collect();
        for name in forbidden.split_whitespace() {
            assert!(
                !names.contains(&name),
                "{features} unexpectedly enables {name}:\n{graph}"
            );
        }
        for name in required.split_whitespace() {
            assert!(
                names.contains(&name),
                "{features} is missing {name}:\n{graph}"
            );
        }
        if !matches!(features, "http" | "bedrock") {
            let json = graph
                .lines()
                .find(|line| line.starts_with("serde_json "))
                .expect("JSON dependency");
            for feature in ["preserve_order", "float_roundtrip"] {
                assert!(
                    !json.contains(feature),
                    "{features} must not enable serde_json/{feature}: {json}"
                );
            }
        }
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
