//! Dependency-graph invariants for the runtime/transport-agnostic split.
//!
//! Runtime crates retain only execution mechanisms and the shared vocabulary.
//! Concrete recording and replay live in rig-cassette, whose runtime adapter
//! and native HTTP engine are independently selectable. Workspace graph checks
//! cover runtime back-edges; separate downstream manifests prove cassette
//! feature isolation without workspace dev-dependency unification.

use std::process::Command;

/// `(package, `cargo tree` feature arguments, forbidden dependency names,
/// required dependency names)`. The three lists are space-separated; empty
/// features select the package's defaults.
type Graph = (&'static str, &'static str, &'static str, &'static str);

const GRAPHS: &[Graph] = &[
    ("rig-core", "", "tokio reqwest rig-cassette", ""),
    // The `reqwest` and `tungstenite` features bring in the bundled transports,
    // and with them reqwest and tokio: only through those two crates.
    (
        "rig-core",
        "--all-features",
        "rig-agent rig-cassette rig-effect-log",
        "rig-reqwest rig-tungstenite",
    ),
    (
        "rig-core",
        "--no-default-features --features derive,rustls,audio,image,pdf,epub,websocket",
        "tokio reqwest rig-reqwest rig-tungstenite rig-cassette",
        "rig-http",
    ),
    (
        "rig-agent",
        "",
        "tokio reqwest rmcp rig-cassette rig-effect-log",
        "rig-core",
    ),
    (
        "rig-agent",
        "--no-default-features",
        "tokio reqwest rmcp rig-cassette rig-effect-log",
        "rig-core",
    ),
    (
        "rig-agent",
        "--all-features",
        "reqwest rmcp rig-cassette rig-effect-log",
        "rig-core",
    ),
    ("rig-rmcp", "", "rig-agent rig-cassette", ""),
    // `rig::cassette` is opt-in: the classic agent alone does not bring the
    // recorder in.
    (
        "rig",
        "--no-default-features --features agent,derive",
        "tokio reqwest rmcp rig-cassette",
        "rig-core rig-agent",
    ),
    ("rig", "", "rig-cassette", "rig-core rig-agent"),
    (
        "rig",
        "--no-default-features --features cassette",
        "tokio reqwest rig-agent",
        "rig-core rig-cassette",
    ),
    (
        "rig",
        "--no-default-features --features agent,cassette",
        "tokio reqwest rmcp",
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
    // only in the metadata.
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
        .find(|package| package["name"] == "rig-agent")
        .expect("runtime package");
    for dependency in package["dependencies"].as_array().expect("dependencies") {
        if dependency["kind"].is_null() {
            assert!(
                !matches!(
                    dependency["name"].as_str(),
                    Some("rig-cassette" | "rig-effect-log")
                ),
                "rig-agent must not declare a concrete recording dependency, even optional or target-specific: {dependency}"
            );
        }
    }
}

#[test]
fn cassette_features_are_isolated_for_downstream_consumers() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let path = serde_json::to_string(&root.join("crates/rig-cassette")).expect("manifest path");
    for (features, forbidden, required) in [
        (
            "",
            "rig rig-agent tokio reqwest rig-reqwest axum httpmock aws-smithy-eventstream aws-smithy-types",
            "rig-core",
        ),
        (
            "agent",
            "rig tokio reqwest rig-reqwest axum httpmock aws-smithy-eventstream aws-smithy-types",
            "rig-core rig-agent",
        ),
        (
            "http",
            "rig rig-agent aws-smithy-eventstream aws-smithy-types",
            "rig-core rig-reqwest reqwest tokio axum httpmock",
        ),
        (
            "bedrock",
            "rig rig-agent",
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
    }
}
