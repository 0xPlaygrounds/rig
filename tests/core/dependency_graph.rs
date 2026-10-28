//! Dependency-graph invariants for the runtime/transport-agnostic split.
//!
//! Runtime crates retain only execution mechanisms and the shared vocabulary.
//! Concrete recording and replay live in rig-cassette, whose runtime adapter
//! and native HTTP engine are independently selectable. Workspace graph checks
//! cover runtime back-edges; separate downstream manifests prove cassette
//! feature isolation without workspace dev-dependency unification.

use std::process::Command;

use super::verification_checks;

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
            "traffic",
            "rig rig-agent tokio reqwest rig-reqwest axum httpmock flate2 aws-smithy-eventstream aws-smithy-types",
            "rig-core url",
        ),
        (
            "http",
            "rig rig-agent httpmock aws-smithy-eventstream aws-smithy-types",
            "rig-core rig-reqwest reqwest tokio axum",
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
        if matches!(features, "" | "agent") {
            assert_minimal_json_features(&graph, features);
        }
    }
}

fn assert_minimal_json_features(graph: &str, selection: &str) {
    let json: Vec<_> = graph
        .lines()
        .filter(|line| line.starts_with("serde_json "))
        .collect();
    assert!(!json.is_empty(), "{selection} must include serde_json");
    for line in json {
        let features = line.split_whitespace().nth(2).unwrap_or_default();
        for forbidden in ["preserve_order", "float_roundtrip"] {
            assert!(
                !features.split(',').any(|feature| feature == forbidden),
                "{selection} must not enable serde_json/{forbidden}: {line}"
            );
        }
    }
}

/// Retain Cargo's package, feature and platform selectors, leaving test-runner
/// options out of the dependency query. Dev and build edges participate in
/// the feature graph of the compiled test targets.
fn test_dependency_graph(step: &verification_checks::Step) -> String {
    assert_eq!(step.program, "cargo");
    let mut args = step.args.iter().map(String::as_str);
    assert_eq!(args.next(), Some("nextest"));
    assert_eq!(args.next(), Some("run"));
    let mut tree = vec![
        "tree",
        "--locked",
        "-e",
        "normal,build,dev",
        "--prefix",
        "none",
        "--format",
        "{p} {f}",
    ];
    while let Some(arg) = args.next() {
        match arg {
            "-p" | "--package" | "--exclude" | "--features" | "-F" | "--target"
            | "--manifest-path" => {
                tree.push(arg);
                tree.push(args.next().expect("Cargo selector value"));
            }
            "--workspace" | "--all-features" | "--no-default-features" => tree.push(arg),
            "--locked" => {}
            "--profile" | "--retries" | "-E" | "--filter-expr" | "--test" | "--bin"
            | "--example" => {
                args.next().expect("test option value");
            }
            "--lib" | "--tests" | "--bins" | "--examples" | "--all-targets" => {}
            other => panic!("unhandled verification argument: {other}"),
        }
    }
    cargo_stdout(&tree)
}

#[test]
fn isolated_verification_preserves_minimal_json_features() {
    let checks = verification_checks::all();
    for id in ["core-all", "bus-verification"] {
        let check = checks
            .iter()
            .find(|check| check.id == id)
            .expect("CI check");
        let steps: Vec<_> = check
            .steps
            .iter()
            .filter(|step| {
                id == "core-all"
                    || step
                        .args
                        .iter()
                        .any(|arg| matches!(arg.as_str(), "-p" | "--package"))
            })
            .collect();
        assert!(!steps.is_empty(), "{id} must have an isolated execution");
        for step in steps {
            let selection = format!("{id}: {}", step.args.join(" "));
            assert_minimal_json_features(&test_dependency_graph(step), &selection);
        }
    }
}
