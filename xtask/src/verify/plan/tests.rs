use super::*;
use crate::{support::output, verify::checks};
use std::path::PathBuf;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf()
}

/// The plan for `paths` over this checkout.
fn planned(paths: &[&str]) -> Plan {
    let root = root();
    let metadata: serde_json::Value = serde_json::from_str(
        &output(
            &root,
            "cargo",
            &["metadata", "--locked", "--no-deps", "--format-version", "1"],
        )
        .unwrap(),
    )
    .unwrap();
    let graph = Graph::from_metadata(&metadata).unwrap();
    let files: Vec<String> = output(&root, "git", &["ls-files"])
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect();
    let paths = paths.iter().map(|p| (*p).to_owned()).collect();
    plan(&root, &graph, &checks::all(), &files, &paths)
        .unwrap()
        .0
}

fn packages<'a>(plan: &'a Plan, id: &str) -> &'a BTreeMap<String, Features> {
    match plan.get(id) {
        Some(Scope::Packages(packages)) => packages,
        other => panic!("{id}: {other:?}"),
    }
}

fn with(features: &[&str]) -> Features {
    Features::Default(features.iter().map(|f| (*f).to_owned()).collect())
}

#[test]
fn a_store_change_runs_the_store_its_facade_feature_and_its_docker_suite_only() {
    let plan = planned(&["crates/rig-qdrant/src/lib.rs"]);
    let tests = packages(&plan, "tests");
    assert_eq!(tests.get("rig-qdrant"), Some(&Features::All));
    assert_eq!(tests.get("rig"), Some(&with(&["qdrant"])));
    for unrelated in ["rig-core", "rig-cassette", "rig-harness", "rig-lancedb"] {
        assert!(!tests.contains_key(unrelated), "{unrelated}: {tests:?}");
    }
    assert_eq!(
        packages(&plan, "service-tests").get("rig-service-tests"),
        Some(&with(&["qdrant"]))
    );
    for skipped in [
        "core-all",
        "conformance",
        "effect-corpus",
        "kernel-only",
        "default-check",
        "dependency-floors",
        "wasm-rig-core",
    ] {
        assert!(!plan.contains_key(skipped), "{skipped}");
    }
    // The facade guard builds every facade feature, the store's among them.
    assert_eq!(plan.get("guard-facade-features"), Some(&Scope::Full));
}

#[test]
fn a_falkordb_change_runs_the_store_facade_and_docker_suite() {
    let plan = planned(&["crates/rig-falkordb/src/lib.rs"]);
    let tests = packages(&plan, "tests");
    assert_eq!(tests.get("rig-falkordb"), Some(&Features::All));
    assert_eq!(tests.get("rig"), Some(&with(&["falkordb"])));
    assert_eq!(
        packages(&plan, "service-tests").get("rig-service-tests"),
        Some(&with(&["falkordb"]))
    );
}

#[test]
fn a_shared_input_or_a_path_in_no_package_runs_everything() {
    let full = full(&checks::all());
    for path in [
        "Cargo.lock",
        ".github/workflows/ci.yaml",
        ".config/nextest.toml",
        "xtask/src/main.rs",
        "test-support/rig-test-support/src/lib.rs",
        "LICENSE",
    ] {
        assert_eq!(planned(&[path]), full, "{path}");
    }
}

#[test]
fn documentation_outside_the_packages_runs_only_the_tree_guards() {
    let plan = planned(&["DEVELOPING.md", "docs/guide.md"]);
    let ids: Vec<&str> = plan.keys().map(String::as_str).collect();
    assert_eq!(
        ids,
        ["acceptance", "bank", "fmt", "source-guards", "tooling"]
    );
}

#[test]
fn a_member_manifest_runs_the_floors_and_the_compile_guards() {
    let plan = planned(&["crates/rig-qdrant/Cargo.toml"]);
    for id in [
        "dependency-floors",
        "guard-facade-features",
        "guard-dependency-rename",
    ] {
        assert_eq!(plan.get(id), Some(&Scope::Full), "{id}");
    }
}

#[test]
fn a_file_read_from_another_package_runs_the_readers_own_tests_only() {
    // rig-core's tests read the cassettes; its dependents do not.
    let plan = planned(&["crates/rig-cassette/fixtures/cassettes/openai/chat.yaml"]);
    let tests = packages(&plan, "tests");
    assert_eq!(tests.get("rig-core"), Some(&Features::All));
    assert_eq!(tests.get("rig-cassette"), Some(&Features::All));
    assert!(!tests.contains_key("rig-qdrant"), "{tests:?}");
}

#[test]
fn the_kernel_alone_runs_its_own_checks() {
    let kernel = planned(&["crates/rig-ecs/src/lib.rs"]);
    assert!(kernel.contains_key("kernel-only"));
    assert!(!kernel.contains_key("core-all"));
    assert!(!planned(&["crates/rig-qdrant/src/lib.rs"]).contains_key("kernel-only"));
}

#[test]
fn a_core_change_runs_the_isolated_graphs_and_every_store() {
    let plan = planned(&["crates/rig-core/src/lib.rs"]);
    for id in [
        "core-all",
        "conformance",
        "effect-corpus",
        "guard-macro-hygiene",
        "wasm-rig-core",
    ] {
        assert!(plan.contains_key(id), "{id}");
    }
    let Features::Default(stores) = packages(&plan, "service-tests")
        .get("rig-service-tests")
        .unwrap()
        .clone()
    else {
        panic!("the service suite runs with its store features");
    };
    for store in [
        "falkordb", "mongodb", "neo4j", "postgres", "qdrant", "sqlite",
    ] {
        assert!(stores.contains(store), "{store}: {stores:?}");
    }
}

#[test]
fn the_plan_round_trips_through_its_ci_encoding() {
    let plan = planned(&["crates/rig-qdrant/src/lib.rs"]);
    assert_eq!(decode(&encode(&plan)).unwrap(), plan);
}

/// Every job a check belongs to is a job of `ci.yaml` running
/// `cargo xtask verify --job <job>`, and the aggregator waits on every job.
#[test]
fn every_check_job_is_a_workflow_job_the_aggregator_needs() {
    let text = std::fs::read_to_string(root().join(".github/workflows/ci.yaml")).unwrap();
    let workflow: serde_yaml::Value = serde_yaml::from_str(&text).unwrap();
    let jobs = workflow.get("jobs").unwrap().as_mapping().unwrap();
    let needs: Vec<&str> = jobs
        .get("required")
        .and_then(|job| job.get("needs"))
        .and_then(|needs| needs.as_sequence())
        .unwrap()
        .iter()
        .filter_map(|need| need.as_str())
        .collect();
    for (name, _) in jobs {
        let name = name.as_str().unwrap();
        assert!(name == "required" || needs.contains(&name), "{name}");
    }
    for check in checks::all() {
        let job = checks::job(&check.id);
        let runs = jobs
            .get(job)
            .and_then(|job| job.get("steps"))
            .and_then(|steps| steps.as_sequence())
            .is_some_and(|steps| {
                steps.iter().any(|step| {
                    step.get("run").and_then(|run| run.as_str())
                        == Some(&format!("cargo xtask verify --job {job}"))
                })
            });
        assert!(runs, "{}: job {job} does not run it", check.id);
    }
    let macos = jobs.get("macos").and_then(|job| job.get("runs-on"));
    assert_eq!(macos.and_then(|r| r.as_str()), Some("macos-latest"));
}

#[test]
fn a_file_library_code_compiles_in_changes_that_package_and_its_dependents() {
    // rig-test-support embeds the image with `include_bytes!`; the cassette
    // suites that send it depend on rig-test-support.
    let plan = planned(&["tests/data/red_square.png"]);
    let tests = packages(&plan, "tests");
    assert_eq!(tests.get("rig-test-support"), Some(&Features::All));
    assert!(tests.contains_key("rig-cassette"), "{tests:?}");
}
