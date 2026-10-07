use super::*;

use std::collections::BTreeSet;

fn opts(mode: &str) -> Options {
    Options::parse(vec![mode.into(), "--base".into(), "HEAD".into()]).unwrap()
}

fn metadata() -> Value {
    // `rig` keeps the live-only provider targets; the cassette-backed ones
    // are targets of `rig-cassette`.
    serde_json::json!({"packages":[
        {"name":"rig","manifest_path":"/repo/Cargo.toml","targets":[{"name":"azure","kind":["test"]},{"name":"core","kind":["test"]}]},
        {"name":"rig-cassette","manifest_path":"/repo/crates/rig-cassette/Cargo.toml","dependencies":[],"targets":[{"name":"anthropic","kind":["test"]},{"name":"openai","kind":["test"]},{"name":"verify","kind":["test"]},{"name":"world_replay","kind":["test"]},{"name":"world_replay_world","kind":["test"]}]},
        {"name":"rig-cassette-minimal","manifest_path":"/repo/crates/rig-cassette/tests/minimal/Cargo.toml","dependencies":[],"targets":[{"name":"verify","kind":["test"]},{"name":"world_replay","kind":["test"]},{"name":"world_replay_world","kind":["test"]},{"name":"effect_log","kind":["test"]}]},
        {"name":"rig-ecs","manifest_path":"/repo/crates/rig-ecs/Cargo.toml","dependencies":[]},
        {"name":"rig-sqlite","manifest_path":"/repo/crates/rig-sqlite/Cargo.toml","dependencies":[]},
        {"name":"example","manifest_path":"/repo/examples/example/Cargo.toml","dependencies":[{"name":"rig-ecs"}]}
    ]})
}

fn ids(mode: &str, paths: &[&str]) -> BTreeSet<String> {
    selection::plan(
        Path::new("/repo"),
        &metadata(),
        &opts(mode),
        &paths.iter().map(|s| (*s).into()).collect(),
        &checks::all(),
    )
    .unwrap()
    .into_iter()
    .map(|c| c.id)
    .collect()
}

#[test]
fn pr_requires_explicit_base() {
    assert!(Options::parse(vec!["--pr".into()]).is_err());
}

#[test]
fn conflicting_modes_fail() {
    assert!(Options::parse(vec!["--changed".into(), "--full".into()]).is_err());
}

#[test]
fn unknown_changes_select_full_coverage() {
    assert_eq!(ids("--changed", &["unknown.xyz"]), ids("--full", &[]));
    assert_eq!(ids("--pr", &["unknown.xyz"]), ids("--full", &[]));
}

#[test]
fn gated_provider_changes_enable_every_capability() {
    let metadata = metadata();
    for path in [
        "crates/rig-cassette/tests/providers/openai/cassette/audio_params_matrix.rs",
        "crates/rig-cassette/tests/providers/openai/cassette/image_params_matrix.rs",
        "crates/rig-cassette/fixtures/cassettes/openai/websocket/example.yaml",
    ] {
        let plan = selection::plan(
            Path::new("/repo"),
            &metadata,
            &opts("--changed"),
            &BTreeSet::from([path.into()]),
            &checks::all(),
        )
        .unwrap();
        let provider = plan.iter().find(|c| c.id == "provider-openai").unwrap();
        assert!(provider.steps[0].args.contains(&"--all-features".into()));
        assert!(!provider.steps[0].args.contains(&"--features".into()));
        assert!(!plan.iter().any(|c| c.id == "full-tests"));
    }
}

#[test]
fn planner_and_dependency_edits_cannot_skip_checks() {
    for p in [
        "xtask/src/main.rs",
        "Cargo.lock",
        "crates/rig-ecs/Cargo.toml",
        "examples/agent/Cargo.toml",
        "rust-toolchain.toml",
        ".cargo/config.toml",
        ".github/actions/rust-setup/action.yml",
    ] {
        assert_eq!(ids("--changed", &[p]), ids("--full", &[]), "{p}");
    }
    // Shared runtime inputs broaden to every runtime check, but cannot
    // change what Cargo resolves, so the floors stay out.
    let mut without_floors = ids("--full", &[]);
    without_floors.remove("dependency-floors");
    for p in [
        ".config/nextest.toml",
        "src/lib.rs",
        "crates/rig-cassette/tests/common/mod.rs",
        "test-support/service-tests/src/lib.rs",
    ] {
        assert_eq!(ids("--changed", &[p]), without_floors, "{p}");
    }
}

#[test]
fn full_and_floor_lanes_select_independently() {
    // A storage-suite edit runs the service suites, not the floors.
    for p in [
        "crates/rig-sqlite/src/lib.rs",
        "tests/integrations/sqlite.rs",
    ] {
        let pr = ids("--pr", &[p]);
        assert!(pr.contains("full-tests"), "{p}: {pr:?}");
        assert!(!pr.contains("dependency-floors"), "{p}: {pr:?}");
    }
    // The floor checker runs the floors and its own tests, nothing else.
    let changed = ids("--changed", &["scripts/check-dependency-floors.py"]);
    assert_eq!(
        changed,
        BTreeSet::from(["dependency-floors".into(), "fmt".into(), "tooling".into()])
    );
    let pr = ids("--pr", &["scripts/check-dependency-floors.py"]);
    assert!(pr.contains("dependency-floors"));
    assert!(!pr.contains("full-tests"));
    // Resolver inputs run both.
    for p in ["Cargo.lock", "crates/rig-ecs/Cargo.toml"] {
        let pr = ids("--pr", &[p]);
        assert!(
            pr.contains("full-tests") && pr.contains("dependency-floors"),
            "{p}"
        );
    }
    // The lane predicates are exactly the CI path filters.
    for p in ["crates/rig-sqlite/src/lib.rs", "tests/integrations.rs"] {
        assert!(checks::full_lane(p) && !checks::floor_lane(p), "{p}");
    }
    for p in [
        "examples/agent/Cargo.toml",
        "rust-toolchain.toml",
        ".cargo/config.toml",
        "scripts/check-dependency-floors.py",
    ] {
        assert!(checks::floor_lane(p) && !checks::full_lane(p), "{p}");
    }
}

#[test]
fn fixture_selects_complete_provider_target() {
    let plan = ids(
        "--changed",
        &["crates/rig-cassette/fixtures/cassettes/anthropic/a.yaml"],
    );
    assert!(plan.contains("provider-anthropic"));
    assert!(!plan.contains("full-tests"));
}

/// Provider targets have two owners now. A moved cassette suite must be run
/// out of `rig-cassette` and a live-only suite out of the facade; picking the
/// wrong `-p` compiles a package that does not declare the target at all.
#[test]
fn a_provider_target_is_run_by_the_package_that_declares_it() {
    let args = |paths: &[&str], id: &str| {
        selection::plan(
            Path::new("/repo"),
            &metadata(),
            &opts("--changed"),
            &paths.iter().map(|s| (*s).into()).collect(),
            &checks::all(),
        )
        .unwrap()
        .into_iter()
        .find(|check| check.id == id)
        .unwrap()
        .steps[0]
            .args
            .clone()
    };
    let moved = args(
        &["crates/rig-cassette/tests/providers/openai/cassette/x.rs"],
        "provider-openai",
    );
    assert!(
        moved.windows(2).any(|p| p == ["-p", "rig-cassette"]),
        "{moved:?}"
    );
    assert!(
        moved.windows(2).any(|p| p == ["--test", "openai"]),
        "{moved:?}"
    );
    let stayed = args(&["tests/providers/azure/live.rs"], "provider-azure");
    assert!(stayed.windows(2).any(|p| p == ["-p", "rig"]), "{stayed:?}");
    // A cassette-package file that names no target is package source, not an
    // unknown provider: it must not broaden the whole plan.
    let corpus = ids("--changed", &["crates/rig-cassette/tests/corpus_hooks.rs"]);
    assert!(corpus.contains("package-rig-cassette"), "{corpus:?}");
    assert!(!corpus.contains("full-tests"), "{corpus:?}");
}

#[test]
fn shared_replay_sources_keep_the_minimal_execution() {
    for path in [
        "crates/rig-cassette/tests/corpus_hooks.rs",
        "crates/rig-cassette/tests/world_replay.rs",
        "crates/rig-cassette/tests/world_replay_world.rs",
        "crates/rig-cassette/src/effect_log/tests.rs",
        "crates/rig-cassette/src/agent/replay/tests.rs",
    ] {
        let plan = ids("--changed", &[path]);
        assert!(
            plan.contains("package-rig-cassette-minimal"),
            "{path}: {plan:?}"
        );
        assert!(!plan.contains("full-tests"), "{path}: {plan:?}");
    }
}

#[test]
fn unknown_fixture_provider_falls_back() {
    assert!(
        ids(
            "--changed",
            &["crates/rig-cassette/fixtures/cassettes/unknown/a.yaml"]
        )
        .contains("full-tests")
    );
}

#[test]
fn ecs_changes_include_reverse_consumers() {
    let p = ids("--changed", &["crates/rig-ecs/src/lib.rs"]);
    assert!(p.contains("package-rig-ecs"));
    assert!(p.contains("consumers"));
}

#[test]
fn pr_preserves_required_platform_and_default_guarantees() {
    let p = ids("--pr", &["README.md"]);
    for c in [
        "default-check",
        "default-tests",
        "wasm-rig-ecs",
        "wasm-rig-ecs-run_wasm",
        "native-only-rig-rmcp",
        "loom",
        "doctests",
        "conformance",
        "derive",
        "ecs-parity",
    ] {
        assert!(p.contains(c), "missing {c}");
    }
    assert!(!p.contains("full-tests"));
    assert!(ids("--pr", &["Cargo.lock"]).contains("full-tests"));
}

#[test]
fn unknown_check_cannot_succeed() {
    let o = Options::parse(vec!["--check".into(), "typo".into()]).unwrap();
    assert!(
        selection::plan(
            Path::new("/repo"),
            &metadata(),
            &o,
            &BTreeSet::new(),
            &checks::all()
        )
        .is_err()
    );
}

#[test]
fn check_ids_are_unique() {
    let all = checks::all();
    assert_eq!(
        all.len(),
        all.iter().map(|c| &c.id).collect::<BTreeSet<_>>().len()
    );
}

#[test]
fn telemetry_retry_policy_is_not_overridden() {
    let all = checks::all();
    let c = all.iter().find(|c| c.id == "core-all").unwrap();
    assert!(c.steps[0].args.contains(&"guards".into()));
    assert!(!c.steps[0].args.contains(&"--retries".into()));
}

#[test]
fn frozen_release_edits_include_untracked_and_working_changes() {
    assert!(
        selection::plan(
            Path::new("/repo"),
            &metadata(),
            &opts("--pr"),
            &BTreeSet::from(["docs/migrations/new.md".into()]),
            &checks::all()
        )
        .is_err()
    );
}

#[test]
fn browser_worker_changes_run_javascript_and_wasm_checks() {
    let p = ids(
        "--changed",
        &["examples/candle_wasm_chat/www/worker-runtime.mjs"],
    );
    assert!(p.contains("source-guards"));
    assert!(p.contains("wasm-candle_wasm_chat"));
}

#[test]
fn unmodeled_package_assets_broaden_instead_of_skipping() {
    assert_eq!(
        ids("--changed", &["crates/rig-ecs/generator.sh"]),
        ids("--full", &[])
    );
}

#[test]
fn reporting_only_selection_skips_work_but_other_docs_keep_executable_checks() {
    assert!(ids("--changed", &["DEVELOPING.md"]).is_empty());
    for path in ["docs/progress.md", "README.md", "crates/rig-ecs/README.md"] {
        assert!(!ids("--changed", &[path]).is_empty());
    }
    assert_eq!(ids("--pr", &["DEVELOPING.md"]), ids("--pr", &[]));
}

#[test]
fn verification_commands_force_replay_shape_matching_and_snapshot_checks_and_preserve_retry_contracts()
 {
    let step = Step::new("cargo", &["test"])
        .env("RIG_PROVIDER_TEST_MODE", "record")
        .env("RIG_CASSETTE_MATCHING", "exact")
        .env("RIG_REGENERATE_GOLDEN", "1")
        .env("RIG_CASSETTE_SNAPSHOTS", "write")
        .env("NEXTEST_RETRIES", "5");
    let command = execute::command(Path::new("/repo"), &step, Path::new("/repo/target"));
    let env: BTreeMap<_, _> = command.get_envs().collect();
    assert_eq!(
        env[std::ffi::OsStr::new("RIG_PROVIDER_TEST_MODE")],
        Some(std::ffi::OsStr::new("replay"))
    );
    assert_eq!(env[std::ffi::OsStr::new("RIG_REGENERATE_GOLDEN")], None);
    assert_eq!(
        env[std::ffi::OsStr::new("RIG_CASSETTE_SNAPSHOTS")],
        Some(std::ffi::OsStr::new("check"))
    );
    assert_eq!(
        env[std::ffi::OsStr::new("RIG_CASSETTE_MATCHING")],
        Some(std::ffi::OsStr::new("shape"))
    );
    assert_eq!(env[std::ffi::OsStr::new("NEXTEST_RETRIES")], None);
}

#[test]
fn cache_warming_compiles_the_test_graphs_without_claiming_test_execution() {
    let all = checks::all();
    for (alias, source) in [
        ("default-test-build", "default-tests"),
        ("full-test-build", "full-tests"),
    ] {
        let options = Options::parse(vec!["--check".into(), alias.into()]).unwrap();
        let plan = selection::plan(
            Path::new("/repo"),
            &metadata(),
            &options,
            &BTreeSet::new(),
            &all,
        )
        .unwrap();
        let executed = all.iter().find(|c| c.id == source).unwrap();
        let mut expected = executed.steps[0].clone();
        let index = expected.args.iter().position(|a| a == "--retries").unwrap();
        expected.args.drain(index..index + 2);
        expected.args.push("--no-run".into());
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].steps, vec![expected], "{alias}");
        assert_eq!(plan[0].id, alias);
        assert!(!all.iter().any(|c| c.id == alias));
        assert!(!executed.steps[0].args.contains(&"--no-run".into()));
        assert!(plan[0].reason.contains("executes no tests"));
    }
}

#[test]
fn conformance_compiles_exactly_its_executed_targets() {
    let all = checks::all();
    let c = all.iter().find(|c| c.id == "conformance").unwrap();
    let targets: BTreeSet<_> = c.steps[0]
        .args
        .windows(2)
        .filter(|w| w[0] == "--test")
        .map(|w| w[1].as_str())
        .collect();
    assert_eq!(
        targets,
        BTreeSet::from([
            "streaming_conformance",
            "streaming_conformance_websocket",
            "driver_adoption",
            "history_conformance"
        ])
    );
    assert!(c.steps[0].args.windows(2).any(|w| w == ["--retries", "0"]));
    assert!(c.steps[0].args.contains(&"--all-features".into()));
}

#[test]
fn full_covers_workspace_and_example_targets() {
    // full-tests owns the locked-version compile of every member's lib, bin,
    // test and example targets: Cargo's default `test` target selection,
    // which any narrowing flag would silently drop.
    let all = checks::all();
    let c = all.iter().find(|c| c.id == "full-tests").unwrap();
    for flag in ["--workspace", "--all-features"] {
        assert!(c.steps[0].args.contains(&flag.into()));
    }
    for flag in [
        "--lib",
        "--bins",
        "--bin",
        "--tests",
        "--test",
        "--examples",
        "--example",
        "--benches",
    ] {
        assert!(!c.steps[0].args.contains(&flag.into()), "{flag}");
    }
    assert!(!all.iter().any(|c| c.id == "workspace-check"));
    // Bench targets are outside cargo test's default selection. Each needs
    // an explicit locked compilation owner that does not execute it.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let metadata: Value = serde_json::from_str(
        &output(
            root,
            "cargo",
            &["metadata", "--locked", "--no-deps", "--format-version", "1"],
        )
        .expect("cargo metadata --no-deps for the bench guard"),
    )
    .unwrap();
    let packages = metadata["packages"].as_array().unwrap();
    assert!(packages.len() > 20, "workspace members not found");
    for package in packages {
        for bench in package["targets"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|target| {
                target["kind"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|kind| kind == "bench")
            })
        {
            let package_name = package["name"].as_str().unwrap();
            let bench_name = bench["name"].as_str().unwrap();
            let owned = all.iter().flat_map(|check| &check.steps).any(|step| {
                step.program == "cargo"
                    && step.args.first().is_some_and(|arg| arg == "check")
                    && step.args.iter().any(|arg| arg == "--locked")
                    && step
                        .args
                        .windows(2)
                        .any(|pair| pair == ["-p", package_name])
                    && step
                        .args
                        .windows(2)
                        .any(|pair| pair == ["--bench", bench_name])
            });
            assert!(
                owned,
                "{package_name} bench {bench_name} has no locked compilation owner"
            );
        }
    }
}

#[test]
fn prioritized_tests_exist() {
    // A renamed test would silently restore the all-features tail.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let config = std::fs::read_to_string(root.join(".config/nextest.toml")).unwrap();
    let block = config
        .split("[[profile.default.overrides]]")
        .skip(1)
        .find(|b| b.contains("priority = 100"))
        .expect("a default-profile override with priority 100");
    let config = block.split("\n[").next().unwrap();
    let sources: Vec<String> = execute::tracked_inputs(root)
        .unwrap()
        .into_iter()
        .filter(|p| p.ends_with(".rs") && p.contains("tests/"))
        .collect();
    let mut checked = 0;
    for line in config
        .lines()
        .filter(|l| l.trim_start().starts_with("filter = "))
    {
        for (kind, rest) in ["test(", "binary("].into_iter().flat_map(|kind| {
            line.match_indices(kind)
                .map(move |(i, _)| (kind, &line[i + kind.len()..]))
        }) {
            let name = rest.split(')').next().unwrap();
            if !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
                continue;
            }
            checked += 1;
            let found = if kind == "binary(" {
                sources
                    .iter()
                    .any(|p| p.ends_with(&format!("/tests/{name}.rs")))
            } else {
                sources.iter().any(|p| {
                    std::fs::read_to_string(root.join(p))
                        .is_ok_and(|text| text.contains(&format!("fn {name}(")))
                })
            };
            assert!(found, "{kind}{name}) names no tracked test");
        }
    }
    assert!(
        checked >= 2,
        "expected the prioritized nested-Cargo tests, found {checked}"
    );
}

#[test]
fn pr_broad_reason_names_verification_inputs() {
    let plan = selection::plan(
        Path::new("/repo"),
        &metadata(),
        &opts("--pr"),
        &BTreeSet::from(["xtask/src/verify.rs".into()]),
        &checks::all(),
    )
    .unwrap();
    assert!(
        plan.iter()
            .all(|c| c.reason.contains("xtask/src/verify.rs"))
    );
}

struct Repo(std::path::PathBuf);
impl Repo {
    fn new() -> Self {
        static SEQUENCE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "rig-planner-repo-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir(&path).unwrap();
        output(&path, "git", &["init", "--quiet"]).unwrap();
        let hooks = path.join(".git/empty-hooks");
        std::fs::create_dir(&hooks).unwrap();
        output(&path, "git", &["config", "commit.gpgsign", "false"]).unwrap();
        output(
            &path,
            "git",
            &["config", "core.hooksPath", hooks.to_str().unwrap()],
        )
        .unwrap();
        std::fs::write(path.join(".gitignore"), "target/\n").unwrap();
        std::fs::write(path.join("file.rs"), "original").unwrap();
        output(&path, "git", &["add", "."]).unwrap();
        output(
            &path,
            "git",
            &[
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "--quiet",
                "-m",
                "fixture",
            ],
        )
        .unwrap();
        Self(path)
    }
}

#[test]
fn staged_then_undone_and_untracked_inputs_are_not_lost() {
    let repo = Repo::new();
    std::fs::write(repo.0.join("file.rs"), "staged").unwrap();
    output(&repo.0, "git", &["add", "file.rs"]).unwrap();
    std::fs::write(repo.0.join("file.rs"), "original").unwrap();
    std::fs::write(repo.0.join("new.rs"), "new").unwrap();
    std::fs::create_dir_all(repo.0.join("target")).unwrap();
    std::fs::write(repo.0.join("target/out"), "ignored").unwrap();
    let paths = selection::changes(&repo.0, &opts("--changed")).unwrap();
    assert!(paths.contains("file.rs"));
    assert!(paths.contains("new.rs"));
    assert!(!paths.iter().any(|p| p.starts_with("target")));
}

fn fake_check(id: &str, script: &str) -> Check {
    Check {
        id: id.into(),
        reason: "fake execution".into(),
        steps: vec![Step::new("bash", &["-c", script])],
    }
}

#[test]
fn failed_required_command_stops_later_checks() {
    let repo = Repo::new();
    let metadata = serde_json::json!({"target_directory": repo.0.join("target")});
    let later = fake_check("later", "touch target/later");
    std::fs::create_dir_all(repo.0.join("target")).unwrap();
    let error = execute::run(
        &repo.0,
        &metadata,
        &[fake_check("first", "exit 3"), later.clone()],
    )
    .unwrap_err();
    assert!(
        error.to_string().contains("required check first failed"),
        "{error}"
    );
    assert!(!repo.0.join("target/later").exists());
    execute::run(&repo.0, &metadata, &[later]).unwrap();
    assert!(repo.0.join("target/later").exists());
}

#[test]
fn missing_prerequisite_stops_before_any_check() {
    let repo = Repo::new();
    let metadata = serde_json::json!({"target_directory": repo.0.join("target")});
    let first = fake_check("first", "touch target/executed");
    let missing = Check {
        id: "missing".into(),
        reason: "test".into(),
        steps: vec![Step::new("rig-nonexistent-tool-for-test", &[])],
    };
    let error = execute::run(&repo.0, &metadata, &[first, missing]).unwrap_err();
    assert!(error.to_string().contains("no checks executed"));
    assert!(error.to_string().contains("rig-nonexistent-tool-for-test"));
    assert!(!repo.0.join("target/executed").exists());
}

#[test]
fn lanes_mode_reports_the_pr_plan_lanes() {
    for (path, full, floors) in [
        ("README.md", false, false),
        ("tests/integrations/sqlite.rs", true, false),
        ("scripts/check-dependency-floors.py", false, true),
        ("Cargo.lock", true, true),
    ] {
        let plan = selection::plan(
            Path::new("/repo"),
            &metadata(),
            &Options::parse(vec!["--lanes".into(), "--base".into(), "HEAD".into()]).unwrap(),
            &BTreeSet::from([path.into()]),
            &checks::all(),
        )
        .unwrap();
        assert_eq!(plan.iter().any(|c| c.id == "full-tests"), full, "{path}");
        assert_eq!(
            plan.iter().any(|c| c.id == "dependency-floors"),
            floors,
            "{path}"
        );
    }
}

#[test]
fn full_lane_covers_every_integration_suite() {
    // The service suites are the only runtime coverage the storage crates
    // have; a suite whose crate is not a full-lane trigger merges its crate
    // with none. bedrock's suite is feature-gated inside the facade, which
    // the PR gate's `--features bedrock` sweep already runs.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let mut suites = BTreeSet::new();
    for entry in std::fs::read_dir(root.join("tests/integrations")).unwrap() {
        let path = entry.unwrap().path();
        let suite = path.file_stem().unwrap().to_string_lossy().into_owned();
        if suite != "bedrock" {
            assert!(
                checks::full_lane(&format!("crates/rig-{suite}/src/lib.rs")),
                "crates/rig-{suite} is not a full-lane trigger"
            );
            suites.insert(suite);
        }
    }
    assert!(suites.len() >= 8, "{suites:?}");
    for crate_dir in std::fs::read_dir(root.join("crates")).unwrap() {
        let name = crate_dir
            .unwrap()
            .file_name()
            .to_string_lossy()
            .into_owned();
        let source = format!("crates/{name}/src/lib.rs");
        let suite = name.strip_prefix("rig-").unwrap_or(&name);
        assert_eq!(
            checks::full_lane(&source),
            suites.contains(suite),
            "{name}: full-lane trigger and integration suite must agree"
        );
    }
}

impl Drop for Repo {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn lanes_mode_does_not_enforce_the_release_document_freeze() {
    let lanes = Options::parse(vec!["--lanes".into(), "--base".into(), "HEAD".into()]).unwrap();
    let paths = BTreeSet::from(["CHANGELOG.md".into()]);
    assert!(
        selection::plan(
            Path::new("/repo"),
            &metadata(),
            &lanes,
            &paths,
            &checks::all()
        )
        .is_ok()
    );
    assert!(
        selection::plan(
            Path::new("/repo"),
            &metadata(),
            &opts("--pr"),
            &paths,
            &checks::all()
        )
        .is_err()
    );
}

#[test]
fn tool_versions_require_the_exact_pinned_version() {
    assert!(preflight::version_matches(
        "rustc 1.95.0 (abc 2026-04-14)",
        "1.95.0"
    ));
    assert!(!preflight::version_matches(
        "rustc 1.95.0 (abc 2026-04-14)",
        "1.95"
    ));
    assert!(preflight::version_matches(
        "wasm-bindgen-test-runner 0.2.118",
        "0.2.118"
    ));
    assert!(!preflight::version_matches(
        "wasm-bindgen-test-runner 0.2.126",
        "0.2.118"
    ));
}

/// Every `cargo xtask verify --check <id>` a workflow step runs, by line
/// scan of the `run:` values (the workflows are plain YAML lists).
fn workflow_check_ids(workflow: &str) -> BTreeSet<String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    std::fs::read_to_string(root.join(workflow))
        .unwrap()
        .lines()
        .filter_map(|line| {
            line.trim()
                .trim_start_matches("- ")
                .strip_prefix("run: cargo xtask verify --check ")
        })
        .map(|id| id.trim().to_string())
        .collect()
}
#[test]
fn every_check_is_run_by_a_workflow_step() {
    // A check that exists only in `checks::all()` would pass `--full`
    // locally and run nowhere in CI. This asserts presence in a step, not
    // reachability: the slow lanes' steps sit behind the plan job's answer
    // and the repository guard by design.
    let defined: BTreeSet<String> = checks::all().into_iter().map(|c| c.id).collect();
    let mut invoked = workflow_check_ids(".github/workflows/ci.yaml");
    invoked.extend(workflow_check_ids(".github/workflows/slow.yaml"));
    let missing: Vec<_> = defined.difference(&invoked).collect();
    assert!(missing.is_empty(), "checks no workflow runs: {missing:?}");
    let unknown: Vec<_> = invoked.difference(&defined).collect();
    assert!(
        unknown.is_empty(),
        "workflows run undefined checks: {unknown:?}"
    );
    let warm = workflow_check_ids(".github/workflows/cache-warm.yaml");
    assert_eq!(
        warm,
        selection::WARMING
            .iter()
            .map(|(alias, _)| (*alias).to_string())
            .collect()
    );
}

#[test]
fn workflows_carry_no_toolchain_copy() {
    // rust-toolchain.toml is the single toolchain source. A literal
    // `toolchain:` or a RUST_VERSION copy in any workflow or action would
    // build one job on a stale compiler; jobs that run no check (release-plz,
    // the plan job) never reach the planner's rustc probe, so scan the text.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let mut files: Vec<_> = std::fs::read_dir(root.join(".github/workflows"))
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .collect();
    files.push(root.join(".github/actions/rust-setup/action.yml"));
    for file in files {
        let text = std::fs::read_to_string(&file).unwrap();
        for line in text.lines() {
            let trimmed = line.trim();
            assert!(
                !trimmed.starts_with("RUST_VERSION:"),
                "{}: RUST_VERSION copy",
                file.display()
            );
            if let Some(value) = trimmed.strip_prefix("toolchain:") {
                assert!(
                    value.contains("${{"),
                    "{}: literal toolchain pin {value:?}",
                    file.display()
                );
            }
        }
    }
}

#[test]
fn a_golden_change_selects_the_parity_lane() {
    let p = ids(
        "--changed",
        &["crates/rig-cassette/fixtures/effects/x.effects.json"],
    );
    for check in ["bus-verification", "default-tests", "ecs-parity"] {
        assert!(p.contains(check), "{p:?}");
    }
}

#[test]
fn world_goldens_select_the_same_lanes_as_agent_goldens() {
    let agent = ids(
        "--changed",
        &["crates/rig-cassette/fixtures/effects/x.effects.json"],
    );
    let world = ids(
        "--changed",
        &["crates/rig-cassette/fixtures/effects/world/x.effects.json"],
    );
    assert_eq!(world, agent);
    for check in ["bus-verification", "default-tests", "ecs-parity"] {
        assert!(world.contains(check), "{world:?}");
    }
}

/// The two corpora share a parent directory inside one package. A provider
/// cassette must still reach its provider target instead of falling through
/// to the owning package's generic asset rule, and neither corpus may be
/// classified as the other.
#[test]
fn the_two_corpora_are_classified_apart() {
    let cassette = ids(
        "--changed",
        &["crates/rig-cassette/fixtures/cassettes/anthropic/a.yaml"],
    );
    assert!(cassette.contains("provider-anthropic"), "{cassette:?}");
    assert!(!cassette.contains("bus-verification"), "{cassette:?}");
    assert!(!cassette.contains("package-rig-cassette"), "{cassette:?}");
    let effects = ids(
        "--changed",
        &["crates/rig-cassette/fixtures/effects/x.effects.json"],
    );
    assert!(!effects.iter().any(|id| id.starts_with("provider-")));
}

/// The fixture metadata declares `rig-ecs` (a runtime crate) and
/// `rig-sqlite` (a store): an edit to the first selects the parity lane on
/// its own account, an edit to the second does not.
#[test]
fn a_runtime_crate_change_selects_the_parity_lane() {
    let p = ids("--changed", &["crates/rig-ecs/src/replay/mod.rs"]);
    assert!(p.contains("ecs-parity"), "{p:?}");
    let p = ids("--changed", &["crates/rig-sqlite/src/lib.rs"]);
    assert!(!p.contains("ecs-parity"), "{p:?}");
}

#[test]
fn default_check_compiles_extracted_regressions_without_extra_features() {
    let all = checks::all();
    let check = all
        .iter()
        .find(|check| check.id == "default-check")
        .unwrap();
    let args = &check.steps[0].args;
    for package in ["rig", "rig-test-support"] {
        assert!(args.windows(2).any(|pair| pair == ["-p", package]));
    }
    assert!(args.iter().any(|arg| arg == "--tests"));
    for flag in ["--features", "--all-features", "--no-default-features"] {
        assert!(!args.iter().any(|arg| arg == flag), "{flag}");
    }
}

#[test]
fn the_coverage_check_runs_the_cheap_gate_without_mutation() {
    let all = checks::all();
    let c = all.iter().find(|c| c.id == "coverage").unwrap();
    assert_eq!(
        c.steps,
        vec![Step::new("cargo", &["xtask", "coverage", "--check"])]
    );
}

#[test]
fn cassettes_snapshots_and_the_index_select_the_acceptance_check() {
    for path in [
        "crates/rig-cassette/fixtures/cassettes/openai/chat.yaml",
        "crates/rig-cassette/fixtures/cassettes/openai/chat.requests.json",
        "crates/rig-cassette/fixtures/acceptance.toml",
    ] {
        assert!(ids("--changed", &[path]).contains("acceptance"), "{path}");
    }
    assert!(!ids("--changed", &["crates/rig-ecs/src/lib.rs"]).contains("acceptance"));
}

#[test]
fn cassettes_and_the_bank_select_the_bank_check() {
    for path in [
        "crates/rig-cassette/fixtures/cassettes/openai/chat.yaml",
        "crates/rig-cassette/fixtures/bank/openai.yaml",
        "crates/rig-cassette/fixtures/bank/scripts.tsv",
    ] {
        assert!(ids("--changed", &[path]).contains("bank"), "{path}");
    }
    let bank = ids(
        "--changed",
        &["crates/rig-cassette/fixtures/bank/pinned.txt"],
    );
    assert!(bank.contains("provider-runtime"), "{bank:?}");
    assert!(bank.contains("coverage"), "{bank:?}");
    assert!(!bank.contains("full-tests"), "{bank:?}");
    assert!(!ids("--changed", &["crates/rig-ecs/src/lib.rs"]).contains("bank"));
}

#[test]
fn sources_cassettes_and_the_baseline_select_the_coverage_gate() {
    for path in [
        "crates/rig-ecs/src/lib.rs",
        "crates/rig-cassette/tests/corpus_hooks.rs",
        "crates/rig-cassette/fixtures/cassettes/openai/websocket/example.yaml",
        "crates/rig-cassette/coverage/lines.tsv",
    ] {
        assert!(ids("--changed", &[path]).contains("coverage"), "{path}");
    }
    let baseline = ids("--changed", &["crates/rig-cassette/coverage/shapes.tsv"]);
    assert!(!baseline.contains("full-tests"), "{baseline:?}");
    assert!(!ids("--changed", &["README.md"]).contains("coverage"));
}

/// The findings of the typed-options guards in `source`, as a listed
/// completion-wire file.
fn guarded(source: &str) -> Vec<String> {
    super::guards::offenders("crates/rig-core/src/providers/openai/wire/chat.rs", source)
        .unwrap_or_else(|error| panic!("the source parses: {error}"))
}

#[test]
fn an_option_fields_pattern_must_name_every_field() {
    for source in [
        "use crate::completion::options::OptionFields;
         fn map(fields: OptionFields<'_>) { let OptionFields { reasoning, .. } = fields; }",
        "use crate::completion::options::{OptionFields as F};
         fn map(fields: F<'_>) { let F { reasoning, .. } = fields; }",
        "use crate::completion::options::OptionFields;
         fn map(fields: OptionFields<'_>) { let OptionFields { stop: _stop, reasoning } = fields; }",
        "use crate::completion::options::OptionFields;
         fn map(fields: OptionFields<'_>) { let OptionFields { stop: _, reasoning } = fields; }",
    ] {
        let findings = guarded(source);
        assert_eq!(findings.len(), 1, "{source}: {findings:?}");
        assert!(findings[0].starts_with("options-mapping: "), "{findings:?}");
    }
    assert!(
        guarded(
            "use crate::completion::options::OptionFields;
             fn map(fields: OptionFields<'_>) { let OptionFields { reasoning, stop } = fields; }"
        )
        .is_empty()
    );
}

#[test]
fn an_option_map_literal_takes_no_base() {
    let findings = guarded(
        "use crate::completion::options::OptionMap as Answers;
         fn map(base: Answers) -> Answers { Answers { seed: Mapping::Nothing, ..base } }",
    );
    assert_eq!(findings.len(), 1, "{findings:?}");
    assert!(findings[0].contains("'..'"), "{findings:?}");
}

#[test]
fn a_wire_neither_builds_nor_rewrites_a_requests_options() {
    for source in [
        "fn encode(request: CompletionRequest) { let copy = CompletionRequest::new(\"hi\"); }",
        "fn encode(request: CompletionRequest) { let copy = request.options(GenerationOptions::default()); }",
        "fn encode(mut request: CompletionRequest) { request.options = GenerationOptions::default(); }",
        "fn encode(request: &CompletionRequest) -> bool { request.options.seed.is_some() }",
    ] {
        let findings = guarded(source);
        assert!(
            findings
                .iter()
                .all(|finding| finding.starts_with("options-mapping: ")),
            "{source}: {findings:?}"
        );
        assert!(!findings.is_empty(), "{source}");
    }
}

#[test]
fn a_mapping_cannot_defer_to_a_raw_key() {
    let findings = guarded(
        "use crate::completion::options::{self, OptionFields};
         fn map(request: &CompletionRequest, fields: OptionFields<'_>) {
             let OptionFields { reasoning, stop } = fields;
             let raw = options::param(self, request, \"thinking\");
         }",
    );
    assert_eq!(findings.len(), 1, "{findings:?}");
    assert!(findings[0].contains("options::param"), "{findings:?}");
    assert!(
        guarded(
            "use crate::completion::options;
             fn continues_stored(request: &CompletionRequest) -> bool {
                 options::param(self, request, \"store\").is_some()
             }"
        )
        .is_empty()
    );
}

#[test]
fn a_wire_reads_no_raw_params_and_builds_no_body_of_its_own() {
    for source in [
        "fn encode(request: CompletionRequest) { let raw = request.additional_params.clone(); }",
        "use crate::wire::Body;
         fn encode(body: Value) -> Body { Body::Bytes(serde_json::to_vec(&body).unwrap_or_default()) }",
        "use crate::wire::{Body as B};
         fn encode(form: Form) -> B { B::Multipart(form) }",
        "fn encode(builder: Builder, body: Vec<u8>) { builder.body(body); }",
        "fn encode(body: FinalBody) { let copy = body.deserialize::<serde_json::Value>(); }",
        "fn encode(request: &CompletionRequest) { let bytes = serde_json::to_vec(request); }",
    ] {
        let findings = guarded(source);
        assert!(
            findings.iter().any(|finding| finding.starts_with("options-precedence: ")),
            "{source}: {findings:?}"
        );
    }
    for source in [
        "fn encode(builder: Builder, body: FinalBody) { builder.body(body.into_body()); }",
        "fn encode(builder: Builder) { builder.body(crate::wire::Body::empty()); }",
    ] {
        assert!(guarded(source).is_empty(), "{source}");
    }
    // A document part's own `additional_params` is another field, allowed
    // where a wire reads it.
    let document =
        "fn part(document: &Document) { let params = document.additional_params.as_ref(); }";
    let anthropic = super::guards::offenders(
        "crates/rig-core/src/providers/anthropic/completion.rs",
        document,
    );
    assert_eq!(anthropic, Ok(Vec::new()));
    assert_eq!(guarded(document).len(), 1);
}

/// Whether `source`, as a listed completion-wire file, fails the precedence
/// guard.
fn fails_precedence(source: &str) -> bool {
    guarded(source)
        .iter()
        .any(|finding| finding.starts_with("options-precedence: "))
}

#[test]
fn a_wire_opens_no_body_in_a_pattern() {
    for source in [
        "fn rewrite(body: &mut crate::wire::Body) {
             if let crate::wire::Body::Bytes(bytes) = body { bytes.clear(); }
         }",
        "use crate::wire::{Body as B};
         fn rewrite(body: B) -> usize { match body { B::Multipart(_) => 0, _ => 1 } }",
        "fn rewrite(body: &crate::wire::Body) -> bool { matches!(body, crate::wire::Body::Bytes(_)) }",
        "use crate::wire::Body;
         type Sent = Body;
         fn rewrite(body: Sent) -> bool { if let Sent::Bytes(_) = body { true } else { false } }",
    ] {
        assert!(fails_precedence(source), "{source}");
    }
    assert!(!fails_precedence(
        "use crate::wire::Body;
         fn send(body: FinalBody) -> Body { body.into_body() }"
    ));
}

#[test]
fn a_wire_writes_to_no_built_body() {
    assert!(fails_precedence(
        "fn rewrite(mut built: http::Request<Body>) { let body = built.body_mut(); }"
    ));
}

#[test]
fn a_wire_destructures_no_raw_params() {
    for source in [
        "fn encode(request: CompletionRequest) { let CompletionRequest { additional_params, .. } = request; }",
        "fn encode(request: CompletionRequest) { let CompletionRequest { additional_params: raw, .. } = request; }",
        "fn encode(requests: Vec<CompletionRequest>) { requests.into_iter().map(|CompletionRequest { additional_params, .. }| additional_params); }",
    ] {
        assert!(fails_precedence(source), "{source}");
    }
    // The allowlist holds for a pattern as for a field read: a document
    // part's own `additional_params` is another field.
    let document =
        "fn part(document: Document) { let Document { additional_params, .. } = document; }";
    let anthropic = super::guards::offenders(
        "crates/rig-core/src/providers/anthropic/completion.rs",
        document,
    );
    assert_eq!(anthropic, Ok(Vec::new()));
    assert!(fails_precedence(document));
}

#[test]
fn a_glob_import_of_body_resolves() {
    for source in [
        "use crate::wire::Body::*;
         fn rewrite(body: &mut crate::wire::Body) { if let Bytes(bytes) = body { bytes.clear(); } }",
        "use crate::wire::Body::*;
         fn encode(bytes: Vec<u8>) -> crate::wire::Body { Bytes(bytes) }",
        "use crate::wire::{Body as B};
         use B::*;
         fn encode(form: Form) -> B { Multipart(form) }",
    ] {
        assert!(fails_precedence(source), "{source}");
    }
    // A glob of another module does not make every `Bytes` a body.
    assert!(!fails_precedence(
        "use crate::types::*;
         fn size(bytes: Bytes) -> usize { bytes.len() }"
    ));
}

/// `Self` in an `impl` of `Body`, a qualified `<Body>::X` and a macro that
/// splices a variant after `Body` all reach `Body::Bytes` without naming it.
#[test]
fn a_body_reached_without_its_variant_path_fails() {
    for source in [
        "impl crate::wire::Body {
             pub(crate) fn raw(bytes: Vec<u8>) -> Self { Self::Bytes(bytes) }
         }",
        "use crate::wire::Body;
         impl From<Vec<u8>> for Body { fn from(bytes: Vec<u8>) -> Self { Self::Bytes(bytes) } }",
        "fn encode(bytes: Vec<u8>) -> crate::wire::Body { <crate::wire::Body>::Bytes(bytes) }",
        "use crate::wire::{Body as B};
         fn encode(form: Form) -> B { <B>::Multipart(form) }",
        "macro_rules! b { ($v:ident, $x:expr) => { crate::wire::Body::$v($x) } }
         fn encode(bytes: Vec<u8>) -> crate::wire::Body { b!(Bytes, bytes) }",
        "macro_rules! b { ($t:path, $x:expr) => { $t($x) } }
         fn encode(bytes: Vec<u8>) { b!(crate::wire::Body, bytes); }",
    ] {
        assert!(fails_precedence(source), "{source}");
    }
    // A macro that names `Body::empty()` sends no body, and another type's
    // `impl` is not `Body`'s.
    for source in [
        "fn send(builder: Builder) { builder.body(crate::wire::Body::empty()); }",
        "macro_rules! on_route { ($c:expr) => { match $c { Self::Chat(w) => w } } }",
        "impl Payload { fn bytes(&self) -> usize { 0 } }",
    ] {
        assert!(!fails_precedence(source), "{source}");
    }
}

#[test]
fn test_items_are_not_guarded() {
    assert!(
        guarded(
            "#[cfg(test)]
             mod tests { fn build() { let raw = request.additional_params.clone(); } }"
        )
        .is_empty()
    );
}

#[test]
fn a_file_with_a_completion_wire_is_found() {
    use super::guards::holds_a_completion_wire;
    let wire = "impl Wire for Chat { type Op = crate::operation::Completion; }";
    assert_eq!(holds_a_completion_wire(wire), Ok(true));
    let target = "impl crate::completion::ReplayTarget for Chat {}";
    assert_eq!(holds_a_completion_wire(target), Ok(true));
    let embeddings = "impl Wire for Embeddings { type Op = crate::operation::Embed; }";
    assert_eq!(holds_a_completion_wire(embeddings), Ok(false));
    let tested = "#[cfg(test)] impl ReplayTarget for Fake {}";
    assert_eq!(holds_a_completion_wire(tested), Ok(false));
}

#[test]
fn a_wire_reads_and_sets_no_provider_options() {
    for source in [
        "fn encode(request: &CompletionRequest) -> bool { request.provider_options.is_empty() }",
        "fn encode(request: CompletionRequest) { let CompletionRequest { provider_options, .. } = request; }",
        "fn encode(request: CompletionRequest) { let copy = request.provider_options(ProviderOptions::new()); }",
        "fn encode(mut request: CompletionRequest) { request.provider_options = ProviderOptions::new(); }",
    ] {
        assert!(fails_precedence(source), "{source}");
    }
}

/// The findings of the `extras-off-decode-path` guard in `source`, as the
/// decode-path file `file`.
fn decoding(file: &str, source: &str) -> Vec<String> {
    super::extras::offenders(file, source, true)
        .unwrap_or_else(|error| panic!("the source parses: {error}"))
}

const CHAT: &str = "crates/rig-core/src/providers/openai/wire/chat.rs";

#[test]
fn a_decoder_names_no_extension_module_in_any_import_form() {
    for source in [
        "use crate::providers::openrouter::{extension as ext};",
        "use crate::providers::openrouter::{self, extension::*};",
        "use super::super::openrouter::extension;",
        "use crate::providers::openrouter::extension::OpenRouterExtras as Extras;",
        "use crate::providers::openrouter as or;
         fn read(extras: or::extension::OpenRouterExtras) {}",
        "use crate::providers::*;
         fn read(extras: openrouter::extension::OpenRouterExtras) {}",
        "type Extras = crate::providers::openrouter::extension::OpenRouterExtras;",
        "fn read(reply: &CompletionResponse) {
             let extras = reply.extras::<crate::providers::openrouter::extension::OpenRouterExt>();
         }",
        "macro_rules! extras { ($i:ident) => { crate::providers::openrouter::extension::$i } }",
        "macro_rules! extras { ($i:ident) => { super::extension::$i } }",
        "pub use super::extension::OpenRouterExt;",
    ] {
        let findings = decoding(CHAT, source);
        assert!(!findings.is_empty(), "{source}");
        assert!(
            findings
                .iter()
                .all(|finding| finding.starts_with("extras-off-decode-path: ")),
            "{findings:?}"
        );
    }
}

#[test]
fn a_decoder_names_no_extension_trait_in_any_import_form() {
    for source in [
        "use crate::completion::provider_options::ProviderExtension as Ext;",
        "use crate::completion::{ReplyExtras, CompletionResponse};",
        "use crate::completion::provider_options::{self as po};
         fn read<P: po::ProviderExtension>(raw: &Value) {}",
        "use crate::completion::*;
         fn read<P: ProviderExtension>(raw: &Value) {}",
        "use crate::completion::*;
         impl ReplyExtras for Usage {}",
    ] {
        assert!(!decoding(CHAT, source).is_empty(), "{source}");
    }
    // An alias is followed to its use.
    let aliased = decoding(
        CHAT,
        "use crate::completion::provider_options::ProviderExtension as Ext;
         fn read<P: Ext>(raw: &Value) {}",
    );
    assert_eq!(aliased.len(), 2, "{aliased:?}");
}

#[test]
fn a_decoder_writes_no_citation_past_the_fold() {
    for source in [
        "use crate::message::Citation;
         fn cite() -> Citation { Citation { span: None, sources: Vec::new() } }",
        "use crate::message::{citation::Span as S};
         fn span() -> S { S { start: 0, end: 4 } }",
        "use crate::message::Text;
         fn cite(text: Text) -> Text { Text::with_citations(text, []) }",
        "use crate::message::Text as T;
         fn span(text: &T) { let _ = T::span(text, 0..4); }",
        "fn spans(texts: &[crate::message::Text]) { texts.iter().map(crate::message::Text::span); }",
        "fn cite(text: Text) -> Text { text.with_citations([]) }",
        "fn span(text: &Text) { let _ = text.span(0..4); }",
        "use crate::completion::message::citation;
         fn cite(text: &mut Text) { citation::attach(text, Vec::new(), Vec::new(), \"p\", 0); }",
    ] {
        let findings = decoding(CHAT, source);
        assert!(!findings.is_empty(), "{source}");
        assert!(
            findings
                .iter()
                .all(|finding| finding.starts_with("extras-off-decode-path: ")),
            "{findings:?}"
        );
    }
    // Handing a citation to the fold, or a tracing span, passes.
    for source in [
        "use crate::wire::{SpanUnit, WireCitation, WireSpan};
         fn cite(out: &mut Out<'_, Completion>) {
             out.cite(0, WireCitation::new(Some(WireSpan::new(0, 4, SpanUnit::Chars)), Vec::new()));
         }",
        "fn trace() { let span = tracing::Span::current(); let _ = span.id(); }",
    ] {
        assert_eq!(decoding(CHAT, source), Vec::<String>::new(), "{source}");
    }
}

#[test]
fn a_companion_crates_extension_is_off_its_decode_path_too() {
    let findings = decoding(
        "crates/rig-bedrock/src/completion.rs",
        "use crate::extension::BedrockExt;",
    );
    assert_eq!(findings.len(), 1, "{findings:?}");
    assert!(
        findings[0].contains("crates/rig-bedrock/src/completion.rs:1"),
        "{findings:?}"
    );
    use super::extras::in_extension_module;
    for file in [
        "crates/rig-bedrock/src/extension.rs",
        "crates/rig-bedrock/src/extension/options.rs",
        "crates/rig-core/src/providers/openrouter/extension.rs",
        "crates/rig-core/src/providers/openai/extension/responses.rs",
    ] {
        assert!(in_extension_module(file), "{file}");
    }
    assert!(!in_extension_module(
        "crates/rig-core/src/providers/openrouter/mod.rs"
    ));
}

#[test]
fn what_the_decode_path_may_still_write() {
    for source in [
        "pub mod extension;",
        "mod extension {
             use crate::completion::{ProviderExtension, ReplyExtras};
             pub struct OpenRouter;
         }",
        "fn suffix(path: &std::path::Path) -> bool {
             let extension = path.extension();
             extension.is_some()
         }",
        "#[cfg(test)]
         mod tests { use super::extension::OpenRouterExt; }",
        "use crate::completion::{CompletionResponse, ProviderOptions};",
    ] {
        assert_eq!(decoding(CHAT, source), Vec::<String>::new(), "{source}");
    }
}

#[test]
fn an_extension_item_is_re_exported_nowhere() {
    let reexports = |source: &str| {
        super::extras::offenders("src/lib.rs", source, false)
            .unwrap_or_else(|error| panic!("the source parses: {error}"))
    };
    for source in [
        "pub use rig_core::providers::openrouter::extension::OpenRouterExt;",
        "pub use rig_core::providers::openrouter::{extension::*};",
        "pub(crate) use rig_core::providers::openrouter::extension;",
        "pub type OpenRouter = rig_core::providers::openrouter::extension::OpenRouterExt;",
    ] {
        assert_eq!(reexports(source).len(), 1, "{source}");
    }
    for source in [
        "use rig_core::providers::openrouter::extension::OpenRouterExt;",
        "pub use rig_core::completion::{ProviderExtension, ReplyExtras};",
        "pub use rig_core::providers;",
    ] {
        assert!(reexports(source).is_empty(), "{source}");
    }
}
