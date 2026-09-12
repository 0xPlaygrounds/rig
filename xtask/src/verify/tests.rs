use super::*;

use std::collections::BTreeSet;

fn opts(mode: &str) -> Options {
    Options::parse(vec![mode.into(), "--base".into(), "HEAD".into()]).unwrap()
}

fn metadata() -> Value {
    serde_json::json!({"packages":[{"name":"rig","manifest_path":"/repo/Cargo.toml","targets":[{"name":"anthropic","kind":["test"]}]},{"name":"rig-ecs","manifest_path":"/repo/crates/rig-ecs/Cargo.toml","dependencies":[]},{"name":"rig-sqlite","manifest_path":"/repo/crates/rig-sqlite/Cargo.toml","dependencies":[]},{"name":"example","manifest_path":"/repo/examples/example/Cargo.toml","dependencies":[{"name":"rig-ecs"}]}]})
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
    let mut metadata = metadata();
    metadata["packages"][0]["targets"] = serde_json::json!([{"name":"openai","kind":["test"]}]);
    for path in [
        "tests/providers/openai/cassette/audio_params_matrix.rs",
        "tests/providers/openai/cassette/image_params_matrix.rs",
        "tests/cassettes/openai/websocket/example.yaml",
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
        "tests/common/mod.rs",
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
    let plan = ids("--changed", &["tests/cassettes/anthropic/a.yaml"]);
    assert!(plan.contains("provider-anthropic"));
    assert!(!plan.contains("full-tests"));
}

#[test]
fn unknown_fixture_provider_falls_back() {
    assert!(ids("--changed", &["tests/cassettes/unknown/a.yaml"]).contains("full-tests"));
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
fn registration_discovery_reuses_default_test_graph_without_filtering() {
    let all = checks::all();
    for id in ["default-tests", "scenario-registrations"] {
        let args = &all.iter().find(|c| c.id == id).unwrap().steps[0].args;
        for flag in [
            "-p",
            "--package",
            "--workspace",
            "--all-features",
            "--no-default-features",
        ] {
            assert!(
                !args.iter().any(|arg| arg == flag),
                "{id}: different graph via {flag}"
            );
        }
        assert!(
            args.windows(2)
                .any(|args| args == ["--features", "bedrock"])
        );
        if id == "scenario-registrations" {
            for flag in ["-E", "--filter-expr", "--test", "--lib"] {
                assert!(
                    !args.iter().any(|arg| arg == flag),
                    "filtered listing via {flag}"
                );
            }
        }
    }
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
fn verification_commands_force_replay_and_preserve_retry_contracts() {
    let step = Step::new("cargo", &["test"])
        .env("RIG_PROVIDER_TEST_MODE", "record")
        .env("RIG_REGENERATE_GOLDEN", "1")
        .env("NEXTEST_RETRIES", "5");
    let command = execute::command(Path::new("/repo"), &step, Path::new("/repo/target"));
    let env: BTreeMap<_, _> = command.get_envs().collect();
    assert_eq!(
        env[std::ffi::OsStr::new("RIG_PROVIDER_TEST_MODE")],
        Some(std::ffi::OsStr::new("replay"))
    );
    assert_eq!(env[std::ffi::OsStr::new("RIG_REGENERATE_GOLDEN")], None);
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
            "driver_adoption"
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
    // Bench targets are the one target class outside cargo test's default
    // selection; none exists, so nothing needs the removed workspace check.
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
        let manifest = Path::new(package["manifest_path"].as_str().unwrap());
        let benches = package["targets"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["kind"].as_array().unwrap().iter().any(|k| k == "bench"));
        assert!(
            !benches && !manifest.with_file_name("benches").is_dir(),
            "{} declares a bench target; full-tests does not compile benches, so a locked-version owner is needed (see checks.rs)",
            manifest.display()
        );
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
