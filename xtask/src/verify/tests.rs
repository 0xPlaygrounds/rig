use super::*;
use std::collections::BTreeSet;
fn opts(mode: &str) -> Options {
    Options::parse(vec![mode.into(), "--base".into(), "HEAD".into()]).unwrap()
}
fn metadata() -> Value {
    serde_json::json!({"packages":[{"name":"rig","manifest_path":"/repo/Cargo.toml","targets":[{"name":"anthropic","kind":["test"]}]},{"name":"rig-ecs","manifest_path":"/repo/crates/rig-ecs/Cargo.toml","dependencies":[]},{"name":"example","manifest_path":"/repo/examples/example/Cargo.toml","dependencies":[{"name":"rig-ecs"}]}]})
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
        ".config/nextest.toml",
    ] {
        assert_eq!(ids("--changed", &[p]), ids("--full", &[]));
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
fn fingerprints_include_content_configuration_and_deletions() {
    let root = std::env::temp_dir().join(format!("rig-verify-test-{}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("fixture.json"), "a").unwrap();
    let files = vec!["fixture.json".into()];
    let a = execute::digest(&root, &files, b"rustc-1;features-a;target-native").unwrap();
    assert_ne!(
        a,
        execute::digest(&root, &files, b"rustc-2;features-b;target-wasm").unwrap()
    );
    std::fs::write(root.join("fixture.json"), "b").unwrap();
    assert_ne!(
        a,
        execute::digest(&root, &files, b"rustc-1;features-a;target-native").unwrap()
    );
    std::fs::remove_file(root.join("fixture.json")).unwrap();
    assert_ne!(
        a,
        execute::digest(&root, &files, b"rustc-1;features-a;target-native").unwrap()
    );
    std::fs::remove_dir(root).unwrap();
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
impl Drop for Repo {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}
#[test]
fn staged_then_undone_and_untracked_inputs_are_not_lost() {
    let repo = Repo::new();
    std::fs::write(repo.0.join("file.rs"), "staged").unwrap();
    output(&repo.0, "git", &["add", "file.rs"]).unwrap();
    std::fs::write(repo.0.join("file.rs"), "original").unwrap();
    std::fs::write(repo.0.join("new.rs"), "new").unwrap();
    let paths = selection::changes(&repo.0, &opts("--changed")).unwrap();
    assert!(paths.contains("file.rs"));
    assert!(paths.contains("new.rs"));
}
#[test]
fn failed_required_command_removes_success_and_stops_later_checks() {
    let repo = Repo::new();
    let metadata = serde_json::json!({"target_directory":repo.0.join("target")});
    let mut opts = opts("--changed");
    opts.reuse = false;
    let ok = Check {
        id: "first".into(),
        reason: "test".into(),
        steps: vec![Step::new("git", &["status", "--porcelain"])],
    };
    execute::run(&repo.0, &metadata, &opts, &[ok]).unwrap();
    let receipt = repo.0.join("target/verify/first.json");
    assert!(receipt.exists());
    let bad = Check {
        id: "first".into(),
        reason: "test".into(),
        steps: vec![Step::new("git", &["not-a-command"])],
    };
    let later = Check {
        id: "later".into(),
        reason: "test".into(),
        steps: vec![Step::new("git", &["status", "--porcelain"])],
    };
    let error = execute::run(&repo.0, &metadata, &opts, &[bad, later]).unwrap_err();
    assert!(
        error.to_string().contains("required check first failed"),
        "{error}"
    );
    assert!(!receipt.exists());
    assert!(!repo.0.join("target/verify/later.json").exists());
}
#[test]
#[cfg(unix)]
fn symlink_inputs_can_still_run_fresh_without_cache() {
    let repo = Repo::new();
    std::os::unix::fs::symlink("file.rs", repo.0.join("link")).unwrap();
    let metadata = serde_json::json!({"target_directory":repo.0.join("target")});
    let check = Check {
        id: "fresh".into(),
        reason: "test".into(),
        steps: vec![Step::new("git", &["status", "--porcelain"])],
    };
    let mut opts = opts("--changed");
    opts.reuse = false;
    execute::run(&repo.0, &metadata, &opts, &[check]).unwrap();
    assert!(!repo.0.join("target/verify/fresh.json").exists());
}
#[test]
fn full_covers_workspace_and_example_targets() {
    let all = checks::all();
    let c = all.iter().find(|c| c.id == "full-tests").unwrap();
    assert!(c.steps[0].args.contains(&"--workspace".into()));
    let c = all.iter().find(|c| c.id == "workspace-check").unwrap();
    for flag in ["--workspace", "--all-features", "--all-targets"] {
        assert!(c.steps[0].args.contains(&flag.into()));
    }
}

#[test]
fn reporting_only_edits_do_not_invalidate_executable_results() {
    let repo = Repo::new();
    let files = vec!["file.rs".into(), "DEVELOPING.md".into()];
    let before = execute::digest(&repo.0, &files, b"configuration").unwrap();
    std::fs::write(repo.0.join("DEVELOPING.md"), "updated timing report").unwrap();
    assert_eq!(
        before,
        execute::digest(&repo.0, &files, b"configuration").unwrap()
    );
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
fn successful_execution_is_reused_until_fixture_or_command_changes() {
    let repo = Repo::new();
    let metadata = serde_json::json!({"target_directory":repo.0.join("target")});
    let mut check = Check {
        id: "count-executions".into(),
        reason: "test".into(),
        // An empty commit changes only .git, which is not an execution input.
        // Its count tells us whether the runner actually invoked the command.
        steps: vec![Step::new(
            "git",
            &[
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "--quiet",
                "--allow-empty",
                "-m",
                "executed",
            ],
        )],
    };
    let opts = opts("--changed");
    execute::run(&repo.0, &metadata, &opts, &[check.clone()]).unwrap();
    execute::run(&repo.0, &metadata, &opts, &[check.clone()]).unwrap();
    // CI deliberately executes even when a matching local result exists.
    let first_count = if std::env::var_os("CI").is_some() {
        3
    } else {
        2
    };
    let count = || {
        output(&repo.0, "git", &["rev-list", "--count", "HEAD"])
            .unwrap()
            .trim()
            .parse::<usize>()
            .unwrap()
    };
    assert_eq!(count(), first_count);
    std::fs::write(repo.0.join("fixture.yaml"), "changed fixture").unwrap();
    execute::run(&repo.0, &metadata, &opts, &[check.clone()]).unwrap();
    assert_eq!(count(), first_count + 1);
    check.steps[0]
        .env
        .insert("RIG_VERIFY_TEST_CONFIG".into(), "different".into());
    execute::run(&repo.0, &metadata, &opts, &[check.clone()]).unwrap();
    assert_eq!(count(), first_count + 2);
    check.id = "different-test-identity".into();
    execute::run(&repo.0, &metadata, &opts, &[check]).unwrap();
    assert_eq!(count(), first_count + 3);
}

#[test]
fn ancestor_cargo_configuration_invalidates_results() {
    let repo = Repo::new();
    let child = repo.0.join("child");
    std::fs::create_dir(&child).unwrap();
    std::fs::create_dir(repo.0.join(".cargo")).unwrap();
    let check = Check {
        id: "config".into(),
        reason: "test".into(),
        steps: vec![],
    };
    let before = execute::config(&child, &check).unwrap();
    std::fs::write(repo.0.join(".cargo/config.toml"), "[build]\njobs = 2\n").unwrap();
    assert_ne!(before, execute::config(&child, &check).unwrap());
}

#[test]
fn verification_commands_force_replay_and_preserve_retry_contracts() {
    let step = Step::new("cargo", &["test"])
        .env("RIG_PROVIDER_TEST_MODE", "record")
        .env("RIG_REGENERATE_GOLDEN", "1")
        .env("NEXTEST_RETRIES", "5");
    let command = execute::command(Path::new("/repo"), &step);
    let env: BTreeMap<_, _> = command.get_envs().collect();
    assert_eq!(
        env[std::ffi::OsStr::new("RIG_PROVIDER_TEST_MODE")],
        Some(std::ffi::OsStr::new("replay"))
    );
    assert_eq!(env[std::ffi::OsStr::new("RIG_REGENERATE_GOLDEN")], None);
    assert_eq!(env[std::ffi::OsStr::new("NEXTEST_RETRIES")], None);
}
