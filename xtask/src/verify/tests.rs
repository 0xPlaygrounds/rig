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
    let paths = selection::changes(&repo.0, &opts("--changed"), None).unwrap();
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
    execute::run(&repo.0, &metadata, &opts, std::slice::from_ref(&check)).unwrap();
    execute::run(&repo.0, &metadata, &opts, std::slice::from_ref(&check)).unwrap();
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
    execute::run(&repo.0, &metadata, &opts, std::slice::from_ref(&check)).unwrap();
    assert_eq!(count(), first_count + 1);
    check.steps[0]
        .env
        .insert("RIG_VERIFY_TEST_CONFIG".into(), "different".into());
    execute::run(&repo.0, &metadata, &opts, std::slice::from_ref(&check)).unwrap();
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

fn fake_check(id: &str, script: &str) -> Check {
    Check {
        id: id.into(),
        reason: "fake execution".into(),
        steps: vec![Step::new("bash", &["-c", script])],
    }
}

#[test]
fn fixture_resolution_precedes_checks_and_continuation_retains_lock() {
    let repo = Repo::new();
    let fixture = repo
        .0
        .join("crates/rig-core/tests/fixtures/telemetry_macro_consumer");
    std::fs::create_dir_all(fixture.join("src")).unwrap();
    std::fs::write(fixture.join("Cargo.toml"), "[package]\nname = \"preparation-test\"\nversion = \"0.0.0\"\nedition = \"2024\"\n[workspace]\n").unwrap();
    std::fs::write(fixture.join("src/lib.rs"), "").unwrap();
    std::fs::write(fixture.join(".gitignore"), "Cargo.lock\n").unwrap();
    let check = fake_check(
        "macro-hygiene",
        "test -f crates/rig-core/tests/fixtures/telemetry_macro_consumer/Cargo.lock && echo ran >> target/count",
    );
    let metadata = repo_metadata(&repo);
    let options = opts("--changed");
    execute::run(&repo.0, &metadata, &options, std::slice::from_ref(&check)).unwrap();
    let lock = std::fs::read(fixture.join("Cargo.lock")).unwrap();
    execute::run(&repo.0, &metadata, &options, std::slice::from_ref(&check)).unwrap();
    assert_eq!(std::fs::read(fixture.join("Cargo.lock")).unwrap(), lock);
    let baseline = if std::env::var_os("CI").is_some() {
        2
    } else {
        1
    };
    let count = || {
        std::fs::read_to_string(repo.0.join("target/count"))
            .unwrap()
            .lines()
            .count()
    };
    assert_eq!(count(), baseline);
    let before = execute::digest(
        &repo.0,
        &["crates/rig-core/tests/fixtures/telemetry_macro_consumer/Cargo.lock".into()],
        &[],
    )
    .unwrap();
    std::fs::write(
        fixture.join("Cargo.lock"),
        [lock.clone(), b"\n# changed input\n".to_vec()].concat(),
    )
    .unwrap();
    assert_ne!(
        before,
        execute::digest(
            &repo.0,
            &["crates/rig-core/tests/fixtures/telemetry_macro_consumer/Cargo.lock".into()],
            &[]
        )
        .unwrap()
    );
    // Cargo may normalize comments. Change resolution itself to exercise a
    // stale fixture lock being updated before execution, then fingerprinted.
    let manifest = std::fs::read_to_string(fixture.join("Cargo.toml")).unwrap();
    std::fs::write(
        fixture.join("Cargo.toml"),
        manifest.replace("0.0.0", "0.0.1"),
    )
    .unwrap();
    execute::run(&repo.0, &metadata, &options, &[check]).unwrap();
    assert_ne!(std::fs::read(fixture.join("Cargo.lock")).unwrap(), lock);
    assert_eq!(count(), baseline + 1);
}

#[test]
fn fixture_preparation_covers_nested_checks_only() {
    let all = checks::all();
    let fixtures = preflight::fixture_manifests(&all);
    assert_eq!(fixtures.len(), 6);
    for (id, count) in [
        ("macro-hygiene", 1),
        ("derive", 4),
        ("full-tests", 2),
        ("package-rig-core", 1),
        ("package-rig-derive", 4),
        ("package-rig", 1),
        ("provider-tool_facade_features", 1),
        ("fmt", 0),
        ("conformance", 0),
    ] {
        assert_eq!(
            preflight::fixture_manifests(&[fake_check(id, "true")]).len(),
            count,
            "{id}"
        );
    }
}

#[test]
fn generated_fixture_locks_select_owners_without_permanent_full_fallback() {
    let all = checks::all();
    for manifest in preflight::fixture_manifests(&all) {
        let lock = manifest.replace("Cargo.toml", "Cargo.lock");
        let owner = preflight::fixture_lock_owner(&lock).unwrap();
        let selected = ids("--changed", &[&lock]);
        assert!(selected.contains(owner), "{lock}: {selected:?}");
        assert_eq!(
            selected.len(),
            2,
            "owner plus formatting only: {selected:?}"
        );
        let pr = ids("--pr", &[&lock]);
        assert!(pr.contains(owner), "PR dropped fixture owner: {pr:?}");
        assert!(!pr.contains("full-tests"));
        assert!(!pr.contains("dependency-floors"));
    }
    assert!(ids("--changed", &["tests/fixtures/unknown/Cargo.lock"]).contains("full-tests"));
    assert!(
        ids(
            "--changed",
            &["crates/rig-core/tests/fixtures/unknown/Cargo.lock"]
        )
        .contains("full-tests")
    );
}

#[cfg(unix)]
#[test]
fn fixture_preparation_refuses_symlinked_lock_without_writing() {
    let repo = Repo::new();
    let fixture = repo
        .0
        .join("crates/rig-core/tests/fixtures/telemetry_macro_consumer");
    std::fs::create_dir_all(&fixture).unwrap();
    let original = repo.0.join("original.lock");
    std::fs::write(&original, "untouched").unwrap();
    std::os::unix::fs::symlink(&original, fixture.join("Cargo.lock")).unwrap();
    let error =
        preflight::prepare_fixtures(&repo.0, &[fake_check("macro-hygiene", "true")]).unwrap_err();
    assert!(error.to_string().contains("symlinked fixture input"));
    assert_eq!(std::fs::read_to_string(original).unwrap(), "untouched");
}
fn repo_metadata(repo: &Repo) -> Value {
    serde_json::json!({"target_directory":repo.0.join("target")})
}
#[test]
fn change_during_execution_cannot_certify_new_inputs() {
    let repo = Repo::new();
    let check = fake_check("mutating", "printf changed > file.rs");
    let error =
        execute::run(&repo.0, &repo_metadata(&repo), &opts("--changed"), &[check]).unwrap_err();
    assert!(error.to_string().contains("inputs changed during mutating"));
    assert!(error.to_string().contains("file.rs"));
    assert!(!repo.0.join("target/verify/mutating.json").exists());
}
#[test]
fn signalled_child_leaves_no_success_and_continuation_executes() {
    let repo = Repo::new();
    let metadata = repo_metadata(&repo);
    let check = fake_check(
        "signalled",
        "if test ! -f target/allow; then kill -TERM $$; fi; echo ran >> target/count",
    );
    let options = opts("--changed");
    assert!(execute::run(&repo.0, &metadata, &options, std::slice::from_ref(&check)).is_err());
    assert!(!repo.0.join("target/verify/signalled.json").exists());
    std::fs::write(repo.0.join("target/allow"), "").unwrap();
    execute::run(&repo.0, &metadata, &options, &[check]).unwrap();
    assert_eq!(
        std::fs::read_to_string(repo.0.join("target/count")).unwrap(),
        "ran\n"
    );
}
#[test]
fn missing_prerequisite_stops_before_any_check() {
    let repo = Repo::new();
    let first = fake_check("first", "touch target/executed");
    let missing = Check {
        id: "missing".into(),
        reason: "test".into(),
        steps: vec![Step::new("rig-nonexistent-tool-for-test", &[])],
    };
    let error = execute::run(
        &repo.0,
        &repo_metadata(&repo),
        &opts("--changed"),
        &[first, missing],
    )
    .unwrap_err();
    assert!(error.to_string().contains("no checks executed"));
    assert!(error.to_string().contains("rig-nonexistent-tool-for-test"));
    assert!(!repo.0.join("target/executed").exists());
}
#[test]
fn reuse_policy_cannot_be_bypassed_by_matching_success() {
    let repo = Repo::new();
    let metadata = repo_metadata(&repo);
    let check = fake_check("count", "echo ran >> target/count");
    for mode in ["--pr", "--full"] {
        let options = opts(mode);
        for _ in 0..2 {
            execute::run(&repo.0, &metadata, &options, std::slice::from_ref(&check)).unwrap();
        }
    }
    for id in ["full-tests", "dependency-floors"] {
        let c = fake_check(id, "true");
        assert!(execute::policy(&opts("--changed"), &c).contains("non-reusable"));
    }
    assert_eq!(
        std::fs::read_to_string(repo.0.join("target/count"))
            .unwrap()
            .lines()
            .count(),
        4
    );
}
#[test]
fn included_docs_build_inputs_moves_and_permissions_invalidate() {
    let repo = Repo::new();
    let files = vec![
        "file.rs".into(),
        "included.md".into(),
        "build.rs".into(),
        "renamed.rs".into(),
    ];
    let mut previous = execute::digest(&repo.0, &files, b"config").unwrap();
    for name in ["included.md", "build.rs"] {
        std::fs::write(repo.0.join(name), "executable input").unwrap();
        let next = execute::digest(&repo.0, &files, b"config").unwrap();
        assert_ne!(previous, next);
        previous = next;
    }
    std::fs::rename(repo.0.join("file.rs"), repo.0.join("renamed.rs")).unwrap();
    assert_ne!(
        previous,
        execute::digest(&repo.0, &files, b"config").unwrap()
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
#[test]
#[cfg(unix)]
fn planner_interruption_summarizes_and_kills_its_child_group() {
    const CHILD: &str = "RIG_VERIFY_INTERRUPT_TEST_ROOT";
    if let Some(root) = std::env::var_os(CHILD) {
        process::install_interrupt_handler().unwrap();
        let root = std::path::PathBuf::from(root);
        let metadata = serde_json::json!({"target_directory":root.join("target")});
        let check = fake_check(
            "interrupt",
            "kill -TERM $PPID; sleep 30; touch target/escaped",
        );
        let later = fake_check("later", "touch target/later");
        let error =
            execute::run(&root, &metadata, &opts("--changed"), &[check, later]).unwrap_err();
        assert!(error.to_string().contains("interrupted"));
        assert!(!root.join("target/verify/interrupt.json").exists());
        assert!(!root.join("target/escaped").exists());
        return;
    }
    let repo = Repo::new();
    let result = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "verify::tests::planner_interruption_summarizes_and_kills_its_child_group",
            "--nocapture",
        ])
        .env(CHILD, &repo.0)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(
        result.status.success(),
        "{stdout}\n{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(stdout.contains("FAILED/INTERRUPTED interrupt"), "{stdout}");
    assert!(stdout.contains("NOT RUN later"), "{stdout}");
    assert!(
        stdout.contains("cargo xtask verify --changed --base 'HEAD' --reuse"),
        "{stdout}"
    );
}

#[test]
fn tool_versions_require_the_exact_pinned_version() {
    assert!(preflight::version_matches(
        "wasm-bindgen-test-runner 0.2.118",
        "0.2.118"
    ));
    assert!(!preflight::version_matches(
        "wasm-bindgen-test-runner 0.2.126",
        "0.2.118"
    ));
    assert!(!preflight::version_matches(
        "rustc 1.95.0-nightly",
        "1.95.0"
    ));
}

#[test]
fn cache_warming_compiles_the_default_graph_without_claiming_test_execution() {
    let mut options = Options::parse(vec!["--check".into(), "default-test-build".into()]).unwrap();
    options.check = Some("default-test-build".into());
    let all = checks::all();
    let plan = selection::plan(
        Path::new("/repo"),
        &metadata(),
        &options,
        &BTreeSet::new(),
        &all,
    )
    .unwrap();
    let default = all.iter().find(|c| c.id == "default-tests").unwrap();
    let mut expected = default.steps[0].clone();
    let index = expected.args.iter().position(|a| a == "--retries").unwrap();
    expected.args.drain(index..index + 2);
    expected.args.push("--no-run".into());
    assert_eq!(plan[0].steps, vec![expected]);
    assert_ne!(plan[0].id, default.id);
    assert!(!all.iter().any(|c| c.id == "default-test-build"));
    assert!(!default.steps[0].args.contains(&"--no-run".into()));
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
fn reporting_only_selection_skips_work_but_other_docs_keep_executable_checks() {
    assert!(ids("--changed", &["DEVELOPING.md"]).is_empty());
    for path in ["docs/progress.md", "README.md", "crates/rig-ecs/README.md"] {
        assert!(!ids("--changed", &[path]).is_empty());
    }
    assert_eq!(ids("--pr", &["DEVELOPING.md"]), ids("--pr", &[]));
}

#[test]
fn ignored_generated_inputs_invalidate_but_build_outputs_do_not() {
    let repo = Repo::new();
    std::fs::write(repo.0.join(".gitignore"), "target/\ngenerated/\n").unwrap();
    std::fs::create_dir(repo.0.join("generated")).unwrap();
    std::fs::write(repo.0.join("generated/included.rs"), "original").unwrap();
    let check = fake_check("ignored-input", "echo ran >> target/count");
    let metadata = repo_metadata(&repo);
    let options = opts("--changed");
    execute::run(&repo.0, &metadata, &options, std::slice::from_ref(&check)).unwrap();
    execute::run(&repo.0, &metadata, &options, std::slice::from_ref(&check)).unwrap();
    let count = || {
        std::fs::read_to_string(repo.0.join("target/count"))
            .unwrap()
            .lines()
            .count()
    };
    let baseline = if std::env::var_os("CI").is_some() {
        2
    } else {
        1
    };
    assert_eq!(count(), baseline);
    std::fs::write(repo.0.join("generated/included.rs"), "changed").unwrap();
    execute::run(&repo.0, &metadata, &options, std::slice::from_ref(&check)).unwrap();
    assert_eq!(count(), baseline + 1);
    let paths = selection::changes(&repo.0, &options, None).unwrap();
    assert!(paths.contains("generated/included.rs"));
    assert!(!paths.iter().any(|p| p.starts_with("target/")));
    let mutating = fake_check("ignored-input", "echo mutated > generated/included.rs");
    assert!(execute::run(&repo.0, &metadata, &options, &[mutating]).is_err());
    assert!(!repo.0.join("target/verify/ignored-input.json").exists());
}
#[test]
fn path_patch_hidden_by_no_deps_metadata_cannot_be_reused() {
    let repo = Repo::new();
    let external = Repo::new();
    std::fs::write(
        repo.0.join("Cargo.toml"),
        format!("[patch.crates-io]\nfoo = {{ path = {:?} }}\n", external.0),
    )
    .unwrap();
    let check = fake_check("patched", "true");
    execute::run(&repo.0, &repo_metadata(&repo), &opts("--changed"), &[check]).unwrap();
    assert!(!repo.0.join("target/verify/patched.json").exists());
}

#[test]
#[cfg(unix)]
fn planning_and_preflight_probes_are_interruptible() {
    const CHILD: &str = "RIG_VERIFY_PROBE_INTERRUPT_TEST";
    if let Some(root) = std::env::var_os(CHILD) {
        process::install_interrupt_handler().unwrap();
        let start = std::time::Instant::now();
        let error = process::capture(
            Path::new(&root),
            "bash",
            &["-c", "kill -TERM $PPID; sleep 30"],
        )
        .unwrap_err();
        assert!(error.to_string().contains("interrupted"));
        assert!(start.elapsed() < std::time::Duration::from_secs(5));
        return;
    }
    let repo = Repo::new();
    let result = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "verify::tests::planning_and_preflight_probes_are_interruptible",
            "--nocapture",
        ])
        .env(CHILD, &repo.0)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
}

#[test]
fn custom_unignored_target_outputs_do_not_invalidate_but_tracked_inputs_do() {
    let repo = Repo::new();
    let target = repo.0.join("build-cache");
    let metadata = serde_json::json!({"target_directory": target});
    let check = fake_check("custom-target", "echo ran >> build-cache/count");
    let options = opts("--changed");
    for _ in 0..2 {
        execute::run(&repo.0, &metadata, &options, std::slice::from_ref(&check)).unwrap();
    }
    let count = || {
        std::fs::read_to_string(target.join("count"))
            .unwrap()
            .lines()
            .count()
    };
    let baseline = if std::env::var_os("CI").is_some() {
        2
    } else {
        1
    };
    assert_eq!(count(), baseline);
    assert!(
        selection::changes(&repo.0, &options, Some(&target))
            .unwrap()
            .is_empty()
    );
    std::fs::write(target.join("tracked.rs"), "source").unwrap();
    output(&repo.0, "git", &["add", "build-cache/tracked.rs"]).unwrap();
    execute::run(&repo.0, &metadata, &options, std::slice::from_ref(&check)).unwrap();
    assert_eq!(count(), baseline + 1);
    std::fs::write(target.join("tracked.rs"), "changed source").unwrap();
    execute::run(&repo.0, &metadata, &options, &[check]).unwrap();
    assert_eq!(count(), baseline + 2);
}

#[test]
fn fixture_guard_checks_untracked_rust_sources_before_staging() {
    let repo = Repo::new();
    let source = repo.0.join("crates/new/src/lib.rs");
    std::fs::create_dir_all(source.parent().unwrap()).unwrap();
    std::fs::write(&source, "let fixture = std::fs::read(\"tests/data/bad\");").unwrap();
    let check = Check {
        id: "fixture-guard".into(),
        reason: "test".into(),
        steps: vec![Step::new("@fixture-paths", &[])],
    };
    let error =
        execute::run(&repo.0, &repo_metadata(&repo), &opts("--changed"), &[check]).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("CWD-relative fixture path in crates/new/src/lib.rs")
    );
}

#[test]
fn model_checks_execute_fresh_without_receipts_and_still_detect_source_changes() {
    let repo = Repo::new();
    let metadata = repo_metadata(&repo);
    let mut plan = vec![fake_check(
        "package-rig-fastembed",
        "echo ran >> target/count",
    )];
    plan[0].steps[0]
        .env
        .insert("HF_HOME".into(), "/inherited/cache".into());
    preflight::configure_model_cache(&metadata, &mut plan).unwrap();
    let cache = repo.0.join("target/verify/fastembed-cache");
    for key in ["HF_HOME", "FASTEMBED_CACHE_DIR"] {
        assert_eq!(plan[0].steps[0].env[key], cache.to_str().unwrap());
    }
    let options = opts("--changed");
    for _ in 0..2 {
        execute::run(&repo.0, &metadata, &options, &plan).unwrap();
        assert!(
            !repo
                .0
                .join("target/verify/package-rig-fastembed.json")
                .exists()
        );
    }
    assert_eq!(
        std::fs::read_to_string(repo.0.join("target/count")).unwrap(),
        "ran\nran\n"
    );
    plan[0].steps[0].args = vec!["-c".into(), "echo changed > file.rs".into()];
    let error = execute::run(&repo.0, &metadata, &options, &plan).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("inputs changed during package-rig-fastembed")
    );
    assert!(
        !repo
            .0
            .join("target/verify/package-rig-fastembed.json")
            .exists()
    );
    assert!(preflight::uses_runtime_model(&fake_check(
        "doctests", "true"
    )));
    assert!(!preflight::uses_runtime_model(&fake_check(
        "default-tests",
        "true"
    )));
}

#[cfg(unix)]
#[test]
fn model_cache_migration_preserves_downloads_and_rejects_unsafe_moves() {
    use std::os::unix::fs::symlink;
    for case in ["safe", "conflict", "tracked", "root-link", "escaping-link"] {
        let repo = Repo::new();
        let relative = "crates/rig-fastembed/.fastembed_cache";
        let source = repo.0.join(relative);
        std::fs::create_dir_all(source.join("blobs")).unwrap();
        std::fs::create_dir_all(source.join("snapshots")).unwrap();
        std::fs::write(source.join("blobs/model"), "model bytes").unwrap();
        symlink("../blobs/model", source.join("snapshots/model")).unwrap();
        std::fs::write(repo.0.join(".gitignore"), format!("{relative}/\ntarget/\n")).unwrap();
        let mut plan = vec![fake_check("doctests", "true")];
        preflight::configure_model_cache(&repo_metadata(&repo), &mut plan).unwrap();
        let destination = repo.0.join("target/verify/fastembed-cache");
        match case {
            "conflict" => {
                std::fs::create_dir_all(&destination).unwrap();
            }
            "tracked" => {
                assert!(
                    Command::new("git")
                        .current_dir(&repo.0)
                        .args(["add", "-f", relative])
                        .status()
                        .unwrap()
                        .success()
                );
            }
            "root-link" => {
                std::fs::rename(&source, repo.0.join("original-cache")).unwrap();
                symlink(repo.0.join("original-cache"), &source).unwrap();
            }
            "escaping-link" => {
                symlink(repo.0.join("file.rs"), source.join("escape")).unwrap();
            }
            _ => {}
        }
        let result = preflight::prepare_model_cache(&repo.0, &plan);
        if case == "safe" {
            result.unwrap();
            assert!(!source.exists());
            assert_eq!(
                std::fs::read(destination.join("snapshots/model")).unwrap(),
                b"model bytes"
            );
            assert_eq!(
                std::fs::read_link(destination.join("snapshots/model")).unwrap(),
                Path::new("../blobs/model")
            );
        } else {
            assert!(result.is_err(), "{case}");
            assert_eq!(
                std::fs::read(source.join("blobs/model")).unwrap(),
                b"model bytes"
            );
        }
    }
}
