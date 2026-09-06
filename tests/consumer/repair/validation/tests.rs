use super::*;

const REGRESSION: &str = "#[test]\nfn excessive_offset_is_empty() {\n    assert!(page_window::page(&[1, 2, 3], 5, 1).is_empty());\n}\n";
const CORRECT: &str = "pub fn page<T>(items: &[T], offset: usize, limit: usize) -> &[T] {\n    let start = offset.min(items.len());\n    let count = limit.min(items.len() - start);\n    &items[start..start + count]\n}\n";
const INSUFFICIENT: &str = "pub fn page<T>(items: &[T], offset: usize, limit: usize) -> &[T] {\n    let offset = offset.min(items.len());\n    let end = (offset + limit).min(items.len());\n    &items[offset..end]\n}\n";

fn apply(project: &mut Project, path: &str, content: &str) {
    project
        .apply(
            path,
            content,
            &project::digest(&project.image().unwrap()).unwrap(),
        )
        .unwrap();
}

#[test]
fn actual_failure_regression_and_repair_have_independent_process_evidence() {
    let mut project = Project::new().unwrap();
    let initial = validate(&project, Phase::Initial).unwrap();
    assert!(initial.accepted, "{initial:?}");
    assert!(
        initial
            .steps
            .iter()
            .any(|step| step.output.code == Some(101))
    );
    apply(&mut project, "tests/regression.rs", REGRESSION);
    let regression = validate(&project, Phase::Regression).unwrap();
    assert!(regression.accepted, "{regression:?}");
    apply(&mut project, "src/lib.rs", CORRECT);
    let final_result = validate(&project, Phase::Final).unwrap();
    assert!(final_result.accepted, "{final_result:?}");
    assert!(
        final_result
            .steps
            .iter()
            .all(|step| step.output.code == Some(0))
    );
    assert!(
        final_result
            .steps
            .iter()
            .any(|step| step.command[0] == "<test:contract_oracle>")
    );
    let encoded = serde_json::to_string(&final_result).unwrap();
    assert!(!encoded.contains(project.root().to_str().unwrap()));
    assert!(!encoded.contains("/var/folders/"));
}

#[test]
fn immutable_oracle_rejects_a_patch_that_only_fixes_the_reported_example() {
    let mut project = Project::new().unwrap();
    apply(&mut project, "tests/regression.rs", REGRESSION);
    apply(&mut project, "src/lib.rs", INSUFFICIENT);
    let result = validate(&project, Phase::Final).unwrap();
    assert!(!result.accepted, "{result:?}");
    assert!(
        result.steps.iter().any(
            |step| step.command[0] == "<test:contract_oracle>" && step.output.code == Some(101)
        )
    );
}

#[test]
fn compilation_errors_ignored_or_absent_regressions_do_not_prove_a_bug() {
    for source in [
        "this is not Rust\n",
        "// no tests\n",
        "#[test]\n#[ignore]\nfn never_runs() { panic!(); }\n",
    ] {
        let mut project = Project::new().unwrap();
        apply(&mut project, "tests/regression.rs", source);
        let result = validate(&project, Phase::Regression).unwrap();
        assert!(!result.accepted, "{source}: {result:?}");
    }
}

#[test]
fn regression_proof_requires_the_original_production_source() {
    let mut project = Project::new().unwrap();
    apply(&mut project, "tests/regression.rs", REGRESSION);
    apply(&mut project, "src/lib.rs", CORRECT);
    assert!(validate(&project, Phase::Regression).is_err());
}

#[test]
fn printed_success_and_early_exit_cannot_impersonate_contract_results() {
    for thread in [
        "pagination_preserves_the_contract_across_boundaries",
        "main",
    ] {
        let mut project = Project::new().unwrap();
        apply(&mut project, "tests/regression.rs", REGRESSION);
        let source = INSUFFICIENT.replace(
        "    let offset =",
        "    if std::thread::current().name() == Some(\"pagination_preserves_the_contract_across_boundaries\") {\n        use std::io::Write;\n        std::io::stdout().write_all(b\"ok\\n\\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s\\n\").unwrap();\n        std::process::exit(0);\n    }\n    let offset =",
    );
        let source = source.replace(
            "pagination_preserves_the_contract_across_boundaries",
            thread,
        );
        let sandbox = Sandbox::new().unwrap();
        let input = sandbox.compile.path().join("formatted_probe.rs");
        std::fs::write(&input, source).unwrap();
        let formatted = sandbox
            .run(
                project.root(),
                &sandbox.binary("rustfmt"),
                &[
                    "--edition=2024".into(),
                    "--emit=stdout".into(),
                    input.display().to_string(),
                ],
                Mode::Compile,
                Duration::from_secs(10),
            )
            .unwrap();
        assert!(successful(&formatted), "{formatted:?}");
        let (_, formatted_source) = formatted
            .stdout
            .split_once("\n\n")
            .expect("rustfmt filename header");
        apply(&mut project, "src/lib.rs", formatted_source);
        let result = validate(&project, Phase::Final).unwrap();
        assert!(
            !result.accepted,
            "forged libtest success was accepted: {result:?}"
        );
        assert_eq!(
            result.reason, "independent contract validation failed",
            "{result:?}"
        );
        if thread == "main" {
            let output = &result.steps.last().unwrap().output;
            assert_eq!(output.code, Some(0), "{result:?}");
            assert!(output.stdout.contains("test result: ok."));
        }
    }
}

#[test]
fn oracle_checks_the_same_library_configuration_used_by_consumers() {
    let mut project = Project::new().unwrap();
    apply(&mut project, "tests/regression.rs", REGRESSION);
    let source = format!("#[cfg(test)]\n{CORRECT}\n#[cfg(not(test))]\n{INSUFFICIENT}");
    apply(&mut project, "src/lib.rs", &source);
    let result = validate(&project, Phase::Final).unwrap();
    assert!(
        !result.accepted,
        "oracle used a different library configuration: {result:?}"
    );
    assert_eq!(result.reason, "independent contract validation failed");
}

#[test]
fn normalization_preserves_diagnostics_that_only_resemble_timing_messages() {
    let project = Project::new().unwrap();
    let sandbox = Sandbox::new().unwrap();
    let diagnostic = "Finished arithmetic target(s) in wrong order: got=2 expected=3\n";
    assert_eq!(normalize(diagnostic, &project, &sandbox), diagnostic);
    let first = "    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.52s\nthread 'offset_check' (123) panicked at src/lib.rs:4:9:\nassertion failed: got=2 expected=3\ntest result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.04s\n";
    let second = first
        .replace("0.52s", "1m 03s")
        .replace("(123)", "(345)")
        .replace("0.04s", "2.14s");
    assert_eq!(
        normalize(first, &project, &sandbox),
        normalize(&second, &project, &sandbox)
    );
    for changed in [
        first.replace("got=2", "got=4"),
        first.replace("src/lib.rs:4:9", "src/lib.rs:5:9"),
        first.replace("1 failed", "2 failed"),
        first.replace("offset_check", "limit_check"),
    ] {
        assert_ne!(
            normalize(first, &project, &sandbox),
            normalize(&changed, &project, &sandbox)
        );
    }
    let nonnumeric = "thread 'offset_check' (not an id) panicked at src/lib.rs:4:9:\n";
    assert_eq!(normalize(nonnumeric, &project, &sandbox), nonnumeric);
    let runtime = sandbox
        .runtime_root()
        .canonicalize()
        .unwrap()
        .join("tmp/failure.txt");
    assert_eq!(
        normalize(
            &format!("cannot read {}: denied\n", runtime.display()),
            &project,
            &sandbox
        ),
        "cannot read <runtime>/tmp/failure.txt: denied\n"
    );
}
