use super::*;
use rig_cassette_inventory::{Scenario, Test};

#[test]
fn compiled_discovery_does_not_require_existing_fixture_directories() {
    let artifact = |name, kind| {
        serde_json::json!({
            "reason": "compiler-artifact", "profile": { "test": true },
            "target": { "name": name, "kind": [kind] }, "executable": format!("/target/{name}")
        })
        .to_string()
    };
    let output = [
        artifact("existing", "test"),
        artifact("first_capture", "test"),
        artifact("library", "lib"),
    ]
    .join("\n");
    let known_directories = BTreeSet::from(["existing".into()]);
    let all = compiled_binaries(&output, &known_directories, true);
    assert_eq!(
        all.keys().cloned().collect::<BTreeSet<_>>(),
        BTreeSet::from(["existing".into(), "first_capture".into()])
    );
    let selected = compiled_binaries(&output, &known_directories, false);
    assert_eq!(
        selected.keys().cloned().collect::<BTreeSet<_>>(),
        known_directories
    );
}

#[test]
fn loose_root_files_are_not_silently_omitted() -> anyhow::Result<()> {
    let root = assert_fs::TempDir::new()?;
    std::fs::create_dir_all(root.path().join(ROOT).join("example"))?;
    ensure!(providers(root.path(), None)? == BTreeSet::from(["example".into()]));
    std::fs::write(root.path().join(ROOT).join("orphan.yaml"), "orphan")?;
    ensure!(providers(root.path(), None).is_err());
    Ok(())
}

#[test]
fn recording_subprocess_receives_exact_scope_and_must_finalize() -> anyhow::Result<()> {
    let root = assert_fs::TempDir::new()?;
    let marker = root.path().join(".recording-driver-fixture");
    std::fs::write(&marker, "receipt")?;
    let executable = std::env::current_exe()?;
    let mut cell = recording();
    cell.test = "cassettes::tests::recording_fixture_process".into();
    cell.ignored = true;
    execute_recording(root.path(), &cell, &executable)?;
    let fixture = root.path().join(ROOT).join("example/cell.yaml");
    ensure!(std::fs::read_to_string(&fixture)? == "local recording driver test");

    let mut zero_tests = cell.clone();
    zero_tests.test = "no_such_recording_test".into();
    ensure!(execute_recording(root.path(), &zero_tests, &executable).is_err());
    let mut wrong_scope = cell.clone();
    wrong_scope.scenarios = BTreeSet::from(["example/unselected".into()]);
    ensure!(execute_recording(root.path(), &wrong_scope, &executable).is_err());
    ensure!(
        !root
            .path()
            .join(ROOT)
            .join("example/unselected.yaml")
            .exists()
    );

    std::fs::remove_file(&fixture)?;
    std::fs::write(&marker, "no receipt")?;
    ensure!(execute_recording(root.path(), &cell, &executable).is_err());
    ensure!(
        fixture.is_file(),
        "a file alone is not a finalization receipt"
    );
    Ok(())
}

// This is an executable subprocess fixture, run by the parent test with the
// same exact/ignored arguments used for real recording plans. It cannot write
// outside that parent's temporary directory because the marker is mandatory.
#[test]
#[ignore = "executed by recording_subprocess_receives_exact_scope_and_must_finalize"]
fn recording_fixture_process() -> anyhow::Result<()> {
    let root = std::env::current_dir()?;
    let marker = std::fs::read_to_string(root.join(".recording-driver-fixture"))?;
    ensure!(std::env::var(MODE)? == "record");
    let scope: BTreeSet<String> = serde_json::from_str(&std::env::var(SCOPE)?)?;
    ensure!(scope == BTreeSet::from(["example/cell".into()]));
    let path = root.join(ROOT).join("example/cell.yaml");
    std::fs::create_dir_all(path.parent().context("fixture parent")?)?;
    std::fs::write(path, "local recording driver test")?;
    if marker == "receipt" {
        println!("RIG_CASSETTE_RECORDED=example/cell");
    }
    Ok(())
}

fn recording() -> Recording {
    Recording {
        provider: "example".into(),
        test: "example::cell".into(),
        ignored: false,
        scenarios: BTreeSet::from(["example/cell".into()]),
    }
}

#[test]
fn successful_zero_test_invocation_is_not_a_capture() {
    assert!(
        verify_execution(
            &recording(),
            "test result: ok. 0 passed; 0 failed; 0 ignored;\n"
        )
        .is_err()
    );
    assert!(
        verify_execution(
            &recording(),
            "test result: ok. 1 passed; 0 failed; 0 ignored;\n"
        )
        .is_err()
    );
    assert!(verify_execution(&recording(), "RIG_CASSETTE_AUTHORIZED=example/cell\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured;\n").is_err());
    assert!(verify_execution(&recording(), "RIG_CASSETTE_RECORDED=example/cell\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured;\n").is_ok());
}

#[test]
fn exact_names_and_ignore_status_are_checked() {
    let mut inventory = Inventory {
        tests: vec![Test {
            name: "example::cell".into(),
            ignored: false,
            scenarios: vec![Scenario::live("example/cell")],
        }],
        families: vec![],
    };
    let listed = BTreeSet::from(["example::cell".into()]);
    assert!(validate_names(&inventory, &listed, &BTreeSet::new(), "example").is_ok());
    assert!(validate_names(&inventory, &BTreeSet::new(), &BTreeSet::new(), "example").is_err());
    inventory.tests[0].ignored = true;
    assert!(validate_names(&inventory, &listed, &BTreeSet::new(), "example").is_err());
    assert!(validate_names(&inventory, &listed, &listed, "example").is_ok());
}

#[test]
fn exports_are_unique_and_structured() {
    assert!(parse_inventory("test result: ok. 0 passed;").is_err());
    let exported =
        "test cassette_inventory ... RIG_CASSETTE_INVENTORY={\"tests\":[],\"families\":[]}\n";
    assert!(parse_inventory(exported).is_ok());
    assert!(parse_inventory(&format!("{exported}{exported}")).is_err());
}

#[test]
fn selections_cannot_silently_disagree() {
    assert!(
        Options::parse(
            "plan",
            ["--provider", "openai", "--scenario", "anthropic/cell"].map(str::to_owned)
        )
        .is_err()
    );
    assert!(Options::parse("plan", ["--scenario", "cell"].map(str::to_owned)).is_err());
    assert!(Options::parse("list", ["--scenario", "openai/cell"].map(str::to_owned)).is_err());
    let options = Options::parse("plan", ["--scenario", "openai/cell"].map(str::to_owned))
        .expect("selection");
    assert_eq!(options.provider.as_deref(), Some("openai"));
}

#[test]
fn ignored_execution_is_explicit_and_scope_is_exact() {
    let mut recording = recording();
    assert!(!test_args(&recording).contains(&"--ignored".into()));
    recording.ignored = true;
    assert!(test_args(&recording).contains(&"--ignored".into()));
    let command = display(&recording).expect("command");
    assert!(command.contains("RIG_CASSETTE_SCENARIOS='[\"example/cell\"]'"));
    assert!(command.contains("--exact example::cell"));
}
