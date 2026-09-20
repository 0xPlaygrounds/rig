use super::*;

#[test]
fn live_modules_expand_only_explicit_nonignored_declarations() {
    let repo = Repo::new();
    let mut manifest = base_manifest();
    repo.scaffold(&manifest);
    manifest["providers"][0]["live"] = json!([]);
    manifest["providers"][0]["live_modules"] = json!(["approved.rs"]);
    repo.write(
        "crates/rig-cassette/tests/providers/p0/cassette/approved.rs",
        &format!(
            "{}{}fn unused() {{ with_p0_cassette(\"unused/scenario\"); }}",
            test_function("with_p0_cassette", "bulk/s0", false),
            test_function("with_p0_cassette", "pending/first_capture", true),
        ),
    );
    repo.manifest(&manifest);
    let loaded = discovery::load(&repo.0).expect("expand modules");
    let p = loaded.provider("p0").expect("provider");
    assert_eq!(p.live, ["bulk/s0"]);
    assert!(p.recordable(&repo.0, "bulk/s0"));
    assert!(p.recordable(&repo.0, "pending/first_capture"));
    assert!(
        !p.recordable(&repo.0, "bulk/s1"),
        "an existing fixture is not authorization"
    );
    assert!(!p.recordable(&repo.0, "edge/object_shape"));
    repo.remove(&fixture_path("p0", "bulk/s0"));
    assert!(
        !p.recordable(&repo.0, "bulk/s0"),
        "first capture needs explicit approval"
    );
    manifest["providers"][0]["derived"][0]["scenario"] = json!("bulk/s0");
    repo.manifest(&manifest);
    assert!(
        discovery::load(&repo.0)
            .expect_err("mixed provenance")
            .to_string()
            .contains("declared twice")
    );
}

#[test]
fn live_module_declarations_cannot_be_empty_duplicated_or_escape_the_suite() {
    for modules in [
        json!(["../outside.rs"]),
        json!(["/absolute.rs"]),
        json!(["suite.rs", "suite.rs"]),
    ] {
        let mut manifest = base_manifest();
        manifest["providers"][0]["live_modules"] = modules;
        assert!(Manifest::parse(&manifest.to_string()).is_err());
    }
    let repo = Repo::new();
    let mut manifest = base_manifest();
    repo.scaffold(&manifest);
    manifest["providers"][0]["live_modules"] = json!(["empty.rs"]);
    repo.manifest(&manifest);
    assert!(discovery::load(&repo.0).is_err(), "missing module");
    repo.write(
        "crates/rig-cassette/tests/providers/p0/cassette/empty.rs",
        "// no declarations",
    );
    assert!(
        discovery::load(&repo.0)
            .expect_err("empty module")
            .to_string()
            .contains("no scenarios")
    );
}

#[test]
fn nested_helpers_with_the_same_name_do_not_overwrite_each_other() {
    let repo = Repo::new();
    let manifest = base_manifest();
    repo.scaffold(&manifest);
    repo.write(
        "crates/rig-cassette/tests/providers/p0/cassette/suite.rs",
        r#"
        mod a { fn fixture() { with_p0_cassette("bulk/s0"); }
                #[test] fn run() { fixture(); } }
        mod b { fn fixture() { with_p0_cassette("bulk/s1"); }
                #[test] fn run() { self::fixture(); } }
    "#,
    );
    let loaded = Manifest::load(&repo.0).expect("manifest");
    let found = discover(&repo.0, loaded.provider("p0").expect("provider")).expect("discovery");
    for (scenario, module) in [("bulk/s0", "a"), ("bulk/s1", "b")] {
        assert_eq!(
            found.tests.get(scenario),
            Some(&BTreeSet::from([format!(
                "p0::cassette::suite::{module}::run"
            )]))
        );
    }
}

#[test]
fn mounted_modules_use_the_declared_name_and_dynamic_calls_are_rejected() {
    let repo = Repo::new();
    repo.scaffold(&base_manifest());
    repo.write(
        "crates/rig-cassette/tests/providers/p0/mod.rs",
        "#[path = \"cassette/suite.rs\"] mod mounted;",
    );
    let manifest = Manifest::load(&repo.0).expect("manifest");
    let provider = manifest.provider("p0").expect("provider");
    let found = discover(&repo.0, provider).expect("mounted source");
    assert_eq!(
        found.tests.get("bulk/s0"),
        Some(&BTreeSet::from(["p0::mounted::t_bulk_s0".into()]))
    );
    repo.write(
        "crates/rig-cassette/tests/providers/p0/cassette/suite.rs",
        r#"
        const OTHER: &str = "bulk/s1";
        #[test] fn both() { with_p0_cassette("bulk/s0"); with_p0_cassette(OTHER); }
    "#,
    );
    assert!(
        discover(&repo.0, provider)
            .expect_err("partial inventory")
            .contains("requires a literal scenario")
    );
}

#[test]
fn an_unrecorded_scenario_requires_an_ignored_producer() {
    let repo = Repo::new();
    let manifest = base_manifest();
    repo.scaffold(&manifest);
    repo.write(
        "crates/rig-cassette/tests/providers/p0/cassette/suite.rs",
        &(0..100)
            .map(|s| test_function("with_p0_cassette", &format!("bulk/s{s}"), false))
            .collect::<String>(),
    );
    assert!(failure(&repo).contains("pending/first_capture has no ignored producer"));
}

#[test]
fn multi_scenario_tests_are_atomic_deduplicated_and_never_mix_provenance() {
    let repo = Repo::new();
    let mut manifest = base_manifest();
    manifest["providers"][1]["live"] = json!(["bulk/s0", "bulk/s1"]);
    repo.scaffold(&manifest);
    repo.write(
        "crates/rig-cassette/tests/providers/p1/cassette/suite.rs",
        r#"
        #[test] fn both() { with_p1_cassette("bulk/s0"); with_p1_cassette("bulk/s1"); }
    "#,
    );
    let loaded = Manifest::load(&repo.0).expect("manifest");
    let mut selection = Options {
        provider: Some("p1".into()),
        ..Options::default()
    };
    let (plan, _) = resolve(&repo.0, &loaded, &selection).expect("plan");
    assert_eq!(plan.len(), 1);
    assert_eq!(plan[0].scenarios.len(), 2);
    assert!(validate_test(&plan[0], &BTreeSet::new(), &BTreeSet::new()).is_err());
    let names = BTreeSet::from([plan[0].test.clone()]);
    assert!(
        validate_test(&plan[0], &names, &names).is_err(),
        "ignored test would run zero cases"
    );
    validate_test(&plan[0], &names, &BTreeSet::new()).expect("executable test");
    selection.scenario = Some("p1/bulk/s0".into());
    assert!(
        resolve(&repo.0, &loaded, &selection)
            .expect_err("partial selection")
            .contains("also records")
    );
    manifest["providers"][1]["live"] = json!(["bulk/s0"]);
    manifest["providers"][1]["derived"] = json!([{
        "scenario": "bulk/s1", "sources": ["p1/bulk/s0"], "reason": "fabricated", "rebuild": "copy live source"
    }]);
    let loaded = Manifest::parse(&manifest.to_string()).expect("manifest");
    selection.scenario = None;
    assert!(
        resolve(&repo.0, &loaded, &selection)
            .expect_err("mixed test")
            .contains("is derived")
    );
}

#[test]
fn planning_and_recording_validate_real_libtest_without_contacting_providers() {
    let repo = Repo::new();
    let mut manifest = base_manifest();
    manifest["providers"]
        .as_array_mut()
        .expect("providers")
        .truncate(1);
    manifest["providers"][0]["live"] = json!(["bulk/s0", "bulk/s1"]);
    manifest["providers"][0]["derived"] = json!([]);
    manifest["providers"][0]["scripted"] = json!([]);
    manifest["providers"][0]["unrecorded"] = json!([]);
    repo.scaffold(&manifest);
    repo.write("Cargo.toml", "[package]\nname = 'rig-cassette'\nversion = '0.0.0'\nedition = '2024'\n[[test]]\nname = 'p0'\npath = 'crates/rig-cassette/tests/p0.rs'\n");
    repo.write(
        "crates/rig-cassette/tests/p0.rs",
        "#[path = \"providers/p0/mod.rs\"] mod p0;",
    );
    repo.write(
        "crates/rig-cassette/tests/providers/p0/mod.rs",
        "mod cassette;",
    );
    repo.write(
        "crates/rig-cassette/tests/providers/p0/cassette/mod.rs",
        "mod suite;",
    );
    repo.write("crates/rig-cassette/tests/providers/p0/cassette/suite.rs", r#"
        fn with_p0_cassette(scenario: &str) {
            assert_eq!(std::env::var("RIG_PROVIDER_TEST_MODE").as_deref(), Ok("record"));
            assert_eq!(std::env::var("RIG_CASSETTE_SCENARIOS").as_deref(), Ok("p0/bulk/s0,p0/bulk/s1"));
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new().create(true).append(true).open("captures").expect("marker");
            writeln!(file, "{scenario}").expect("capture marker");
        }
        #[test] fn both() { with_p0_cassette("bulk/s0"); with_p0_cassette("bulk/s1"); }
    "#);
    assert!(
        Command::new("cargo")
            .args(["generate-lockfile", "--offline"])
            .current_dir(&repo.0)
            .status()
            .expect("lockfile")
            .success()
    );
    let selection = Options {
        provider: Some("p0".into()),
        ..Options::default()
    };
    plan(&repo.0, &selection).expect("credential-free preview");
    assert!(
        !repo.0.join("captures").exists(),
        "planning must not run tests"
    );
    record(&repo.0, &selection).expect("one command executes both scenarios");
    assert_eq!(
        std::fs::read_to_string(repo.0.join("captures")).expect("captures"),
        "bulk/s0\nbulk/s1\n"
    );
    repo.remove("captures");
    repo.write(
        "crates/rig-cassette/tests/p0.rs",
        "#[path = \"providers/p0/mod.rs\"] mod renamed;",
    );
    assert!(
        record(&repo.0, &selection)
            .expect_err("stale module path")
            .contains("exact test")
    );
    assert!(
        !repo.0.join("captures").exists(),
        "successful zero-test commands cannot record"
    );
}

#[test]
fn aliases_and_comments_do_not_bypass_source_conventions() {
    let repo = Repo::new();
    repo.scaffold(&base_manifest());
    for source in [
        "use rig_cassette::http::ProviderCassette as Session;",
        "type Session = rig_cassette::http::ProviderCassette;",
    ] {
        repo.write("bypass.rs", source);
        assert!(failure(&repo).contains("alias Session"));
    }
    repo.remove("bypass.rs");
    repo.write(
        "crates/rig-cassette/tests/providers/p0/cassette/faults.rs",
        "// faults status_429 status_503\nfn unrelated() {}",
    );
    assert!(failure(&repo).contains("does not construct the scripted family"));
}
