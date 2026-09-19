//! Every case builds its own workspace in a temp directory. The committed
//! corpus is never read and never written: a guard that can only be tested
//! against the real tree is a guard nobody dares change.

use super::*;

use serde_json::{Value, json};

/// A throwaway workspace. Ten providers with a hundred cassettes each, which
/// is also the smallest corpus that clears `check`'s vacuity floor.
struct Repo(PathBuf);

impl Repo {
    fn new() -> Self {
        static SEQUENCE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "rig-xtask-cassettes-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).expect("temp workspace");
        Self(path)
    }

    fn write(&self, relative: &str, body: &str) {
        let path = self.0.join(relative);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("temp parent directory");
        }
        std::fs::write(path, body).expect("write temp file");
    }

    fn remove(&self, relative: &str) {
        std::fs::remove_file(self.0.join(relative)).expect("remove temp file");
    }

    fn manifest(&self, manifest: &Value) {
        self.write(
            manifest::MANIFEST_PATH,
            &serde_json::to_string_pretty(manifest).expect("render manifest"),
        );
    }

    /// Write the fixtures, test sources and scripted modules the manifest
    /// claims, so a freshly scaffolded workspace is one `check` passes.
    fn scaffold(&self, manifest: &Value) {
        for provider in manifest["providers"].as_array().expect("providers") {
            let name = provider["provider"].as_str().expect("provider name");
            let wrapper = provider["wrappers"][0].as_str().expect("wrapper");
            let mut source = String::new();

            let mut live: Vec<String> = provider["live"]
                .as_array()
                .expect("live")
                .iter()
                .map(|s| s.as_str().expect("scenario").to_owned())
                .collect();
            for entry in provider["derived"].as_array().expect("derived") {
                live.push(entry["scenario"].as_str().expect("scenario").to_owned());
            }
            for scenario in &live {
                self.write(&fixture_path(name, scenario), "interactions: []\n");
                source.push_str(&test_function(wrapper, scenario, false));
            }
            for entry in provider["unrecorded"].as_array().unwrap_or(&Vec::new()) {
                let scenario = entry["scenario"].as_str().expect("scenario");
                source.push_str(&test_function(wrapper, scenario, true));
            }
            self.write(
                &format!("crates/rig-cassette/tests/providers/{name}/cassette/suite.rs"),
                &source,
            );

            for entry in provider["scripted"].as_array().expect("scripted") {
                let family = entry["family"].as_str().expect("family");
                let cases: Vec<&str> = entry["cases"]
                    .as_array()
                    .expect("cases")
                    .iter()
                    .map(|c| c.as_str().expect("case"))
                    .collect();
                let module = entry["module"].as_str().expect("module");
                // Literals, not identifiers: a family name is an arbitrary
                // string, and the guard reads the module as text.
                self.write(
                    &format!("crates/rig-cassette/{module}"),
                    &format!(
                        "const FAMILY: &str = {family:?};\nconst CASES: [&str; {}] = [{}];\n",
                        cases.len(),
                        cases
                            .iter()
                            .map(|case| format!("{case:?}"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                );
            }
        }
        self.manifest(manifest);
    }
}

impl Drop for Repo {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn test_function(wrapper: &str, scenario: &str, ignored: bool) -> String {
    let name = scenario.replace(['/', '-', '.'], "_");
    let ignore = if ignored { "#[ignore]\n" } else { "" };
    format!(
        "{ignore}#[tokio::test]\nasync fn t_{name}() {{\n    {wrapper}({scenario:?}).await;\n}}\n"
    )
}

/// Ten providers, a thousand recordings, and one of each exotic provenance on
/// the first provider.
fn base_manifest() -> Value {
    let mut providers = Vec::new();
    for index in 0..10 {
        let name = format!("p{index}");
        let live: Vec<String> = (0..100).map(|s| format!("bulk/s{s}")).collect();
        providers.push(json!({
            "provider": name,
            "source_dir": format!("tests/providers/{name}/cassette"),
            "wrappers": [format!("with_{name}_cassette")],
            "live": live,
            "derived": [],
            "scripted": [],
        }));
    }
    providers[0]["derived"] = json!([{
        "scenario": "edge/object_shape",
        "sources": ["p0/bulk/s0"],
        "reason": "pins a compatible-endpoint response shape no key can produce",
        "rebuild": "copy p0/bulk/s0 and replace the top_p scalar with an object",
    }]);
    providers[0]["scripted"] = json!([{
        "family": "faults",
        "module": "tests/providers/p0/cassette/faults.rs",
        "sources": ["p0/bulk/s1"],
        "reason": "injects transport faults the provider will not emit on demand",
        "cases": ["status_429", "status_503"],
    }]);
    providers[0]["unrecorded"] = json!([{
        "scenario": "pending/first_capture",
        "reason": "needs a live key; the producing test is ignored until then",
    }]);
    json!({ "providers": providers })
}

fn failure(repo: &Repo) -> String {
    check(&repo.0).expect_err("check should fail")
}

// ------------------------------------------------------------- manifest

#[test]
fn unknown_keys_are_rejected() {
    let error = Manifest::parse(
        r#"{"providers":[{"provider":"p","source_dir":"d","wrappers":["w"],"live":[],
            "derived":[],"scripted":[],"provenance":"live"}]}"#,
    )
    .expect_err("unknown key");
    assert!(error.contains("unknown key \"provenance\""), "{error}");
}

/// `derived`, `scripted` and `unrecorded` are absent-means-empty, but the
/// four keys that say which suite this is and what it owns are not.
#[test]
fn missing_keys_are_rejected() {
    for (missing, json) in [
        (
            "live",
            r#"{"providers":[{"provider":"p","source_dir":"d","wrappers":["w"]}]}"#,
        ),
        (
            "wrappers",
            r#"{"providers":[{"provider":"p","source_dir":"d","live":[]}]}"#,
        ),
        (
            "source_dir",
            r#"{"providers":[{"provider":"p","wrappers":["w"],"live":[]}]}"#,
        ),
    ] {
        let error = Manifest::parse(json).expect_err("missing key");
        assert!(
            error.contains(&format!("missing key {missing:?}")),
            "{error}"
        );
    }
}

#[test]
fn a_derivation_must_explain_and_reproduce_itself() {
    for (field, replacement) in [
        ("reason", r#""reason":"""#),
        ("rebuild", r#""rebuild":" ""#),
    ] {
        let json = format!(
            r#"{{"providers":[{{"provider":"p","source_dir":"d","wrappers":["w"],"live":["a"],
               "derived":[{{"scenario":"b","sources":["p/a"],"reason":"because","rebuild":"copy a",
               {replacement}}}],"scripted":[]}}]}}"#
        );
        let error = Manifest::parse(&json).expect_err("empty justification");
        assert!(error.contains(field), "{field}: {error}");
    }
}

#[test]
fn a_scenario_belongs_to_exactly_one_category() {
    let json = r#"{"providers":[{"provider":"p","source_dir":"d","wrappers":["w"],
        "live":["a"],"derived":[{"scenario":"a","sources":["p/a"],
        "reason":"the same id, hand-derived","rebuild":"copy p/a"}]}]}"#;
    let error = Manifest::parse(json).expect_err("duplicate id");
    assert!(
        error.contains("scenario \"a\" is declared twice"),
        "{error}"
    );

    let json = r#"{"providers":[{"provider":"p","source_dir":"d","wrappers":["w"],
        "live":["a"],"derived":[],"scripted":[],
        "unrecorded":[{"scenario":"a","reason":"pending"}]}]}"#;
    let error = Manifest::parse(json).expect_err("duplicate across unrecorded");
    assert!(error.contains("declared twice"), "{error}");
}

#[test]
fn unrecorded_is_optional() {
    let manifest = Manifest::parse(
        r#"{"providers":[{"provider":"p","source_dir":"d","wrappers":["w"],"live":["a"],
            "derived":[],"scripted":[]}]}"#,
    )
    .expect("parse");
    let provider = manifest.provider("p").expect("provider");
    assert!(provider.unrecorded.is_empty());
    assert_eq!(provider.live_scenarios(), vec!["a"]);
}

// ---------------------------------------------------------------- check

#[test]
fn a_fully_declared_corpus_passes() {
    let repo = Repo::new();
    repo.scaffold(&base_manifest());
    // The engine and the shared helper may open cassettes directly; that is
    // what the rest of the tree must go through.
    repo.write(
        "crates/rig-cassette/src/http/mod.rs",
        "fn open() { ProviderCassette::start_at(root, spec); }\n",
    );
    repo.write(
        "test-support/rig-test-support/src/cassettes.rs",
        "fn open() { ProviderCassette::start_at(root, spec); }\n",
    );
    check(&repo.0).expect("a fully declared corpus is sound");
}

#[test]
fn an_undeclared_fixture_is_an_orphan() {
    let repo = Repo::new();
    repo.scaffold(&base_manifest());
    repo.write(
        &fixture_path("p3", "smuggled/recording"),
        "interactions: []\n",
    );
    let error = failure(&repo);
    assert!(error.contains("no provenance declared"), "{error}");
    assert!(error.contains("p3/smuggled/recording"), "{error}");
}

#[test]
fn a_declared_scenario_without_a_fixture_fails() {
    let repo = Repo::new();
    repo.scaffold(&base_manifest());
    repo.remove(&fixture_path("p4", "bulk/s7"));
    let error = failure(&repo);
    assert!(
        error.contains("declared live but the fixture does not exist"),
        "{error}"
    );
    assert!(error.contains("p4/bulk/s7.yaml"), "{error}");
}

#[test]
fn an_unrecorded_scenario_must_not_have_a_fixture() {
    let repo = Repo::new();
    repo.scaffold(&base_manifest());
    repo.write(
        &fixture_path("p0", "pending/first_capture"),
        "interactions: []\n",
    );
    let error = failure(&repo);
    assert!(
        error.contains("declared unrecorded but a fixture exists"),
        "{error}"
    );
}

#[test]
fn a_test_may_not_record_an_undeclared_scenario() {
    let repo = Repo::new();
    repo.scaffold(&base_manifest());
    repo.write(
        "crates/rig-cassette/tests/providers/p2/cassette/extra.rs",
        &test_function("with_p2_cassette", "smuggled/scenario", false),
    );
    let error = failure(&repo);
    assert!(
        error.contains("a test records p2/smuggled/scenario"),
        "{error}"
    );
}

#[test]
fn a_declaration_no_test_uses_is_dead() {
    let repo = Repo::new();
    let mut manifest = base_manifest();
    manifest["providers"][5]["live"]
        .as_array_mut()
        .expect("live")
        .push(json!("ghost/scenario"));
    let repo_manifest = manifest.clone();
    repo.scaffold(&repo_manifest);
    // The fixture exists, so only the "nobody uses it" rule can fire.
    repo.write(&fixture_path("p5", "ghost/scenario"), "interactions: []\n");
    repo.write(
        "crates/rig-cassette/tests/providers/p5/cassette/suite.rs",
        &(0..100)
            .map(|s| test_function("with_p5_cassette", &format!("bulk/s{s}"), false))
            .collect::<String>(),
    );
    let error = failure(&repo);
    assert!(
        error.contains("p5/ghost/scenario is declared but no test"),
        "{error}"
    );
}

#[test]
fn an_unrecorded_scenario_needs_no_running_test() {
    // The producing test is `#[ignore]`d, so discovery cannot see it; that
    // must not read as a dead declaration.
    let repo = Repo::new();
    repo.scaffold(&base_manifest());
    check(&repo.0).expect("an ignored producer is not a dead declaration");
}

#[test]
fn a_derivation_source_must_exist() {
    let repo = Repo::new();
    let mut manifest = base_manifest();
    repo.scaffold(&manifest);
    manifest["providers"][0]["derived"][0]["sources"] = json!(["p0/bulk/nonexistent"]);
    repo.manifest(&manifest);
    let error = failure(&repo);
    assert!(
        error.contains("derived from p0/bulk/nonexistent, which is not a recorded scenario"),
        "{error}"
    );
}

#[test]
fn a_derivation_may_not_source_itself() {
    let repo = Repo::new();
    let mut manifest = base_manifest();
    repo.scaffold(&manifest);
    manifest["providers"][0]["derived"][0]["sources"] = json!(["p0/edge/object_shape"]);
    repo.manifest(&manifest);
    let error = failure(&repo);
    assert!(
        error.contains("p0/edge/object_shape is derived from itself"),
        "{error}"
    );
}

#[test]
fn a_derivation_cycle_has_no_recording_underneath_it() {
    let repo = Repo::new();
    let mut manifest = base_manifest();
    manifest["providers"][0]["derived"] = json!([
        {
            "scenario": "edge/one",
            "sources": ["p0/edge/two"],
            "reason": "first half of a cycle",
            "rebuild": "copy the other one",
        },
        {
            "scenario": "edge/two",
            "sources": ["p0/edge/one"],
            "reason": "second half of a cycle",
            "rebuild": "copy the other one",
        },
    ]);
    repo.scaffold(&manifest);
    let error = failure(&repo);
    assert!(error.contains("derivation cycle"), "{error}");
}

#[test]
fn a_scripted_family_must_be_implemented_by_its_module() {
    let repo = Repo::new();
    repo.scaffold(&base_manifest());
    repo.write(
        "crates/rig-cassette/tests/providers/p0/cassette/faults.rs",
        "fn unrelated() {}\n",
    );
    let error = failure(&repo);
    assert!(
        error.contains("does not construct the scripted family \"faults\""),
        "{error}"
    );

    repo.remove("crates/rig-cassette/tests/providers/p0/cassette/faults.rs");
    let error = failure(&repo);
    assert!(
        error.contains("names a module that does not exist"),
        "{error}"
    );
}

/// One id, one meaning. A family sharing its name with a recorded scenario
/// would make `--scenario p0/bulk/s0` mean both "record this" and "this can
/// never be recorded".
#[test]
fn a_family_may_not_share_an_id_with_a_recording() {
    let repo = Repo::new();
    let mut manifest = base_manifest();
    manifest["providers"][0]["scripted"][0]["family"] = json!("bulk/s0");
    manifest["providers"][0]["scripted"][0]["module"] =
        json!("tests/providers/p0/cassette/bulk_s0.rs");
    repo.scaffold(&manifest);
    let error = failure(&repo);
    assert!(
        error.contains("is both a scripted family and a fixture-backed scenario"),
        "{error}"
    );
}

#[test]
fn a_scripted_family_may_borrow_an_unrecorded_scenario() {
    let repo = Repo::new();
    let mut manifest = base_manifest();
    manifest["providers"][0]["scripted"][0]["sources"] = json!(["p0/pending/first_capture"]);
    repo.scaffold(&manifest);
    check(&repo.0).expect("a script may replay a scenario nobody has recorded yet");
}

#[test]
fn a_derivation_may_not_borrow_an_unrecorded_scenario() {
    let repo = Repo::new();
    let mut manifest = base_manifest();
    manifest["providers"][0]["derived"][0]["sources"] = json!(["p0/pending/first_capture"]);
    repo.scaffold(&manifest);
    let error = failure(&repo);
    assert!(
        error.contains("which is not a recorded scenario"),
        "{error}"
    );
}

#[test]
fn only_the_engine_and_its_helper_may_open_a_cassette_directly() {
    let repo = Repo::new();
    repo.scaffold(&base_manifest());
    repo.write(
        "crates/rig-cassette/tests/providers/p1/cassette/bypass.rs",
        "async fn sneak() { ProviderCassette::start_at(&root, spec).await; }\n",
    );
    let error = failure(&repo);
    assert!(
        error.contains("calls ProviderCassette::start_at directly"),
        "{error}"
    );
    assert!(error.contains("p1/cassette/bypass.rs"), "{error}");
}

#[test]
fn a_corpus_too_small_to_be_real_never_passes() {
    let repo = Repo::new();
    let manifest = json!({"providers": [{
        "provider": "p0",
        "source_dir": "tests/providers/p0/cassette",
        "wrappers": ["with_p0_cassette"],
        "live": ["only/one"],
        "derived": [],
        "scripted": [],
    }]});
    repo.scaffold(&manifest);
    let error = failure(&repo);
    assert!(error.contains("refusing to pass vacuously"), "{error}");
}

// ----------------------------------------------------------- resolution

/// The module path a scenario's test lives under, pinned against the names
/// `cargo test -p rig-cassette --all-features --test openai -- --list` prints:
/// the provider's integration root maps `providers/<provider>/mod.rs` onto a
/// module named after the provider, so `providers` is not a segment.
#[test]
fn test_paths_match_the_names_libtest_prints() {
    let root = Path::new("/repo");
    let path = |relative: &str| module_path(root, &root.join(relative)).expect("module path");
    assert_eq!(
        path("crates/rig-cassette/tests/providers/openai/cassette/agent.rs"),
        "openai::cassette::agent"
    );
    assert_eq!(
        path("crates/rig-cassette/tests/providers/anthropic/cassette/ecs_outcome/delivery.rs"),
        "anthropic::cassette::ecs_outcome::delivery"
    );
    assert_eq!(
        path("crates/rig-cassette/tests/providers/openai/mod.rs"),
        "openai"
    );
}

#[test]
fn a_plan_names_the_test_that_records_each_scenario() {
    let repo = Repo::new();
    repo.scaffold(&base_manifest());
    let manifest = Manifest::load(&repo.0).expect("manifest");
    let selection = Selection {
        provider: None,
        scenario: Some("p0/bulk/s3".to_owned()),
    };
    let (recordings, _) = resolve(&repo.0, &manifest, &selection).expect("resolve");
    assert_eq!(recordings.len(), 1);
    assert_eq!(
        recordings[0].display(),
        "RIG_PROVIDER_TEST_MODE=record cargo test -p rig-cassette --all-features --test p0 -- \
         --exact p0::cassette::suite::t_bulk_s3 --nocapture --test-threads=1"
    );
}

#[test]
fn recording_an_unrecorded_scenario_asks_for_the_ignored_test() {
    let repo = Repo::new();
    repo.scaffold(&base_manifest());
    let manifest = Manifest::load(&repo.0).expect("manifest");
    let selection = Selection {
        provider: None,
        scenario: Some("p0/pending/first_capture".to_owned()),
    };
    let (recordings, _) = resolve(&repo.0, &manifest, &selection).expect("resolve");
    assert!(
        recordings[0].display().ends_with("--ignored"),
        "{}",
        recordings[0].display()
    );
}

#[test]
fn a_plan_excludes_what_it_must_not_record() {
    let repo = Repo::new();
    repo.scaffold(&base_manifest());
    let manifest = Manifest::load(&repo.0).expect("manifest");
    let selection = Selection {
        provider: Some("p0".to_owned()),
        scenario: None,
    };
    let (recordings, excluded) = resolve(&repo.0, &manifest, &selection).expect("resolve");
    assert_eq!(recordings.len(), 101, "100 bulk recordings and one pending");
    let ids: Vec<&str> = excluded.iter().map(|e| e.id.as_str()).collect();
    assert_eq!(ids, ["p0/edge/object_shape", "p0:faults"]);
}

#[test]
fn recording_refuses_anything_it_would_have_to_invent() {
    let repo = Repo::new();
    repo.scaffold(&base_manifest());
    let manifest = Manifest::load(&repo.0).expect("manifest");
    let refuse = |scenario: &str| {
        resolve(
            &repo.0,
            &manifest,
            &Selection {
                provider: None,
                scenario: Some(scenario.to_owned()),
            },
        )
        .expect_err("recording must be refused")
    };

    let derived = refuse("p0/edge/object_shape");
    assert!(derived.contains("is derived, not recorded"), "{derived}");
    assert!(derived.contains("rebuild it instead"), "{derived}");

    let scripted = refuse("p0/status_429");
    assert!(scripted.contains("scripted family"), "{scripted}");

    let undeclared = refuse("p0/not/declared");
    assert!(undeclared.contains("is not declared in"), "{undeclared}");

    let unknown = refuse("nope/whatever");
    assert!(unknown.contains("unknown provider"), "{unknown}");
}

// ----------------------------------------------------------------- cli

#[test]
fn misuse_is_rejected_with_the_usage_text() {
    let root = Path::new("/nonexistent");
    for args in [
        vec!["list", "--help"],
        vec!["list", "--provenance"],
        vec!["plan", "--scenario", "openai"],
        vec!["records"],
        vec![],
    ] {
        let error = run(root, args.iter().map(|a| (*a).to_owned()).collect())
            .expect_err("misuse must not be silently accepted");
        assert!(!error.is_empty(), "{args:?}");
    }
    let error = run(
        Path::new("/nonexistent"),
        vec!["list".into(), "--help".into()],
    )
    .expect_err("unknown flag");
    assert!(
        error.contains("unknown cassettes list option --help"),
        "{error}"
    );
    let error = run(
        Path::new("/nonexistent"),
        vec!["list".into(), "--provenance".into(), "invented".into()],
    )
    .expect_err("bad provenance");
    assert!(
        error.contains("expected live, derived or scripted"),
        "{error}"
    );
}

#[test]
fn listing_renders_every_provenance() {
    let repo = Repo::new();
    repo.scaffold(&base_manifest());
    list(
        &repo.0,
        &ListOptions {
            provider: Some("p0".to_owned()),
            provenance: None,
            json: true,
        },
    )
    .expect("list");
    let manifest = Manifest::load(&repo.0).expect("manifest");
    let rows = rows(&manifest);
    let p0: Vec<&Row> = rows.iter().filter(|row| row.provider == "p0").collect();
    assert_eq!(p0.len(), 103);
    let scripted = p0
        .iter()
        .find(|row| row.category == Category::Scripted)
        .expect("scripted row");
    assert_eq!(scripted.id, "p0:faults");
    assert!(scripted.fixture.is_none());
    let unrecorded = p0
        .iter()
        .find(|row| row.unrecorded)
        .expect("unrecorded row");
    assert_eq!(unrecorded.category, Category::Live);
    assert!(unrecorded.fixture.is_none());
}

// ------------------------------------------------------------ discovery

#[test]
fn matrix_rows_are_scenarios_with_test_names() {
    let matrix = parse_matrix(
        r#"wrapper: with_openai_cassette, wire: w, run: r, oracle: o;
           #[tokio::test]
           first: ("a/one", cell::path, "golden_one");
           #[ignore]
           #[tokio::test]
           second: ("a/two", cell::path, "golden_two");"#
            .parse()
            .expect("tokens"),
    )
    .expect("parse matrix");
    assert_eq!(matrix.wrapper.as_deref(), Some("with_openai_cassette"));
    let rows: Vec<(&str, &str, bool)> = matrix
        .rows
        .iter()
        .map(|row| (row.name.as_str(), row.scenario.as_str(), row.ignored))
        .collect();
    assert_eq!(rows, [("first", "a/one", false), ("second", "a/two", true)]);
}

#[test]
fn a_helper_that_opens_the_cassette_belongs_to_its_callers() {
    let repo = Repo::new();
    let mut manifest = base_manifest();
    manifest["providers"][6]["live"] = json!(["helper/indirect"]);
    repo.scaffold(&manifest);
    repo.write(
        "crates/rig-cassette/tests/providers/p6/cassette/suite.rs",
        "fn fixture() { with_p6_cassette(\"helper/indirect\"); }\n\
         #[tokio::test]\nasync fn drives_it() { fixture(); }\n",
    );
    let loaded = Manifest::load(&repo.0).expect("manifest");
    let provider = loaded.provider("p6").expect("p6");
    let discovered = discover(&repo.0, provider).expect("discover");
    assert_eq!(
        discovered
            .tests
            .get("helper/indirect")
            .map(|tests| tests.iter().cloned().collect::<Vec<_>>()),
        Some(vec!["p6::cassette::suite::drives_it".to_owned()])
    );
}
