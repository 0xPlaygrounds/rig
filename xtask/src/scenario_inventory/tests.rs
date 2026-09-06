use super::*;

#[test]
fn conditional_test_attributes_are_discovery_candidates_not_proof_of_selection() {
    let syntax = syn::parse_file(
        r#"
        #[cfg_attr(feature = "wire", tokio::test)] async fn conditional() {}
        #[cfg_attr(test, allow(dead_code))] fn helper() {}
        #[cfg_attr(feature = "wire", cfg_attr(feature = "async", test))] fn nested() {}
    "#,
    )
    .expect("source parses");
    let names: Vec<_> = syntax
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Fn(f) if is_test(&f.attrs) => Some(f.sig.ident.to_string()),
            _ => None,
        })
        .collect();
    assert_eq!(names, ["conditional", "nested"]);
}

#[test]
fn macro_evidence_keeps_binary_and_inherited_configuration() {
    let syntax = syn::parse_file(
        r#"
        #[cfg(feature = "bedrock")]
        mod wire {
                macro_rules! named_suite { () => { #[test] fn generated() {} } }
                conformance_suite!(provider: "bedrock");
            }
    "#,
    )
    .expect("source parses");
    let mut found = BTreeMap::new();
    let mut macros = Vec::new();
    Walker {
        root: Path::new("/repo"),
        tests: &mut found,
        macros: &mut macros,
        binary: "bedrock",
        source_hash: None,
        helpers: &mut Vec::new(),
        imports: &mut Vec::new(),
    }
    .items(
        &syntax.items,
        Path::new("/repo/tests/bedrock.rs"),
        Path::new("/repo/tests"),
        Path::new("/repo/tests"),
        "bedrock",
        &[],
    )
    .expect("discovery succeeds");
    assert_eq!(macros.len(), 2);
    assert_eq!(macros[0]["definition_name"], "named_suite");
    assert!(macros[1]["tokens"].as_str().unwrap().contains("bedrock"));
    assert_eq!(macros[0]["binary"], "bedrock");
    assert!(macros[0]["attributes"].to_string().contains("bedrock"));
    assert!(
        found.is_empty(),
        "a macro invocation is not itself proof of an executable test"
    );
}

#[test]
fn comments_do_not_create_tests_and_disabled_tests_remain_discovery_evidence() {
    let syntax = syn::parse_file(
        r#"
        // #[test] fn imaginary() {}
        #[cfg(any())] #[tokio::test] async fn disabled() { assert_eq!(1, 2); }
        fn helper() {}
    "#,
    )
    .expect("source parses");
    let mut found = BTreeMap::new();
    let mut macros = Vec::new();
    Walker {
        root: Path::new("/repo"),
        tests: &mut found,
        macros: &mut macros,
        binary: "provider",
        source_hash: None,
        helpers: &mut Vec::new(),
        imports: &mut Vec::new(),
    }
    .items(
        &syntax.items,
        Path::new("/repo/tests/provider.rs"),
        Path::new("/repo/tests"),
        Path::new("/repo/tests"),
        "provider",
        &[],
    )
    .expect("discovery succeeds");
    assert_eq!(found.len(), 1);
    let disabled = &found["provider::disabled"];
    assert_eq!(disabled["assertions"].as_array().unwrap().len(), 1);
    assert!(disabled["attributes"].to_string().contains("any"));
}

#[test]
fn inline_modules_and_shared_fixture_calls_remain_distinct_tests() {
    let syntax = syn::parse_file(
        r#"
        mod cassette {
            #[test] fn plain() { with_provider("same", || {}); }
            #[test] fn memory() { with_provider("same", || {}); }
        }
    "#,
    )
    .expect("source parses");
    let mut found = BTreeMap::new();
    let mut macros = Vec::new();
    Walker {
        root: Path::new("/repo"),
        tests: &mut found,
        macros: &mut macros,
        binary: "provider",
        source_hash: None,
        helpers: &mut Vec::new(),
        imports: &mut Vec::new(),
    }
    .items(
        &syntax.items,
        Path::new("/repo/tests/provider.rs"),
        Path::new("/repo/tests"),
        Path::new("/repo/tests"),
        "provider",
        &[],
    )
    .expect("discovery succeeds");
    assert_eq!(found.len(), 2);
    assert!(found.contains_key("provider::cassette::plain"));
    assert!(found.contains_key("provider::cassette::memory"));
}

#[test]
fn helper_assertions_and_aliased_calls_retain_scoped_evidence() {
    let syntax = syn::parse_file(
        r#"
        mod common {
            pub fn validate(actual: &str) {
                assert_eq!(actual, "answer");
                anyhow::ensure!(actual.len() > 0, "must have content");
                ensure!(actual.contains("answer"));
            }
        }
        mod wire {
            use super::common::validate as check;
            async fn drive(client: Client, scenario: &str) {
                with_provider(scenario, |client| async move {
                    check(&client.agent("model").build().prompt("question").await);
                }).await;
            }
            #[test] fn first() { drive(client(), "one"); }
            #[test] fn second() { drive(client(), "two"); }
        }
    "#,
    )
    .expect("source parses");
    let mut found = BTreeMap::new();
    let mut helpers = Vec::new();
    let mut imports = Vec::new();
    Walker {
        root: Path::new("/repo"),
        tests: &mut found,
        macros: &mut Vec::new(),
        binary: "wire",
        source_hash: None,
        helpers: &mut helpers,
        imports: &mut imports,
    }
    .items(
        &syntax.items,
        Path::new("/repo/tests/wire.rs"),
        Path::new("/repo/tests"),
        Path::new("/repo/tests"),
        "",
        &[],
    )
    .expect("discovery succeeds");
    assert_eq!(found.len(), 2, "helpers never inflate the test denominator");
    assert_eq!(helpers.len(), 2);
    assert!(helpers.iter().all(|h| h["body"].is_string()));
    let validator = helpers
        .iter()
        .find(|h| h["function"] == "common::validate")
        .unwrap();
    let assertions = validator["assertions"].as_array().unwrap();
    assert_eq!(assertions.len(), 3);
    assert!(
        assertions[1]["assertion"]
            .as_str()
            .unwrap()
            .contains("anyhow :: ensure")
    );
    assert!(
        assertions[2]["assertion"]
            .as_str()
            .unwrap()
            .contains("contains")
    );
    let driver = helpers
        .iter()
        .find(|h| h["function"] == "wire::drive")
        .unwrap();
    assert!(driver["parameters"].as_str().unwrap().contains("scenario"));
    assert!(
        driver["method_calls"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["method"] == "agent")
    );
    assert_eq!(imports[0]["module"], "wire");
    assert!(imports[0]["import"].as_str().unwrap().contains("as check"));
    assert_eq!(found["wire::first"]["calls"][0]["arguments"][1], "\"one\"");
    assert_eq!(found["wire::second"]["calls"][0]["arguments"][1], "\"two\"");
}

#[test]
fn explicit_paths_inside_inline_modules_match_rustc_registration() {
    // A real rustc listing is the oracle: path attributes differ between file
    // modules and inline modules, especially below a non-mod.rs file.
    let root = std::env::temp_dir().join(format!("rig-inventory-paths-{}", std::process::id()));
    std::fs::create_dir_all(root.join("outer/inside/renamed")).unwrap();
    std::fs::create_dir_all(root.join("renamed")).unwrap();
    std::fs::write(root.join("lib.rs"), "mod outer;").unwrap();
    std::fs::write(
        root.join("outer.rs"),
        r#"
        mod inside {
            #[path = "selected.rs"] mod leaf;
            #[path = "renamed"] mod alias {
                #[path = "second.rs"] mod leaf;
            }
        }
        #[path = "beside.rs"] mod beside;
        #[path = "renamed"] mod renamed_inline {
            #[path = "leaf.rs"] mod leaf;
        }
    "#,
    )
    .unwrap();
    std::fs::write(
        root.join("outer/inside/selected.rs"),
        "#[test] fn selected() {}",
    )
    .unwrap();
    std::fs::write(
        root.join("outer/inside/renamed/second.rs"),
        "#[test] fn second() {}",
    )
    .unwrap();
    std::fs::write(root.join("beside.rs"), "#[test] fn beside() {}").unwrap();
    std::fs::write(root.join("renamed/leaf.rs"), "#[test] fn renamed() {}").unwrap();
    let binary = root.join("registered-tests");
    let compilation = std::process::Command::new("rustc")
        .args(["--edition=2024", "--test"])
        .arg(root.join("lib.rs"))
        .arg("-o")
        .arg(&binary)
        .output()
        .unwrap();
    assert!(
        compilation.status.success(),
        "{}",
        String::from_utf8_lossy(&compilation.stderr)
    );
    let listing = std::process::Command::new(&binary)
        .args(["--list", "--format", "terse"])
        .output()
        .unwrap();
    assert!(listing.status.success());
    let listed = String::from_utf8(listing.stdout).unwrap();
    let mut expected: Vec<_> = listed
        .lines()
        .filter_map(|line| line.strip_suffix(": test"))
        .collect();
    expected.sort();
    let mut found = BTreeMap::new();
    Walker {
        root: &root.canonicalize().unwrap(),
        tests: &mut found,
        macros: &mut Vec::new(),
        binary: "paths",
        source_hash: None,
        helpers: &mut Vec::new(),
        imports: &mut Vec::new(),
    }
    .file(&root.join("lib.rs"), &root, "", &[])
    .unwrap();
    assert_eq!(
        found.keys().map(String::as_str).collect::<Vec<_>>(),
        expected
    );
    assert_eq!(found.len(), 4);
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn generated_mappings_require_source_hashes_and_preserve_skip_obligations() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .canonicalize()
        .unwrap();
    let mapping = root.join("tests/ecs_parity/generated-registrations.json");
    let document = generated::load(&root, Some(&mapping)).expect("reviewed source hashes match");
    let macros = vec![json!({
        "binary": "bedrock", "module": "bedrock::streaming_conformance",
        "source": "tests/providers/bedrock/streaming_conformance.rs",
        "macro": "rig_core :: streaming_conformance_suite",
        "source_sha256": "invocation source hash", "line": 159,
        "attributes": ["#[cfg(feature = bedrock)]"], "tokens": "fixture: fixture()",
    })];
    let mut found = BTreeMap::new();
    generated::apply("bedrock", &macros, &mut found, &document).unwrap();
    assert_eq!(found.len(), 12);
    let skipped = &found["bedrock::streaming_conformance::unknown_event_is_skipped"];
    assert_eq!(skipped["classification"], "shared_provider");
    assert_eq!(
        skipped["capability_requirements"],
        json!(["unknown_event_frame"])
    );
    assert!(skipped["semantic_outcome"].is_null());
    assert!(
        skipped["skip_obligation"]
            .as_str()
            .unwrap()
            .contains("not proof")
    );
    assert!(
        generated::apply("bedrock", &macros, &mut found, &document).is_err(),
        "duplicates fail"
    );
    assert!(
        generated::apply("bedrock", &[], &mut BTreeMap::new(), &document).is_err(),
        "missing invocation fails"
    );
    let mut corrupt = document.clone();
    corrupt["suites"][0]["source_dependencies"][0]["sha256"] = json!("wrong");
    let corrupt_path =
        std::env::temp_dir().join(format!("rig-generated-map-{}.json", std::process::id()));
    std::fs::write(&corrupt_path, serde_json::to_vec(&corrupt).unwrap()).unwrap();
    let error = generated::load(&root, Some(&corrupt_path)).unwrap_err();
    assert!(error.contains("source changed"));
    let mut missing_assertion_hash = document.clone();
    missing_assertion_hash["suites"][0]["source_dependencies"]
        .as_array_mut()
        .unwrap()
        .pop();
    std::fs::write(
        &corrupt_path,
        serde_json::to_vec(&missing_assertion_hash).unwrap(),
    )
    .unwrap();
    assert!(
        generated::load(&root, Some(&corrupt_path))
            .unwrap_err()
            .contains("unhashed assertion source")
    );
    let mut missing_metadata = document.clone();
    missing_metadata["suites"][0]["cases"][0]
        .as_object_mut()
        .unwrap()
        .remove("skip_obligation");
    std::fs::write(
        &corrupt_path,
        serde_json::to_vec(&missing_metadata).unwrap(),
    )
    .unwrap();
    assert!(
        generated::load(&root, Some(&corrupt_path))
            .unwrap_err()
            .contains("skip_obligation")
    );
    let mut missing_definition = document.clone();
    missing_definition["suites"][0]["source_dependencies"]
        .as_array_mut()
        .unwrap()
        .remove(1);
    std::fs::write(
        &corrupt_path,
        serde_json::to_vec(&missing_definition).unwrap(),
    )
    .unwrap();
    assert!(
        generated::load(&root, Some(&corrupt_path))
            .unwrap_err()
            .contains("unhashed macro definition")
    );
    std::fs::remove_file(corrupt_path).unwrap();
}
