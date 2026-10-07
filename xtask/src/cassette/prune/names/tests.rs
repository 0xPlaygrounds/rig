use super::*;

fn corpus() -> Corpus {
    Corpus::new(
        [
            "openai/streaming/smoke.yaml",
            "openai/corpus/text.yaml",
            "anthropic/corpus/text.yaml",
            "openai/frames/cut.yaml",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect(),
        BTreeSet::from([Golden {
            name: "openai_text".into(),
        }]),
    )
}

#[test]
fn a_literal_names_a_fixture_by_scenario_path_or_provider() {
    let corpus = corpus();
    let named = |literal: &str, provider: Option<&str>| resolve_fixture(literal, provider, &corpus);
    assert_eq!(
        named("streaming/smoke", Some("openai")),
        ["openai/streaming/smoke.yaml"]
    );
    assert_eq!(
        named("streaming/smoke.yaml", Some("openai")),
        ["openai/streaming/smoke.yaml"]
    );
    assert_eq!(
        named("anthropic/corpus/text", Some("openai")),
        ["anthropic/corpus/text.yaml"]
    );
    assert_eq!(
        named("/x/fixtures/cassettes/openai/frames/cut.yaml", None),
        ["openai/frames/cut.yaml"]
    );
    // Outside a provider's tests a bare scenario names it for every provider.
    assert_eq!(
        named("corpus/text", None),
        ["anthropic/corpus/text.yaml", "openai/corpus/text.yaml"]
    );
    assert!(named("streaming/smoke", Some("anthropic")).is_empty());
    assert!(named("a sentence with streaming/smoke", Some("openai")).is_empty());
}

#[test]
fn a_test_owns_its_wrapper_and_row_literals_and_reads_the_rest() {
    let source = r#"
        const FRAMES: &str = "frames/cut";

        /// Mentions "corpus/text" in prose only.
        #[tokio::test]
        async fn smoke() {
            with_openai_cassette("streaming/smoke", |client| async move {
                let _ = recorded_json_request("openai", "corpus/text");
            })
            .await;
        }

        crate::matrix::golden_matrix! {
            wrapper: with_openai_cassette, wire: wire, run: run_agent, oracle: golden_effects;
            /// A row.
            #[tokio::test]
            row_one: ("corpus/text", CELL, "openai_text");
            #[tokio::test]
            scripted: SCRIPTED => "openai_text";
        }
    "#;
    let found = file_names(source, Some("openai::cassette::f")).expect("valid Rust");
    let smoke = found
        .tests
        .get("openai::cassette::f::smoke")
        .expect("the test fn");
    assert_eq!(smoke.0, BTreeSet::from(["streaming/smoke".to_owned()]));
    assert!(smoke.1.contains("corpus/text"));
    let row = found
        .tests
        .get("openai::cassette::f::row_one")
        .expect("the row");
    assert_eq!(row.0, BTreeSet::from(["corpus/text".to_owned()]));
    assert!(row.1.contains("openai_text"));
    let scripted = found
        .tests
        .get("openai::cassette::f::scripted")
        .expect("the scripted row");
    assert!(scripted.0.is_empty());
    assert!(!found.tests.contains_key("openai::cassette::f::wrapper"));
    assert!(found.outside.contains("frames/cut"));
    assert!(
        !found
            .outside
            .iter()
            .any(|literal| literal.contains("prose")),
        "doc comments read nothing"
    );
}

#[test]
fn a_file_level_literal_is_read_by_every_test_of_its_file() {
    let corpus = corpus();
    let source = r#"
        const FRAMES: &str = "frames/cut";
        #[test]
        fn one() { let _ = "openai_text"; }
        #[test]
        fn two() {}
    "#;
    let found = file_names(source, Some("openai::cassette::corpus_f")).expect("valid Rust");
    let mut names = Names::default();
    names.add(
        "crates/rig-cassette/tests/providers/openai/cassette/corpus_f.rs",
        Some("openai"),
        Some("openai::cassette::corpus_f"),
        found,
        &corpus,
    );
    let test = |name: &str| {
        names
            .tests
            .get(&(
                "rig-cassette::openai".to_owned(),
                format!("openai::cassette::corpus_f::{name}"),
            ))
            .expect("the test")
    };
    assert!(test("one").file_reads.contains("openai/frames/cut.yaml"));
    assert!(test("two").file_reads.contains("openai/frames/cut.yaml"));
    assert_eq!(
        test("one").goldens,
        BTreeSet::from([Golden {
            name: "openai_text".into(),
        }])
    );
    assert_eq!(test("one").module, "openai::cassette::corpus_f");
}

#[test]
fn a_name_outside_the_provider_tests_protects_what_it_names() {
    let corpus = corpus();
    let source = r#"
        fn helper() {
            let _ = ("corpus/text", "openai_text");
        }
    "#;
    let found = file_names(source, None).expect("valid Rust");
    let mut names = Names::default();
    names.add("tests/core/x.rs", None, None, found, &corpus);
    assert!(names.tests.is_empty());
    assert!(names.fixtures.contains_key("openai/corpus/text.yaml"));
    assert!(names.fixtures.contains_key("anthropic/corpus/text.yaml"));
    assert!(names.goldens.contains_key(&Golden {
        name: "openai_text".into(),
    }));
}

#[test]
fn a_row_body_is_parenthesized_or_names_a_golden() {
    let body = |text: &str| -> Vec<TokenTree> {
        text.parse::<TokenStream>()
            .expect("tokens")
            .into_iter()
            .collect()
    };
    assert!(is_row_body(&body(r#"("a", CELL, "g")"#)));
    assert!(is_row_body(&body(r#"SCRIPTED => "g""#)));
    assert!(!is_row_body(&body("wire_matrix_case")));
    assert!(!is_row_body(&body("with_x, wire: y")));
}

#[test]
fn a_provider_file_another_target_compiles_is_shared() {
    let source = r#"
        #[path = "."]
        mod cassette {
            #[path = "../providers/anthropic/cassette/lifecycle_matrix.rs"]
            mod lifecycle_matrix;
        }
        #[path = "../common/corpus_matrix.rs"]
        mod corpus_matrix;
    "#;
    assert_eq!(
        shared_files(source),
        BTreeSet::from([format!(
            "{PROVIDERS}/anthropic/cassette/lifecycle_matrix.rs"
        )])
    );
}
