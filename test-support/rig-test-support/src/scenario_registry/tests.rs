use super::*;

const WRAPPERS: &[&str] = &["with_cassette"];

#[test]
fn direct_calls_specs_and_ignored_functions_keep_their_registration_semantics() {
    let source = r#"
        fn direct() { crate::support::with_cassette("direct", callback); }
        fn spec() { with_cassette((CassetteSpec::new("spec").timeout(5)), callback); }
        #[test] #[ignore = "not recorded"] fn ignored() { with_cassette("absent", callback); }
        fn unrelated() { with_other("other", callback); }
    "#;
    assert_eq!(
        cassette_scenarios(source, WRAPPERS).expect("registrations"),
        ["direct", "spec"]
    );
    for source in [
        "fn bad() { with_cassette(name, callback); }",
        "fn bad() { with_cassette(CassetteSpec::new(name), callback); }",
        "fn bad() { with_cassette(); }",
        "not Rust",
    ] {
        assert!(
            cassette_scenarios(source, WRAPPERS).is_err(),
            "accepted {source}"
        );
    }
}

#[test]
fn qualified_matrix_invocations_in_modules_claim_only_recorded_rows() {
    let source = r#"
        mod nested {
            crate::matrix::golden_matrix! {
                wrapper: super::with_cassette, wire: wire, run: run, oracle: crate::goldens::golden_effects;
                #[tokio::test] first: ("first", CELL, "first_golden");
                #[tokio::test] #[ignore = "unrecorded"] ignored: ("absent", CELL, "absent_golden");
            }
            crate::matrix::native_matrix! {
                wrapper: super::with_cassette, wire: wire, run: run;
                #[tokio::test] native: ("first", CELL, "native");
                #[tokio::test] #[ignore = "unrecorded"] absent: ("absent", CELL, "absent");
            }
            crate::matrix::resume_matrix! {
                wrapper: with_cassette, wire: wire, run: run;
                #[tokio::test] resume: ("first", CELL, Some(1), "resume");
            }
            crate::matrix::case_matrix! {
                family: wire_matrix_case;
                #[tokio::test] scripted: truncated_after_text;
                #[tokio::test] #[ignore = "unsupported"] skipped: refusal;
            }
            crate::matrix::case_matrix! {
                wrapper: with_cassette, family: tool_lifecycle_matrix_case;
                #[tokio::test] tools: ("tools", configured, cell(Blocking, Model, Shape));
            }
        }
    "#;
    assert_eq!(
        cassette_scenarios(source, WRAPPERS).expect("matrix rows"),
        ["first", "first", "first", "tools"]
    );
}

#[test]
fn post_cassette_assertions_preserve_literal_scenario_registration() {
    let source = r#"
        crate::matrix::resume_matrix! {
            wrapper: with_cassette, wire: wire, run: run, after: checks::requests;
            #[tokio::test] task: ("long_task_matrix/inventory", CELL, None, "task");
            #[tokio::test] #[ignore = "unrecorded"] absent: ("long_task_matrix/absent", CELL, None, "absent");
        }
    "#;
    assert_eq!(
        cassette_scenarios(source, WRAPPERS).expect("post-cassette rows"),
        ["long_task_matrix/inventory"]
    );
}

#[test]
fn comments_strings_and_macro_definitions_are_not_invocations() {
    let source = r#"
        // with_cassette("comment", callback);
        const TEXT: &str = "with_cassette(\"string\", callback)";
        macro_rules! definition { () => { with_cassette("definition", callback); }; }
    "#;
    assert!(
        cassette_scenarios(source, WRAPPERS)
            .expect("source")
            .is_empty()
    );
}

#[test]
fn malformed_matrix_rows_fail_instead_of_vanishing() {
    for source in [
        "golden_matrix! { wrapper: with_cassette, wire: wire, run: run, oracle: golden; #[tokio::test] a: (dynamic(), CELL, \"golden\"); }",
        "native_matrix! { wrapper: with_cassette, wire: wire, run: run; #[tokio::test] a: (\"scenario\", CELL); }",
        "native_matrix! { wrapper: with_cassette, wire: wire, run: run; #[tokio::test] a: (\"scenario\", CELL, golden); }",
        "native_matrix! { wrapper: with_cassette, wire: wire, run: run; #[tokio::test] a: (\"scenario\", CELL, \"golden\", extra); }",
        "resume_matrix! { wrapper: with_cassette, wire: wire, run: run; #[tokio::test] a: (\"scenario\", CELL, None); }",
        "resume_matrix! { wrapper: with_cassette, wire: wire, run: run; #[tokio::test] a: (\"scenario\", CELL, None, golden); }",
        "resume_matrix! { wrapper: with_cassette, wire: wire, run: run; #[tokio::test] a: (\"scenario\", CELL, None, \"golden\", extra); }",
        "resume_matrix! { wrapper: with_cassette, wire: wire, run: run; a: (\"scenario\", CELL, None); }",
        "case_matrix! { family: wire_matrix_case; missing_attribute: truncation; }",
        "case_matrix! { wrapper: with_cassette, family: family; #[tokio::test] a: (\"a\", row); #[tokio::test] a: (\"b\", row); }",
    ] {
        assert!(
            cassette_scenarios(source, WRAPPERS).is_err(),
            "accepted {source}"
        );
    }
}

#[test]
fn sites_keep_the_wrapper_and_the_specs_declarations() {
    let source = r#"
        async fn cells() {
            with_x_cassette("plain/cell", |c| async {}).await;
            with_x_bogus_key(
                CassetteSpec::new("auth/cell")
                    .unordered()
                    .expects_account_failure(crate::cassettes::AccountFailure::Auth),
                |c| async {},
            )
            .await;
        }
    "#;
    let sites = cassette_scenario_sites(source, &["with_x_cassette", "with_x_bogus_key"])
        .expect("sites parse");
    assert_eq!(
        sites,
        [
            ScenarioSite {
                scenario: "plain/cell".into(),
                wrapper: "with_x_cassette".into(),
                declared: Vec::new(),
            },
            ScenarioSite {
                scenario: "auth/cell".into(),
                wrapper: "with_x_bogus_key".into(),
                declared: vec!["Auth".into()],
            },
        ]
    );
}

#[test]
fn a_wrapper_declares_through_its_session() {
    let source = r#"
        async fn with_x_bogus_key() {
            let cassette = ProviderCassette::start(root, "x", spec, url).await;
            cassette.expect_account_failure(crate::cassettes::AccountFailure::Auth);
        }
        async fn with_x_rejected_key() {
            let cassette = ProviderCassette::start(root, "x", spec, url).await;
            let key = cassette.bogus_api_key();
        }
        async fn with_x_cassette() {
            let cassette = ProviderCassette::start(root, "x", spec, url).await;
            let key = cassette.api_key("X_API_KEY");
        }
    "#;
    let declaring = declaring_functions(source).expect("declarations parse");
    assert_eq!(
        declaring,
        [
            ("with_x_bogus_key".to_owned(), vec!["Auth".to_owned()]),
            ("with_x_rejected_key".to_owned(), vec!["Auth".to_owned()]),
        ]
    );
}
