use super::*;

fn collect(source: &str) -> Sites {
    let mut sites = Sites {
        file: "test.rs".into(),
        ..Sites::default()
    };
    sites.visit_file(&syn::parse_file(source).expect("test source"));
    sites
}

#[test]
fn parity_is_a_consumer_and_does_not_duplicate_the_original_producer() {
    let sites = collect(
        r#"
        fn original() { crate::goldens::golden_effects("original", log); }
        mod nested {
            crate::matrix::golden_matrix! {
                wrapper: wrapper, wire: wire, run: run, oracle: crate::ecs_goldens::golden_effects;
                #[tokio::test] native: ("scenario", CELL, "original");
            }
        }
        fn reused() { crate::ecs_goldens::compare_to_original("original", log); }
    "#,
    );
    let fixtures = ["original".to_owned()];
    let identities = BTreeSet::from(["original".to_owned()]);
    assert!(pairing_problems(&fixtures, &sites, &identities).is_empty());
    assert_eq!(sites.producers["original"].len(), 1);
    assert_eq!(sites.native["original"].len(), 1);
    assert_eq!(sites.references["original"].len(), 1);
}

#[test]
fn duplicate_missing_and_orphaned_fixtures_fail() {
    let sites = collect(
        r#"
        fn first() { crate::goldens::golden_effects("duplicate", log); }
        fn second() { crate::goldens::golden_effects("duplicate", log); }
        fn missing() { crate::goldens::golden_effects("missing", log); }
        fn native() { crate::ecs_goldens::golden_effects("duplicate", log); }
    "#,
    );
    let problems = pairing_problems(
        &["duplicate".into(), "orphan".into()],
        &sites,
        &BTreeSet::from(["unused".into()]),
    );
    for message in [
        "2 producers",
        "`orphan` has no producer",
        "missing golden `missing`",
        "no identity expectation",
        "no consumer",
    ] {
        assert!(problems.iter().any(|p| p.contains(message)), "{problems:?}");
    }
}

#[test]
fn dynamic_names_and_unqualified_helpers_are_rejected() {
    for source in [
        "fn test() { crate::goldens::golden_effects(name, log); }",
        "fn test() { golden_effects(\"name\", log); }",
        "golden_matrix! { wrapper: wrapper, wire: wire, run: run, oracle: crate::goldens::other; #[tokio::test] test: (\"scenario\", CELL, \"name\"); }",
    ] {
        assert!(!collect(source).failures.is_empty(), "accepted {source}");
    }
}

#[test]
fn comments_strings_definitions_and_ignored_tests_do_not_claim_goldens() {
    let sites = collect(
        r#"
        // crate::goldens::golden_effects("comment", log);
        const TEXT: &str = "golden_effects(\"string\", log)";
        macro_rules! definition { () => { crate::goldens::golden_effects("definition", log); }; }
        #[test] #[ignore] fn ignored() { crate::goldens::golden_effects("absent", log); }
    "#,
    );
    assert!(sites.failures.is_empty());
    assert!(sites.producers.is_empty());
}
