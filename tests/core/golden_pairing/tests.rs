use super::*;

fn scan(source: &str) -> Sites {
    let mut sites = Sites {
        file: "corpus_sample.rs".into(),
        ..Sites::default()
    };
    sites.visit_file(&syn::parse_file(source).expect("test source"));
    sites
}

#[test]
fn producers_are_registered_by_literal_name() {
    let sites =
        scan(r#"#[tokio::test] async fn cell() { crate::goldens::golden_effects("a", &log); }"#);
    assert!(sites.failures.is_empty());
    assert_eq!(sites.agent["a"], ["corpus_sample.rs::cell"]);
    for source in [
        r#"fn cell() { crate::goldens::golden_effects(name, &log); }"#,
        r#"fn cell() { golden_effects("a", &log); }"#,
    ] {
        assert!(!scan(source).failures.is_empty(), "accepted {source}");
    }
}

#[test]
fn ignored_rows_are_not_producers() {
    let sites = scan(
        r#"
        golden_matrix! { wrapper: wrapper, wire: wire, run: run, oracle: crate::goldens::golden_effects;
            #[tokio::test] a: ("recording", CELL, "a");
            #[tokio::test] #[ignore = "unrecorded"] b: ("absent", CELL, "b");
        }
    "#,
    );
    assert!(sites.failures.is_empty());
    assert_eq!(sites.agent.len(), 1);
    assert!(sites.agent.contains_key("a"));
}

#[test]
fn missing_or_duplicate_producers_fail() {
    let producers = BTreeMap::from([
        (
            "duplicate".into(),
            vec!["first.rs".into(), "second.rs".into()],
        ),
        ("absent".into(), vec!["third.rs".into()]),
    ]);
    let failures = pairing_problems(
        &["duplicate".into(), "orphan".into()],
        &producers,
        &BTreeSet::new(),
    );
    assert_eq!(failures.len(), 3, "{failures:?}");
}

#[test]
fn a_golden_checked_in_test_needs_one_producer_and_no_file() {
    let producers = BTreeMap::from([
        ("checked".into(), vec!["first.rs".into()]),
        ("twice".into(), vec!["second.rs".into(), "third.rs".into()]),
        ("committed".into(), vec!["fourth.rs".into()]),
    ]);
    let in_test = BTreeSet::from([
        "checked".to_owned(),
        "twice".to_owned(),
        "committed".to_owned(),
    ]);
    let failures = pairing_problems(&["committed".into()], &producers, &in_test);
    assert_eq!(failures.len(), 2, "{failures:?}");
    assert!(failures.iter().any(|failure| failure.contains("`twice`")));
    assert!(
        failures
            .iter()
            .any(|failure| failure.contains("`committed` is committed"))
    );
}
