use super::*;

fn scan(source: &str, native: bool) -> Sites {
    let mut sites = Sites {
        file: if native {
            "ecs_sample.rs"
        } else {
            "corpus_sample.rs"
        }
        .into(),
        native,
        ..Sites::default()
    };
    sites.visit_file(&syn::parse_file(source).expect("test source"));
    sites
}

#[test]
fn helpers_cannot_cross_runtime_boundaries() {
    for (source, native, file) in [
        (
            r#"fn cell() { crate::goldens::golden_effects("a", &log); }"#,
            true,
            "ecs_sample.rs",
        ),
        (
            r#"fn cell() { crate::goldens::world_golden_effects("a", &log); }"#,
            false,
            "corpus_sample.rs",
        ),
    ] {
        let sites = scan(source, native);
        assert!(
            sites
                .failures
                .iter()
                .any(|failure| failure.contains(file) && failure.contains("other runtime"))
        );
    }
}

#[test]
fn native_logs_require_exactly_one_literal_world_golden() {
    for body in [
        "let log = ecs.effect_log();",
        "let log = ecs.effect_log(); crate::goldens::world_golden_effects(name, &log);",
        r#"let log = ecs.effect_log(); crate::goldens::world_golden_effects("a", &log); crate::goldens::world_golden_effects("b", &log);"#,
    ] {
        assert!(
            !scan(
                &format!("#[tokio::test] async fn cell() {{ {body} }}"),
                true
            )
            .failures
            .is_empty()
        );
    }
    let sites = scan(
        r#"#[tokio::test] async fn cell() { let log = ecs.effect_log(); crate::goldens::world_golden_effects("a", &log); }"#,
        true,
    );
    assert!(sites.failures.is_empty());
    assert_eq!(sites.world["a"], ["ecs_sample.rs::cell"]);
}

#[test]
fn ignored_rows_are_not_producers() {
    let sites = scan(
        r#"
        native_matrix! { wrapper: wrapper, wire: wire, run: run;
            #[tokio::test] a: ("recording", CELL, "a");
            #[tokio::test] #[ignore = "unrecorded"] b: ("absent", CELL, "b");
        }
        resume_matrix! { wrapper: wrapper, wire: wire, run: run;
            #[tokio::test] cut: ("recording", CELL, Some(1), "cut");
        }
    "#,
        true,
    );
    assert!(sites.failures.is_empty());
    assert_eq!(sites.world.len(), 2);
    assert_eq!(sites.ignored_world, ["b"]);
}

#[test]
fn missing_or_duplicate_producers_fail_in_each_corpus() {
    let producers = BTreeMap::from([
        (
            "duplicate".into(),
            vec!["first.rs".into(), "second.rs".into()],
        ),
        ("absent".into(), vec!["third.rs".into()]),
    ]);
    for corpus in ["agent", "world"] {
        let failures = pairing_problems(
            &["duplicate".into(), "orphan".into()],
            &producers,
            corpus,
            &BTreeSet::new(),
        );
        assert_eq!(failures.len(), 3);
        assert!(failures.iter().all(|failure| failure.contains(corpus)));
    }
}

#[test]
fn a_golden_checked_in_test_needs_one_producer_and_no_file() {
    let producers = BTreeMap::from([
        ("checked".into(), vec!["first.rs".into()]),
        ("twice".into(), vec!["second.rs".into(), "third.rs".into()]),
        ("committed".into(), vec!["fourth.rs".into()]),
    ]);
    let in_test = BTreeSet::from([
        "world/checked".to_owned(),
        "world/twice".to_owned(),
        "world/committed".to_owned(),
    ]);
    let failures = pairing_problems(&["committed".into()], &producers, "world", &in_test);
    assert_eq!(failures.len(), 2, "{failures:?}");
    assert!(failures.iter().any(|failure| failure.contains("`twice`")));
    assert!(
        failures
            .iter()
            .any(|failure| failure.contains("`committed` is committed"))
    );
    // The agent corpus reads its own labels, without the `world/` prefix.
    let failures = pairing_problems(&[], &producers, "agent", &in_test);
    assert_eq!(failures.len(), 3, "{failures:?}");
}
