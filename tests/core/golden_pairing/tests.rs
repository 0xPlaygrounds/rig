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
        let failures = pairing_problems(&["duplicate".into(), "orphan".into()], &producers, corpus);
        assert_eq!(failures.len(), 3);
        assert!(failures.iter().all(|failure| failure.contains(corpus)));
    }
}
