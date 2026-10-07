use super::*;

fn parse(rows: &str) -> syn::Result<GoldenMatrix> {
    syn::parse_str(&format!(
        "wrapper: super::with_cassette, wire: wire, run: cells::run, oracle: crate::goldens::golden_effects; {rows}"
    ))
}

#[test]
fn literal_rows_preserve_ignored_status_and_qualified_paths() {
    let matrix = parse(r#"
        #[tokio::test] normal: ("matrix/normal", cells::NORMAL, "normal");
        #[tokio::test] #[ignore = "not recorded"] absent: ("matrix/absent", cells::ABSENT, "absent");
    "#).expect("valid rows");
    assert_eq!(
        matrix.wrapper.segments.last().expect("wrapper").ident,
        "with_cassette"
    );
    assert_eq!(
        matrix
            .oracle
            .segments
            .iter()
            .rev()
            .nth(1)
            .expect("helper")
            .ident,
        "goldens"
    );
    assert_eq!(matrix.rows[0].scenario.value(), "matrix/normal");
    assert_eq!(matrix.rows[0].golden.value(), "normal");
    assert!(!matrix.rows[0].ignored);
    assert!(matrix.rows[1].ignored);
}

#[test]
fn dynamic_missing_duplicate_or_unregistered_rows_fail() {
    for rows in [
        "",
        "test: (\"scenario\", CELL, \"golden\");",
        "#[tokio::test] test: (scenario(), CELL, \"golden\");",
        "#[tokio::test] test: (\"scenario\", CELL, golden());",
        "#[tokio::test] test: (\"scenario\", CELL);",
        "#[tokio::test] test: (\"a\", A, \"a\"); #[tokio::test] test: (\"b\", B, \"b\");",
        "#[tokio::test] #[cfg_attr(feature = \"live\", ignore)] test: (\"a\", A, \"a\");",
    ] {
        assert!(parse(rows).is_err(), "accepted {rows}");
    }
}

#[test]
fn comments_do_not_register_rows_or_calls() {
    let matrix = parse(
        r#"
        // #[tokio::test] fake: ("not-a-scenario", FAKE, "not-a-golden");
        #[tokio::test] real: (r"real", cells::REAL, "real");
    "#,
    )
    .expect("valid rows");
    assert_eq!(matrix.rows.len(), 1);
    assert_eq!(matrix.rows[0].scenario.value(), "real");
}

#[test]
fn scripted_rows_have_no_cassette_and_still_require_test_registration() {
    let matrix: super::CaseMatrix = syn::parse_str(
        "family: scripted_case; #[tokio::test] truncation: text_prefix;          #[tokio::test] #[ignore = \"unsupported shape\"] refusal: refusal;",
    ).unwrap();
    assert!(matrix.wrapper.is_none());
    assert!(matrix.rows.is_empty());
    for source in [
        "family: scripted_case; missing_attribute: text_prefix;",
        "family: scripted_case; #[tokio::test] dynamic: make_case();",
        "family: scripted_case;",
    ] {
        assert!(syn::parse_str::<super::CaseMatrix>(source).is_err());
    }
}
