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
fn resume_rows_preserve_scenarios_and_validate_registration() {
    let header = "wrapper: super::with_cassette, wire: wire, run: world::run;";
    let matrix: ResumeMatrix = syn::parse_str(&format!(
        r#"{header}
        #[tokio::test] cut: ("recording", cells::CELL, Some(1), "cut");
        #[tokio::test] #[ignore = "unrecorded"] end: ("absent", cells::CELL, Some(usize::MAX), "end");
    "#
    ))
    .expect("resume matrix");
    assert_eq!(matrix.rows[0].golden.value(), "cut");
    assert_eq!(
        matrix.wrapper.segments.last().expect("wrapper").ident,
        "with_cassette"
    );
    assert_eq!(matrix.rows[0].scenario.value(), "recording");
    assert!(!matrix.rows[0].ignored);
    assert!(matrix.rows[1].ignored);
    for row in [
        "",
        "cut: (\"recording\", CELL, None);",
        "#[tokio::test] cut: (scenario(), CELL, None);",
        "#[tokio::test] cut: (\"recording\", CELL, None, golden);",
        "#[tokio::test] cut: (\"recording\", CELL, None);",
        "#[tokio::test] cut: (\"recording\", CELL, None, \"golden\", extra);",
    ] {
        assert!(syn::parse_str::<ResumeMatrix>(&format!("{header}{row}")).is_err());
    }
}

#[test]
fn native_rows_preserve_scenarios_and_world_goldens() {
    let header = "wrapper: super::with_cassette, wire: wire, run: world::run_world;";
    let matrix: NativeMatrix = syn::parse_str(&format!(
        r#"{header}
        #[tokio::test] plain: ("recording", cells::CELL, "plain");
        #[tokio::test] #[ignore = "unrecorded"] absent: ("absent", cells::CELL, "absent");
    "#
    ))
    .expect("native matrix");
    assert_eq!(matrix.rows[0].golden.value(), "plain");
    assert_eq!(
        matrix.wrapper.segments.last().expect("wrapper").ident,
        "with_cassette"
    );
    assert_eq!(matrix.rows[0].scenario.value(), "recording");
    assert!(!matrix.rows[0].ignored);
    assert!(matrix.rows[1].ignored);
    for row in [
        "",
        "plain: (\"recording\", CELL);",
        "#[tokio::test] plain: (scenario(), CELL);",
        "#[tokio::test] plain: (\"recording\", CELL);",
        "#[tokio::test] plain: (\"recording\", CELL, golden);",
        "#[tokio::test] plain: (\"recording\", CELL, \"golden\", extra);",
    ] {
        assert!(syn::parse_str::<NativeMatrix>(&format!("{header}{row}")).is_err());
    }
}

#[test]
fn scripted_world_rows_keep_literal_goldens_and_ignores() {
    let header = "family: wire_matrix_case;";
    let matrix: CaseMatrix = syn::parse_str(&format!(
        r#"{header}
        #[tokio::test] text: truncated_after_text_0 => "text";
        #[tokio::test] #[ignore = "unrecorded"] absent: filtered_empty_4 => "absent";
    "#
    ))
    .expect("scripted world rows");
    assert_eq!(matrix.registrations, 2);
    assert_eq!(matrix.family, "wire_matrix_case");
    assert!(matrix.rows.is_empty());
    assert_eq!(matrix.world_goldens[0].0.value(), "text");
    assert!(!matrix.world_goldens[0].1);
    assert!(matrix.world_goldens[1].1);
    for row in [
        "#[tokio::test] text: truncated_after_text_0 => dynamic();",
        "#[tokio::test] text: truncated_after_text_0 =>;",
        "#[tokio::test] text: truncated_after_text_0 => \"text\", extra;",
    ] {
        assert!(syn::parse_str::<CaseMatrix>(&format!("{header}{row}")).is_err());
    }
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
