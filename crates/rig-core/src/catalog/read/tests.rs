//! The strict reader reports every mistake in an override file with its
//! path and a close alternative; the lenient one reads what it can of a
//! models.dev file and lists what it skipped.

use super::*;

fn errors(json: &str) -> Vec<OverrideError> {
    match Catalog::from_overrides(json, Catalog::builtin()) {
        Ok(_) => panic!("{json} should not read"),
        Err(OverrideErrors(errors)) => errors,
    }
}

fn error(path: &str, kind: OverrideErrorKind) -> OverrideError {
    OverrideError {
        path: path.to_owned(),
        kind,
    }
}

fn unknown_field(suggestion: Option<&'static str>) -> OverrideErrorKind {
    OverrideErrorKind::UnknownField { suggestion }
}

#[test]
fn a_misspelt_vendor_is_an_error_with_the_closest_key() {
    assert_eq!(
        errors(r#"{"antropic": {"models": {}}, "amazon-bedrok": {"models": {}}, "acme": {}}"#),
        [
            error(
                "antropic",
                OverrideErrorKind::UnknownVendor {
                    suggestion: Some("anthropic")
                }
            ),
            error(
                "amazon-bedrok",
                OverrideErrorKind::UnknownVendor {
                    suggestion: Some("amazon-bedrock")
                }
            ),
            error(
                "acme",
                OverrideErrorKind::UnknownVendor { suggestion: None }
            ),
        ]
    );
}

/// Every unknown key is reported in one pass, at every depth of a row, and
/// the row is not applied.
#[test]
fn every_unknown_field_is_reported_with_its_path() {
    let found = errors(
        r#"{"anthropic": {"model": {}, "models": {"claude-opus-5-5": {
            "limits": {"output": 1},
            "limit": {"contxt": 1},
            "cost": {"cache_reed": 1, "inptu": 1},
            "modalities": {"input": ["text"], "output": ["text"]},
            "reasoning_options": [{"type": "effort", "valus": ["low"]}],
            "rig": {"binds_contxt": true}
        }}}}"#,
    );
    assert_eq!(
        found,
        [
            error("anthropic.model", unknown_field(Some("models"))),
            error(
                "anthropic.models.claude-opus-5-5.limits",
                unknown_field(Some("limit"))
            ),
            error(
                "anthropic.models.claude-opus-5-5.limit.contxt",
                unknown_field(Some("context"))
            ),
            error(
                "anthropic.models.claude-opus-5-5.cost.cache_reed",
                unknown_field(Some("cache_read"))
            ),
            // Two swapped letters are one edit.
            error(
                "anthropic.models.claude-opus-5-5.cost.inptu",
                unknown_field(Some("input"))
            ),
            error(
                "anthropic.models.claude-opus-5-5.modalities.output",
                unknown_field(None)
            ),
            error(
                "anthropic.models.claude-opus-5-5.reasoning_options.0.valus",
                unknown_field(Some("values"))
            ),
            error(
                "anthropic.models.claude-opus-5-5.rig.binds_contxt",
                unknown_field(Some("binds_context"))
            ),
        ]
    );
}

#[test]
fn a_value_of_the_wrong_type_names_its_path() {
    let found =
        errors(r#"{"anthropic": {"models": {"claude-opus-5-5": {"limit": {"context": "big"}}}}}"#);
    assert_eq!(found.len(), 1, "{found:?}");
    assert_eq!(
        found[0].path,
        "anthropic.models.claude-opus-5-5.limit.context"
    );
    assert!(matches!(found[0].kind, OverrideErrorKind::InvalidValue(_)));

    for not_a_catalog in ["", "[]", "{\"anthropic\": "] {
        let found = errors(not_a_catalog);
        assert_eq!(found.len(), 1, "{not_a_catalog:?}");
        assert_eq!(found[0].path, "", "{not_a_catalog:?}");
    }
}

/// A row for a model the base does not list adds it, so it must say what
/// the model takes; a misspelt id or a snapshot id is caught with the id it
/// most likely means.
#[test]
fn a_new_model_names_what_it_takes() {
    let new_model = |suggestion: Option<&str>| OverrideErrorKind::NewModel {
        missing: vec!["reasoning", "tool_call", "modalities"],
        suggestion: suggestion.map(str::to_owned),
    };
    assert_eq!(
        errors(
            r#"{"anthropic": {"models": {
                "claude-opus-5.5": {"cost": {"input": 4}},
                "claude-opus-5-5-20260601": {"cost": {"input": 4}}
            }}}"#
        ),
        [
            error(
                "anthropic.models.claude-opus-5.5",
                new_model(Some("claude-opus-5-5"))
            ),
            error(
                "anthropic.models.claude-opus-5-5-20260601",
                new_model(Some("claude-opus-5-5"))
            ),
        ]
    );

    let catalog = Catalog::from_overrides(
        r#"{"anthropic": {"models": {
            "claude-opus-5-5": {"cost": {"input": 4}},
            "claude-house": {"reasoning": false, "tool_call": true, "modalities": {"input": ["text"]}}
        }}}"#,
        Catalog::builtin(),
    )
    .expect("a listed model and a complete new one read");
    let ids: Vec<&str> = catalog.iter().map(|spec| spec.id.as_str()).collect();
    assert_eq!(ids, ["claude-house", "claude-opus-5-5"]);
}

#[test]
fn the_message_says_where_and_what_was_meant() {
    let error = Catalog::from_overrides(
        r#"{"antropic": {"models": {}}, "anthropic": {"models": {"claude-opus-5.5": {}}}}"#,
        Catalog::builtin(),
    )
    .expect_err("two mistakes");
    let message = error.to_string();
    assert!(message.starts_with("the catalog override file has 2 errors:"));
    assert!(
        message.contains("`antropic`: unknown vendor; did you mean `anthropic`?"),
        "{message}"
    );
    assert!(
        message.contains(
            "`anthropic.models.claude-opus-5.5`: not a listed model; did you mean \
             `claude-opus-5-5`?; to add it, set `reasoning`, `tool_call`, `modalities`"
        ),
        "{message}"
    );
}

/// models.dev's file reads whole: unknown providers, sections and rows that
/// do not read are skipped and listed, and keys rig does not read are
/// ignored.
#[test]
fn the_lenient_reader_skips_what_it_cannot_use() {
    let (catalog, skipped) = Catalog::from_models_dev(
        r#"{
            "acme": {"id": "acme", "models": {"m": {}}},
            "anthropic": {"id": "anthropic", "env": ["ANTHROPIC_API_KEY"], "models": {
                "claude-x": {"name": "Claude X", "release_date": "2026-01-01", "limit": {"context": 1, "input": 1}, "rig": {"later": 1}},
                "claude-bad": {"limit": {"context": "big"}}
            }},
            "groq": {"name": "Groq"}
        }"#,
    )
    .expect("an object reads");
    let ids: Vec<&str> = catalog.iter().map(|spec| spec.id.as_str()).collect();
    assert_eq!(ids, ["claude-x"]);
    let skipped: Vec<(&str, bool)> = skipped
        .iter()
        .map(|skip| (skip.path.as_str(), skip.reason == SkipReason::UnknownVendor))
        .collect();
    assert_eq!(
        skipped,
        [
            ("acme", true),
            ("anthropic.models.claude-bad", false),
            ("groq", false)
        ]
    );
    assert!(Catalog::from_models_dev("[]").is_err());
}
