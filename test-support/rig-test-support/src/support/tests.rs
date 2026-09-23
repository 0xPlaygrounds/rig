use super::{
    SmokePerson, SmokeStructuredOutput, ecs_synthetic_output_tool_name,
    matches_recorded_document_in,
};
use crate::cassettes::CassetteMode;
use serde_json::json;

// These fixed schemas predate the support-crate extraction. In particular, adding
// Rust documentation must not alter recorded requests or synthetic output-tool IDs.
#[test]
fn structured_smoke_schema_preserves_recorded_shape() {
    assert_eq!(
        schemars::schema_for!(SmokeStructuredOutput).as_value(),
        &json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "SmokeStructuredOutput",
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "category": {"type": "string"},
                "summary": {"type": "string"}
            },
            "required": ["title", "category", "summary"]
        })
    );
    assert_eq!(
        ecs_synthetic_output_tool_name::<SmokeStructuredOutput>(),
        "__rig_output_8d3dd766"
    );
}

#[test]
fn person_smoke_schema_preserves_recorded_shape() {
    assert_eq!(
        schemars::schema_for!(SmokePerson).as_value(),
        &json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "SmokePerson",
            "type": "object",
            "properties": {
                "first_name": {"type": "string"},
                "last_name": {"type": "string"},
                "job": {"type": "string"}
            },
            "required": ["first_name", "last_name", "job"]
        })
    );
    assert_eq!(
        ecs_synthetic_output_tool_name::<SmokePerson>(),
        "__rig_output_8ea10729"
    );
}

fn document_matches(
    mode: CassetteMode,
    live: &serde_json::Value,
    recorded: &serde_json::Value,
) -> bool {
    std::panic::catch_unwind(|| matches_recorded_document_in(mode, live, recorded, &["id"], "doc"))
        .is_ok()
}

#[test]
fn a_recorded_document_compares_volatile_keys_by_type_only_when_recording() {
    let recorded = json!({"id": "resp_1", "created": 0, "model": "m", "object": "chat.completion"});
    let live = json!({"id": "resp_2", "created": 1_790_000_000, "model": "m", "object": "chat.completion"});
    // A recording pass sees the live timestamp; the fixture holds the normalized one.
    assert!(document_matches(CassetteMode::Record, &live, &recorded));
    assert!(!document_matches(CassetteMode::Replay, &live, &recorded));
    assert!(document_matches(CassetteMode::Replay, &recorded, &recorded));

    // Every other key is exact in both modes, and so are the key sets.
    let other_model =
        json!({"id": "resp_2", "created": 1, "model": "n", "object": "chat.completion"});
    assert!(!document_matches(
        CassetteMode::Record,
        &other_model,
        &recorded
    ));
    let wrong_type =
        json!({"id": "resp_2", "created": "1", "model": "m", "object": "chat.completion"});
    assert!(!document_matches(
        CassetteMode::Record,
        &wrong_type,
        &recorded
    ));
    let extra_key =
        json!({"id": "resp_2", "created": 1, "model": "m", "object": "chat.completion", "x": 1});
    assert!(!document_matches(
        CassetteMode::Record,
        &extra_key,
        &recorded
    ));
}

#[test]
fn nested_volatile_keys_are_compared_by_type_when_recording() {
    let recorded = json!({"data": [{"created_at": 0, "id": "a"}], "meta": {"Updated": "1970-01-01T00:00:00Z"}});
    let live = json!({"data": [{"created_at": 1_790_000_000, "id": "a"}], "meta": {"Updated": "2026-09-23T00:00:00Z"}});
    assert!(document_matches(CassetteMode::Record, &live, &recorded));
    assert!(!document_matches(CassetteMode::Replay, &live, &recorded));
    let other_nested = json!({"data": [{"created_at": 1, "id": "b"}], "meta": {"Updated": "x"}});
    assert!(!document_matches(
        CassetteMode::Record,
        &other_nested,
        &recorded
    ));
    let longer = json!({"data": [{"created_at": 1, "id": "a"}, {}], "meta": {"Updated": "x"}});
    assert!(!document_matches(CassetteMode::Record, &longer, &recorded));
}

#[test]
fn a_nested_difference_names_its_path() {
    let recorded = json!({"data": [{"created_at": 0}]});
    let live = json!({"data": [{"created_at": "1"}]});
    let message = std::panic::catch_unwind(|| {
        matches_recorded_document_in(CassetteMode::Record, &live, &recorded, &[], "doc")
    })
    .expect_err("a type change fails")
    .downcast::<String>()
    .map(|message| *message)
    .unwrap_or_default();
    assert!(message.contains("`.data[0].created_at`"), "{message}");
}
