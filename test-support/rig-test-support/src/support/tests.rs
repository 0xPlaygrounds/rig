use super::{SmokePerson, SmokeStructuredOutput, ecs_synthetic_output_tool_name};
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
