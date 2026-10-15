use schemars::JsonSchema;
use serde::Serialize;
use serde_json::json;

use super::*;

/// Each problem of `value` as a schema, as `at: what`.
fn problems(value: Value) -> Vec<String> {
    Schema::new(value)
        .err()
        .map(|error| error.problems().iter().map(Problem::to_string).collect())
        .unwrap_or_default()
}

#[test]
fn a_well_formed_schema_is_kept_as_written() {
    let value = json!({
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "A file.", "x-provider": 1},
            "lines": {"type": ["array", "null"], "items": {"type": "integer", "minimum": 1}},
            "mode": {"enum": ["fast", "careful"]},
            "pair": {"type": "array", "items": [{"type": "string"}, true]},
        },
        "required": ["path"],
        "additionalProperties": false,
        "$defs": {"node": {"anyOf": [{"$ref": "#/$defs/node"}, {"type": "null"}]}},
    });
    assert_eq!(
        Schema::new(value.clone()).map(Schema::into_value),
        Ok(value)
    );
    assert_eq!(problems(json!(true)), Vec::<String>::new());
}

#[test]
fn every_malformed_keyword_is_reported_where_it_is() {
    assert_eq!(
        problems(json!({
            "type": "object",
            "properties": {
                "path": {"type": "strng"},
                "limit": {"type": "integer", "minimum": "1", "maxLength": -1},
                "tags": {"type": ["string", "string"], "items": 3},
            },
            "required": "path",
            "anyOf": [],
        })),
        [
            "/properties/path/type: unknown type `strng`",
            "/properties/limit/minimum: expected a number, got a string",
            "/properties/limit/maxLength: expected a non-negative integer, got a number",
            "/properties/tags/type: a type is listed twice",
            "/properties/tags/items: expected a schema (an object or a boolean), got a number",
            "/required: expected a list of property names, got a string",
            "/anyOf: expected a non-empty list of schemas, got a list",
        ]
    );
    assert_eq!(
        problems(json!("object")),
        ["expected a schema (an object or a boolean), got a string"]
    );
}

#[test]
fn deserializing_checks_the_schema() {
    let error = serde_json::from_value::<Schema>(json!({"type": 7})).err();
    assert_eq!(
        error.map(|error| error.to_string()),
        Some("not a valid JSON Schema: /type: expected a type name, got a number".to_owned())
    );
}

#[derive(Serialize, JsonSchema)]
#[allow(dead_code)]
struct Weather {
    /// Degrees Celsius.
    celsius: f64,
    note: Option<String>,
}

#[test]
fn an_output_schema_keeps_null_for_an_option() {
    let schema = Schema::of::<Weather>();
    assert_eq!(schema.get("$schema"), None);
    assert_eq!(
        schema["properties"]["note"]["type"],
        json!(["string", "null"])
    );
    assert_eq!(schema["required"], json!(["celsius"]));
}

#[derive(JsonSchema)]
#[allow(dead_code)]
struct NoRequired {
    page: Option<u32>,
}

#[test]
fn an_arguments_schema_lists_required_even_when_empty() {
    assert_eq!(Schema::arguments::<NoRequired>()["required"], json!([]));
}
