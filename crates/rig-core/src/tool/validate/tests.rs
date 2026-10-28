use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use super::*;
use crate::tool::args_schema;

/// Each problem as `at: what`.
fn problems(schema: &Value, value: &Value) -> Vec<String> {
    validate(schema, value)
        .err()
        .unwrap_or_default()
        .iter()
        .map(Problem::to_string)
        .collect()
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
#[serde(rename_all = "snake_case")]
enum Mode {
    Fast,
    Careful,
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct Edit {
    old_text: String,
    new_text: String,
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
#[serde(untagged)]
enum Count {
    Exact(u32),
    Named(String),
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct Args {
    path: String,
    limit: Option<u32>,
    mode: Mode,
    edits: Vec<Edit>,
    first: Option<Edit>,
    order: Option<Mode>,
    count: Option<Count>,
}

#[test]
fn an_args_schema_accepts_well_typed_arguments() {
    let schema = args_schema::<Args>();
    assert!(
        !schema.to_string().contains("null"),
        "an `Option` is optional, not nullable: {schema}"
    );
    assert_eq!(
        schema["properties"]["count"],
        json!({"anyOf": [{"type": "integer", "minimum": 0}, {"type": "string"}]})
    );
    for value in [
        json!({"path": "a.rs", "mode": "fast", "edits": []}),
        json!({"path": "a.rs", "limit": 3, "mode": "careful",
            "edits": [{"old_text": "a", "new_text": "b"}],
            "first": {"old_text": "a", "new_text": "b"}, "order": "fast", "count": "all"}),
        json!({"path": "a.rs", "limit": null, "mode": "fast", "edits": [], "first": null}),
    ] {
        assert_eq!(validate(&schema, &value), Ok(()), "{value}");
    }
}

#[test]
fn problems_are_reported_at_their_pointers() {
    let schema = args_schema::<Args>();
    let value = json!({"limit": -1.5, "mode": "slow",
        "edits": [{"old_text": 3, "new_text": "b", "extra": true}],
        "first": "a"});
    let edit = args_schema::<Edit>();
    let allowed = edit
        .get("properties")
        .and_then(Value::as_object)
        .expect("edit properties")
        .keys()
        .map(|key| format!("`{key}`"))
        .collect::<Vec<_>>()
        .join(", ");
    let mut found = problems(&schema, &value);
    found.sort();
    let mut expected = vec![
        "/limit: expected an integer, got a number with a fraction".to_owned(),
        "/mode: expected one of \"fast\", \"careful\", got \"slow\"".to_owned(),
        "/edits/0/old_text: expected a string, got a number".to_owned(),
        format!("/edits/0/extra: not allowed; the keys are {allowed}"),
        "/first: expected an object, got a string".to_owned(),
        "/path: missing".to_owned(),
    ];
    expected.sort();
    assert_eq!(found, expected,);
    let bounded = json!({"type": "array", "maxItems": 1,
        "items": {"type": "string", "minLength": 2}});
    assert_eq!(
        problems(&bounded, &json!(["a", "bc"])),
        [
            "expected at most 1 item, got 2",
            "/0: expected at least 2 characters, got 1"
        ]
    );
    assert_eq!(
        problems(&json!({"minimum": 0, "exclusiveMaximum": 10}), &json!(10)),
        ["expected less than 10, got 10"]
    );
}

#[test]
fn combinators_and_local_refs_are_checked_and_a_remote_ref_accepts() {
    let either = json!({"anyOf": [{"type": "string"}, {"type": "integer"}]});
    assert_eq!(validate(&either, &json!(3)), Ok(()));
    assert_eq!(
        problems(&either, &json!(true)),
        ["expected a string or an integer, got a boolean"]
    );
    let one = json!({"oneOf": [{"type": "number"}, {"type": "integer"}]});
    assert_eq!(validate(&one, &json!(1.5)), Ok(()));
    assert_eq!(
        problems(&one, &json!(1)),
        ["fits 2 of its `oneOf` alternatives, not exactly one"]
    );

    // A recursive draft 2020-12 tree, as schemars 1.x emits it.
    let tree = json!({
        "$ref": "#/$defs/node",
        "$defs": {"node": {"type": "object", "required": ["name"],
            "properties": {"name": {"type": "string"},
                "children": {"type": "array", "items": {"$ref": "#/$defs/node"}}}}},
    });
    assert_eq!(
        validate(&tree, &json!({"name": "a", "children": [{"name": "b"}]})),
        Ok(())
    );
    assert_eq!(
        problems(&tree, &json!({"name": "a", "children": [{"children": []}]})),
        ["/children/0/name: missing"]
    );

    // A draft-07 schema, as most MCP servers send.
    let mcp = json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {"range": {"$ref": "#/definitions/Range"}},
        "definitions": {"Range": {"type": "object",
            "properties": {"start": {"type": "integer", "minimum": 0}}}},
    });
    assert_eq!(
        problems(&mcp, &json!({"range": {"start": -1}})),
        ["/range/start: expected at least 0, got -1"]
    );

    // A remote, an unresolvable and a self-looping ref accept.
    for schema in [
        json!({"$ref": "https://example.com/schema.json"}),
        json!({"$ref": "#/$defs/missing"}),
        json!({"$ref": "#"}),
    ] {
        assert_eq!(validate(&schema, &json!(42)), Ok(()), "{schema}");
    }
}
