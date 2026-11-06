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

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct Nullable {
    samples: Vec<Option<u32>>,
    updates: std::collections::BTreeMap<String, Option<String>>,
    pair: (Option<bool>, u32),
    limit: Option<Vec<Option<u32>>>,
}

#[test]
fn only_an_omittable_property_loses_its_nullability() {
    let schema = args_schema::<Nullable>();
    let value = json!({"samples": [1, null], "updates": {"a": null}, "pair": [null, 2],
        "limit": [null]});
    assert_eq!(validate(&schema, &value), Ok(()), "{schema}");
    assert_eq!(
        schema["properties"]["limit"]["type"],
        json!("array"),
        "{schema}"
    );
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
fn an_integer_is_a_whole_number_a_64_bit_integer_holds() {
    let offset =
        json!({"type": "object", "properties": {"offset": {"type": "integer", "minimum": 0}}});
    for whole in [json!(1), json!(1.0), json!(30.0), json!(1e19)] {
        assert_eq!(
            problems(&offset, &json!({"offset": whole})),
            Vec::<String>::new()
        );
    }
    let refused = |offset_value: Value| problems(&offset, &json!({"offset": offset_value}));
    assert_eq!(
        refused(json!(1.5)),
        ["/offset: expected an integer, got a number with a fraction"]
    );
    assert_eq!(refused(json!(-1)), ["/offset: expected at least 0, got -1"]);
    assert_eq!(
        refused(json!(1e30)),
        ["/offset: expected an integer, got a number out of an integer's range"]
    );
    let text = json!({"type": "string"});
    assert_eq!(
        problems(&text, &json!(1e30)),
        ["expected a string, got a number"]
    );
}

#[test]
fn bounds_and_const_compare_large_integers_exactly() {
    let below = json!(9_007_199_254_740_992_u64);
    let above = json!(9_007_199_254_740_993_u64);
    let fits = |schema: Value, value: &Value| validate(&schema, value).is_ok();
    assert!(!fits(json!({"maximum": below}), &above));
    assert!(fits(json!({"exclusiveMinimum": below}), &above));
    assert!(!fits(json!({"minimum": above}), &below));
    assert!(fits(
        json!({"exclusiveMaximum": 18_446_744_073_709_551_616.0}),
        &json!(u64::MAX)
    ));
    assert!(fits(json!({"exclusiveMinimum": -3.5}), &json!(-3)));
    assert!(fits(json!({"maximum": 5.0}), &json!(5)));
    assert!(!fits(json!({"const": 9_007_199_254_740_992.0}), &above));
    assert!(fits(json!({"enum": [5.0]}), &json!(5)));

    // What `arbitrary_precision` compares, reachable in any build.
    use crate::json_utils::compare_literals;
    use std::cmp::Ordering::{Equal, Greater, Less};
    let far = "9".repeat(41);
    let wide = [
        (format!("10e{far}"), format!("1e1{}", "0".repeat(41)), Equal),
        (format!("1e-{far}"), format!("2e-{far}"), Less),
    ];
    let literals = [
        ("9007199254740993", "9007199254740993.0", Equal),
        (
            "123456789012345678901234567890",
            "1.2345678901234567890123456789e29",
            Equal,
        ),
        ("1e400", "1.0E+400", Equal),
        ("0", "-0.0e5", Equal),
        ("0", "0e9223372036854775808", Equal),
        ("18446744073709551617.0", "18446744073709551616", Greater),
        (
            "170141183460469231731687303715884105729",
            "170141183460469231731687303715884105728",
            Greater,
        ),
        ("5", "5.0000000000000000000001", Less),
        ("0.1", "0.01", Greater),
        ("-3", "-3.5", Greater),
        ("-1", "1e-400", Less),
        ("1e-9223372036854775809", "0", Greater),
        ("1e99999999999999999999", "5", Greater),
        ("-1e99999999999999999999", "-5", Less),
    ];
    let wide = wide
        .iter()
        .map(|(a, b, order)| (a.as_str(), b.as_str(), *order));
    for (a, b, order) in literals.into_iter().chain(wide) {
        assert_eq!(compare_literals(a, b), order, "{a} against {b}");
        assert_eq!(compare_literals(b, a), order.reverse(), "{b} against {a}");
    }
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
