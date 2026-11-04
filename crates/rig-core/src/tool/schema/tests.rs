use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

use super::args_schema;
use crate::message::{ToolCall, ToolFunction, ToolName};
use crate::tool::{decode_args, feedback::arguments_refusal, validate::validate};

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
struct Row {
    value: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum Mode {
    On,
    Off,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
struct Collections {
    samples: Vec<Option<u32>>,
    updates: BTreeMap<String, Option<String>>,
    pairs: Vec<(Option<bool>, Option<u32>)>,
    rows: Vec<Option<Row>>,
    modes: Vec<Option<Mode>>,
    nested: Option<Vec<Option<u32>>>,
}

#[test]
fn nullable_collection_values_pass_the_generated_schema_and_dispatch_check() {
    let schema = args_schema::<Collections>();
    let name = ToolName::new("record");
    assert!(name.is_ok());
    let Ok(name) = name else { return };
    for args in [
        json!({
            "samples": [1, null, 3],
            "updates": {"nickname": null, "name": "Ada"},
            "pairs": [[null, 1], [true, null]],
            "rows": [null, {"value": 4}, {}],
            "modes": ["on", null, "off"],
            "nested": [null, 2],
        }),
        json!({"samples": [], "updates": {}, "pairs": [], "rows": [], "modes": []}),
    ] {
        assert!(decode_args::<Collections>(&args).is_ok());
        assert_eq!(validate(&schema, &args), Ok(()), "{schema}");
        let call = ToolCall::from_wire("call-1", ToolFunction::new(name.clone(), args));
        assert_eq!(arguments_refusal(&schema, &call), None);
    }
    assert_eq!(
        schema.pointer("/properties/nested/type"),
        Some(&json!("array"))
    );
    assert!(
        validate(
            &schema,
            &json!({
                "samples": [], "updates": {}, "pairs": [], "modes": [],
                "rows": [{"unexpected": true}],
            }),
        )
        .is_err(),
        "nullable row objects must still reject undeclared keys",
    );
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct NonNullableCollections {
    samples: Vec<u32>,
    updates: BTreeMap<String, String>,
}

#[test]
fn non_nullable_collection_values_still_reject_null() {
    let schema = args_schema::<NonNullableCollections>();
    for args in [
        json!({"samples": [null], "updates": {}}),
        json!({"samples": [], "updates": {"nickname": null}}),
    ] {
        assert!(validate(&schema, &args).is_err());
        assert!(decode_args::<NonNullableCollections>(&args).is_err());
    }
}

#[test]
fn required_nullable_properties_keep_null() {
    let mut schema = schemars::json_schema!({
        "type": "object",
        "properties": {
            "required": {"type": ["string", "null"]},
            "optional": {"type": ["string", "null"]}
        },
        "required": ["required"]
    });
    super::plain(&mut schema);
    assert_eq!(
        schema.as_value().pointer("/properties/required/type"),
        Some(&json!(["string", "null"])),
    );
    assert_eq!(
        schema.as_value().pointer("/properties/optional/type"),
        Some(&json!("string")),
    );
}
