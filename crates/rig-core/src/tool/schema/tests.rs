use super::*;
use crate::tool::validate::validate;
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct Args {
    path: String,
    #[serde(flatten)]
    operation: Operation,
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
#[serde(tag = "op", rename_all = "snake_case")]
enum Operation {
    Read { limit: usize },
    Write { edit: Edit },
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct Edit {
    text: String,
}

#[test]
fn flattened_tagged_enums_accept_each_variant_and_keep_field_constraints() {
    let schema = args_schema::<Args>();
    for value in [
        json!({"path": "a", "op": "read", "limit": 3}),
        json!({"path": "a", "op": "write", "edit": {"text": "hello"}}),
    ] {
        assert!(serde_json::from_value::<Args>(value.clone()).is_ok());
        assert_eq!(validate(&schema, &value), Ok(()));
    }
    for value in [
        json!({"op": "read", "limit": 3}),
        json!({"path": 1, "op": "read", "limit": 3}),
        json!({"path": "a", "op": "read"}),
        json!({"path": "a", "op": "read", "limit": "3"}),
        json!({"path": "a", "op": "unknown", "limit": 3}),
        json!({"path": "a", "op": "write", "edit": {"text": "hello", "extra": true}}),
    ] {
        assert!(
            validate(&schema, &value).is_err(),
            "accepted {value}: {schema}"
        );
    }
}

#[test]
fn standalone_enum_variants_and_objects_still_reject_unknown_keys() {
    assert!(
        validate(
            &args_schema::<Operation>(),
            &json!({"op": "read", "limit": 3, "extra": 1})
        )
        .is_err()
    );
    assert!(validate(&args_schema::<Edit>(), &json!({"text": "ok", "extra": 1})).is_err());
}

#[test]
fn composed_fragments_keep_explicit_constraints_and_maps() {
    let mut schema = schemars::json_schema!({
        "allOf": [
            {"properties": {"path": {"type": "string"}}, "required": ["path"]},
            {"properties": {"limit": {"type": "integer"}}, "required": ["limit"]}
        ]
    });
    close_objects(&mut schema, false);
    assert_eq!(
        validate(schema.as_value(), &json!({"path": "a", "limit": 2})),
        Ok(())
    );
    assert!(validate(schema.as_value(), &json!({"path": "a", "limit": "2"})).is_err());

    let mut siblings = schemars::json_schema!({
        "anyOf": [{"properties": {"path": {"type": "string"}}, "required": ["path"]}],
        "oneOf": [{"properties": {"limit": {"type": "integer"}}, "required": ["limit"]}]
    });
    close_objects(&mut siblings, false);
    assert_eq!(
        validate(siblings.as_value(), &json!({"path": "a", "limit": 2})),
        Ok(())
    );

    let mut explicit = schemars::json_schema!({
        "properties": {"path": {"type": "string"}},
        "additionalProperties": false,
        "anyOf": [{"required": ["path"]}]
    });
    close_objects(&mut explicit, false);
    assert!(validate(explicit.as_value(), &json!({"path": "a", "extra": 1})).is_err());
    let map = args_schema::<std::collections::BTreeMap<String, String>>();
    assert_eq!(validate(&map, &json!({"arbitrary": "value"})), Ok(()));
}
