use super::sanitize_schema;
use serde_json::json;

#[test]
fn test_sanitize_strips_ref_sibling_keywords() {
    let mut schema = json!({
        "type": "object",
        "properties": {
            "location": {
                "$ref": "#/$defs/Location",
                "description": "The user's location"
            }
        },
        "$defs": {
            "Location": {
                "type": "object",
                "properties": {
                    "city": { "type": "string" },
                    "state": { "type": "string" }
                }
            }
        }
    });

    sanitize_schema(&mut schema);

    // $ref node should only contain "$ref", no "description"
    let location = &schema["properties"]["location"];
    assert_eq!(location, &json!({ "$ref": "#/$defs/Location" }));

    // The referenced $def should still be fully sanitized
    let location_def = &schema["$defs"]["Location"];
    assert_eq!(location_def["additionalProperties"], json!(false));
    assert!(location_def["required"].as_array().is_some());
}

#[test]
fn test_sanitize_adds_additional_properties_false() {
    let mut schema = json!({
        "type": "object",
        "properties": {
            "name": { "type": "string" }
        }
    });

    sanitize_schema(&mut schema);

    assert_eq!(schema["additionalProperties"], json!(false));
}

#[test]
fn test_sanitize_marks_all_properties_required() {
    let mut schema = json!({
        "type": "object",
        "properties": {
            "a": { "type": "string" },
            "b": { "type": "number" }
        }
    });

    sanitize_schema(&mut schema);

    let required = schema["required"].as_array().unwrap();
    assert!(required.contains(&json!("a")));
    assert!(required.contains(&json!("b")));
    assert_eq!(required.len(), 2);
}

#[test]
fn test_sanitize_converts_one_of_to_any_of() {
    let mut schema = json!({
        "oneOf": [
            { "type": "string" },
            { "type": "number" }
        ]
    });

    sanitize_schema(&mut schema);

    assert!(schema.get("oneOf").is_none());
    assert!(schema["anyOf"].as_array().is_some());
}

#[test]
fn test_sanitize_recurses_into_nested_objects() {
    let mut schema = json!({
        "type": "object",
        "properties": {
            "inner": {
                "type": "object",
                "properties": {
                    "value": { "type": "string" }
                }
            }
        }
    });

    sanitize_schema(&mut schema);

    assert_eq!(
        schema["properties"]["inner"]["additionalProperties"],
        json!(false)
    );
    let inner_required = schema["properties"]["inner"]["required"]
        .as_array()
        .unwrap();
    assert!(inner_required.contains(&json!("value")));
}
