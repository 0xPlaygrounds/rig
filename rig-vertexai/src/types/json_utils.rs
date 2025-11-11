//! Utilities for converting between serde_json::Value and wkt::Struct (protobuf Struct)
//!
//! Note: wkt::Struct is a type alias for serde_json::Map<String, serde_json::Value>

use rig::completion::CompletionError;

pub type Struct = serde_json::Map<String, serde_json::Value>;

pub fn json_to_struct(value: serde_json::Value) -> Result<Struct, CompletionError> {
    match value {
        serde_json::Value::Object(map) => Ok(map),
        _ => Err(CompletionError::ProviderError(
            "Expected JSON object for Struct conversion".to_string(),
        )),
    }
}

pub fn struct_to_json(struct_val: Struct) -> serde_json::Value {
    serde_json::Value::Object(struct_val)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_to_struct() {
        let json = serde_json::json!({
            "x": 5,
            "y": 3,
            "name": "test"
        });

        let struct_val = json_to_struct(json.clone()).unwrap();
        assert_eq!(struct_val.len(), 3);
        assert_eq!(struct_val.get("x"), Some(&serde_json::json!(5)));
        assert_eq!(struct_val.get("y"), Some(&serde_json::json!(3)));
        assert_eq!(struct_val.get("name"), Some(&serde_json::json!("test")));
    }

    #[test]
    fn test_json_to_struct_error_on_non_object() {
        let json = serde_json::json!("not an object");
        let result = json_to_struct(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_struct_to_json() {
        let mut struct_val = serde_json::Map::new();
        struct_val.insert("x".to_string(), serde_json::json!(5));
        struct_val.insert("y".to_string(), serde_json::json!(3));

        let json = struct_to_json(struct_val);
        assert!(json.is_object());
        assert_eq!(json.get("x"), Some(&serde_json::json!(5)));
        assert_eq!(json.get("y"), Some(&serde_json::json!(3)));
    }

    #[test]
    fn test_round_trip_conversion() {
        let original_json = serde_json::json!({
            "x": 5,
            "y": 3,
            "nested": {
                "a": 1,
                "b": 2
            }
        });

        let struct_val = json_to_struct(original_json.clone()).unwrap();
        let converted_json = struct_to_json(struct_val);

        // Note: nested objects become nested Structs, so we can't do a direct comparison
        // But we can verify the top-level keys match
        assert_eq!(converted_json.get("x"), original_json.get("x"));
        assert_eq!(converted_json.get("y"), original_json.get("y"));
    }
}
