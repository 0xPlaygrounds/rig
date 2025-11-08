//! Utilities for converting between serde_json::Value and wkt::Struct (protobuf Struct)
//!
//! Note: wkt::Struct is a type alias for serde_json::Map<String, serde_json::Value>

use rig::completion::CompletionError;

/// Type alias for wkt::Struct (which is serde_json::Map<String, serde_json::Value>)
pub type Struct = serde_json::Map<String, serde_json::Value>;

/// Convert serde_json::Value to Struct
///
/// Struct is a type alias for serde_json::Map<String, serde_json::Value>
/// so we can convert directly if the value is an object
pub fn json_to_struct(value: serde_json::Value) -> Result<Struct, CompletionError> {
    match value {
        serde_json::Value::Object(map) => Ok(map),
        _ => Err(CompletionError::ProviderError(
            "Expected JSON object for Struct conversion".to_string(),
        )),
    }
}

/// Convert Struct to serde_json::Value
///
/// Struct is a type alias for serde_json::Map<String, serde_json::Value>
pub fn struct_to_json(struct_val: Struct) -> serde_json::Value {
    serde_json::Value::Object(struct_val)
}
