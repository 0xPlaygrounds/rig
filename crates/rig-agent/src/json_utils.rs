//! Runtime-local JSON helpers used by hook patching and tool serialization.

pub(crate) fn merge(a: serde_json::Value, b: serde_json::Value) -> serde_json::Value {
    match (a, b) {
        (serde_json::Value::Object(mut a_map), serde_json::Value::Object(b_map)) => {
            for (key, value) in b_map {
                a_map.insert(key, value);
            }
            serde_json::Value::Object(a_map)
        }
        (a, _) => a,
    }
}

pub(crate) fn serialize_json_value(value: &serde_json::Value) -> String {
    match serde_json::to_string(value) {
        Ok(serialized) => serialized,
        Err(_) => value.to_string(),
    }
}
