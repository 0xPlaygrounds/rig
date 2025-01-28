pub fn merge(a: serde_json::Value, b: serde_json::Value) -> serde_json::Value {
    match (a, b) {
        (serde_json::Value::Object(mut a_map), serde_json::Value::Object(b_map)) => {
            b_map.into_iter().for_each(|(key, value)| {
                a_map.insert(key, value);
            });
            serde_json::Value::Object(a_map)
        }
        (a, _) => a,
    }
}

pub fn merge_inplace(a: &mut serde_json::Value, b: serde_json::Value) {
    if let (serde_json::Value::Object(a_map), serde_json::Value::Object(b_map)) = (a, b) {
        b_map.into_iter().for_each(|(key, value)| {
            a_map.insert(key, value);
        });
    }
}

/// This module is helpful in cases where raw json objects are serialized and deserialized as
///  strings such as `"{\"key\": \"value\"}"`. This might seem odd but it's actually how some
///  some providers such as OpenAI return function arguments (for some reason).
pub mod stringified_json {
    use serde::{self, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(value: &serde_json::Value, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let s = value.to_string();
        serializer.serialize_str(&s)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<serde_json::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        serde_json::from_str(&s).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct Dummy {
        #[serde(with = "stringified_json")]
        data: serde_json::Value,
    }

    #[test]
    fn test_merge() {
        let a = serde_json::json!({"key1": "value1"});
        let b = serde_json::json!({"key2": "value2"});
        let result = merge(a, b);
        let expected = serde_json::json!({"key1": "value1", "key2": "value2"});
        assert_eq!(result, expected);
    }

    #[test]
    fn test_merge_inplace() {
        let mut a = serde_json::json!({"key1": "value1"});
        let b = serde_json::json!({"key2": "value2"});
        merge_inplace(&mut a, b);
        let expected = serde_json::json!({"key1": "value1", "key2": "value2"});
        assert_eq!(a, expected);
    }

    #[test]
    fn test_stringified_json_serialize() {
        let dummy = Dummy {
            data: serde_json::json!({"key": "value"}),
        };
        let serialized = serde_json::to_string(&dummy).unwrap();
        let expected = r#"{"data":"{\"key\":\"value\"}"}"#;
        assert_eq!(serialized, expected);
    }

    #[test]
    fn test_stringified_json_deserialize() {
        let json_str = r#"{"data":"{\"key\":\"value\"}"}"#;
        let dummy: Dummy = serde_json::from_str(json_str).unwrap();
        let expected = Dummy {
            data: serde_json::json!({"key": "value"}),
        };
        assert_eq!(dummy, expected);
    }
}
