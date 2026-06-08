use serde::Deserialize;
use serde::de::{self, Deserializer, SeqAccess, Visitor};
use std::convert::Infallible;
use std::fmt;
use std::marker::PhantomData;
use std::str::FromStr;

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

/// Convert a serde_json::Value to a JSON string for tool arguments.
/// Handles the case where vLLM returns arguments as a JSON string (Value::String)
/// instead of a JSON object (Value::Object) like OpenAI does.
pub fn value_to_json_string(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Deserialize a field that may arrive as either a JSON-encoded string or any other
/// JSON value, into `Option<String>`.
///
/// - A string is taken verbatim.
/// - Any other JSON value is re-serialized to its compact JSON-string form (via
///   [`value_to_json_string`]). Object key order is not preserved, which is fine
///   because callers re-parse the string.
/// - `null` or a missing field becomes `None`.
///
/// Tolerates OpenAI-compatible gateways that stream `tool_calls[].function.arguments`
/// as an object (e.g. `{}`) instead of the spec-mandated JSON string (`"{}"`).
pub fn deserialize_json_string_or_value<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    Ok(match value {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => Some(value_to_json_string(&v)),
    })
}

/// Parse tool arguments from a streamed string payload.
/// Some providers emit an empty string for parameterless tool calls; normalize that to `{}`.
pub fn parse_tool_arguments(arguments: &str) -> serde_json::Result<serde_json::Value> {
    if arguments.trim().is_empty() {
        return Ok(serde_json::Value::Object(serde_json::Map::new()));
    }

    serde_json::from_str(arguments)
}

/// This module is helpful in cases where raw json objects are serialized and deserialized as
///  strings such as `"{\"key\": \"value\"}"`. This might seem odd but it's actually how some
///  some providers such as OpenAI return function arguments (for some reason).
pub mod stringified_json {
    use super::parse_tool_arguments;
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
        if s.trim().is_empty() {
            return Ok(serde_json::Value::Object(serde_json::Map::new()));
        }
        serde_json::from_str(&s).map_err(serde::de::Error::custom)
    }

    /// Deserialize JSON that may be encoded either as a string or as a raw JSON value.
    /// OpenAI-compatible providers typically return tool arguments as a stringified JSON
    /// object, while some implementations such as Hugging Face and `llama.cpp` return the
    /// JSON object directly.
    pub fn deserialize_maybe_stringified<'de, D>(
        deserializer: D,
    ) -> Result<serde_json::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        match serde_json::Value::deserialize(deserializer)? {
            serde_json::Value::String(s) => {
                parse_tool_arguments(&s).map_err(serde::de::Error::custom)
            }
            other => Ok(other),
        }
    }
}

pub fn string_or_vec<'de, T, D>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    T: Deserialize<'de> + FromStr<Err = Infallible>,
    D: Deserializer<'de>,
{
    struct StringOrVec<T>(PhantomData<fn() -> T>);

    impl<'de, T> Visitor<'de> for StringOrVec<T>
    where
        T: Deserialize<'de> + FromStr<Err = Infallible>,
    {
        type Value = Vec<T>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a string, sequence, or null")
        }

        fn visit_str<E>(self, value: &str) -> Result<Vec<T>, E>
        where
            E: de::Error,
        {
            let item = FromStr::from_str(value).map_err(de::Error::custom)?;
            Ok(vec![item])
        }

        fn visit_seq<A>(self, seq: A) -> Result<Vec<T>, A::Error>
        where
            A: SeqAccess<'de>,
        {
            Deserialize::deserialize(de::value::SeqAccessDeserializer::new(seq))
        }

        fn visit_none<E>(self) -> Result<Vec<T>, E>
        where
            E: de::Error,
        {
            Ok(vec![])
        }

        fn visit_unit<E>(self) -> Result<Vec<T>, E>
        where
            E: de::Error,
        {
            Ok(vec![])
        }
    }

    deserializer.deserialize_any(StringOrVec(PhantomData))
}

pub fn null_or_vec<'de, T, D>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    T: Deserialize<'de>,
    D: Deserializer<'de>,
{
    struct NullOrVec<T>(PhantomData<fn() -> T>);

    impl<'de, T> Visitor<'de> for NullOrVec<T>
    where
        T: Deserialize<'de>,
    {
        type Value = Vec<T>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a sequence or null")
        }

        fn visit_seq<A>(self, seq: A) -> Result<Vec<T>, A::Error>
        where
            A: SeqAccess<'de>,
        {
            Deserialize::deserialize(de::value::SeqAccessDeserializer::new(seq))
        }

        fn visit_none<E>(self) -> Result<Vec<T>, E>
        where
            E: de::Error,
        {
            Ok(vec![])
        }

        fn visit_unit<E>(self) -> Result<Vec<T>, E>
        where
            E: de::Error,
        {
            Ok(vec![])
        }
    }

    deserializer.deserialize_any(NullOrVec(PhantomData))
}

pub fn null_or_default<'de, T, D>(deserializer: D) -> Result<T, D::Error>
where
    T: Deserialize<'de> + Default,
    D: Deserializer<'de>,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
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

    #[derive(Serialize, Deserialize, Debug, PartialEq)]
    struct DummyMaybeStringified {
        #[serde(deserialize_with = "stringified_json::deserialize_maybe_stringified")]
        data: serde_json::Value,
    }

    #[derive(serde::Deserialize)]
    struct ArgWrapper {
        #[serde(default, deserialize_with = "deserialize_json_string_or_value")]
        arguments: Option<String>,
    }

    /// Spec-compliant case: `arguments` is already a JSON-encoded string, taken verbatim.
    #[test]
    fn json_string_or_value_string_passthrough() {
        let w: ArgWrapper = serde_json::from_str(r#"{"arguments":"{\"a\":1}"}"#).unwrap();
        assert_eq!(w.arguments.as_deref(), Some(r#"{"a":1}"#));
    }

    /// Non-compliant gateway: an empty object `{}` must serialize to the string `"{}"`,
    /// not be treated as absent (None).
    #[test]
    fn json_string_or_value_empty_object() {
        let w: ArgWrapper = serde_json::from_str(r#"{"arguments":{}}"#).unwrap();
        assert_eq!(w.arguments.as_deref(), Some("{}"));
    }

    /// Non-compliant gateway: a nested object is re-serialized to a string.
    #[test]
    fn json_string_or_value_nested_object() {
        let w: ArgWrapper =
            serde_json::from_str(r#"{"arguments":{"path":"/tmp","depth":2}}"#).unwrap();
        // `arguments` is re-serialized from a Value; object key order is not guaranteed
        // (depends on serde_json's `preserve_order` feature), so re-parse and compare
        // values rather than the raw string.
        let parsed: serde_json::Value =
            serde_json::from_str(w.arguments.as_deref().unwrap()).unwrap();
        assert_eq!(parsed["path"], "/tmp");
        assert_eq!(parsed["depth"], 2);
    }

    /// Non-compliant gateway: an array is also "any other JSON value" and serializes to a
    /// string. Array order is meaningful and preserved by serde_json, so compare the string
    /// directly.
    #[test]
    fn json_string_or_value_array() {
        let w: ArgWrapper = serde_json::from_str(r#"{"arguments":[1,2,3]}"#).unwrap();
        assert_eq!(w.arguments.as_deref(), Some("[1,2,3]"));
    }

    /// Regression test: JSON null must collapse to None (not the string "null").
    /// Removing `.filter(|v| !v.is_null())` from the deserializer would fail this test.
    #[test]
    fn json_string_or_value_null_is_none() {
        let w: ArgWrapper = serde_json::from_str(r#"{"arguments":null}"#).unwrap();
        assert!(w.arguments.is_none());
    }

    /// A missing field is likewise None (relies on `#[serde(default)]`).
    #[test]
    fn json_string_or_value_missing_is_none() {
        let w: ArgWrapper = serde_json::from_str(r#"{}"#).unwrap();
        assert!(w.arguments.is_none());
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

    #[test]
    fn test_stringified_json_deserialize_empty_string() {
        let json_str = r#"{"data":""}"#;
        let dummy: Dummy = serde_json::from_str(json_str).unwrap();
        assert_eq!(dummy.data, serde_json::json!({}));
    }

    #[test]
    fn test_deserialize_maybe_stringified_value_from_string() {
        let json_str = r#"{"data":"{\"key\":\"value\"}"}"#;
        let dummy: DummyMaybeStringified = serde_json::from_str(json_str).unwrap();
        assert_eq!(dummy.data, serde_json::json!({"key": "value"}));
    }

    #[test]
    fn test_deserialize_maybe_stringified_value_from_object() {
        let json_str = r#"{"data":{"key":"value"}}"#;
        let dummy: DummyMaybeStringified = serde_json::from_str(json_str).unwrap();
        assert_eq!(dummy.data, serde_json::json!({"key": "value"}));
    }

    #[test]
    fn test_deserialize_maybe_stringified_value_from_empty_string() {
        let json_str = r#"{"data":""}"#;
        let dummy: DummyMaybeStringified = serde_json::from_str(json_str).unwrap();
        assert_eq!(dummy.data, serde_json::json!({}));
    }

    #[test]
    fn test_parse_tool_arguments_empty_string() {
        let parsed = parse_tool_arguments("").unwrap();
        assert_eq!(parsed, serde_json::json!({}));
    }

    #[test]
    fn test_parse_tool_arguments_whitespace_string() {
        let parsed = parse_tool_arguments("   ").unwrap();
        assert_eq!(parsed, serde_json::json!({}));
    }

    #[test]
    fn test_parse_tool_arguments_valid_json() {
        let parsed = parse_tool_arguments(r#"{"key":"value"}"#).unwrap();
        assert_eq!(parsed, serde_json::json!({"key": "value"}));
    }
}
