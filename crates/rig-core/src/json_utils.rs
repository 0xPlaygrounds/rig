use serde::Deserialize;
use serde::de::{self, Deserializer, SeqAccess, Visitor};
use std::convert::Infallible;
use std::fmt;
use std::marker::PhantomData;
use std::str::FromStr;

/// Invalid provider extension parameters at a request-owned serialization
/// boundary.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum RequestOverlayError {
    /// Extension parameters must be a JSON object (or `null`).
    #[error("{context} additional parameters must be a JSON object, got {kind}")]
    NonObject {
        /// Provider/request surface being built.
        context: &'static str,
        /// JSON value kind supplied by the caller.
        kind: &'static str,
    },
    /// An extension key tried to replace a request-owned field.
    #[error("{context} additional parameter `{key}` collides with a request-owned field")]
    Collision {
        /// Provider/request surface being built.
        context: &'static str,
        /// Rejected field name.
        key: String,
    },
    /// The canonical request builder did not produce an object.
    #[error("{context} canonical request must serialize as a JSON object")]
    CanonicalNotObject {
        /// Provider/request surface being built.
        context: &'static str,
    },
}

fn json_kind(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// Validate and borrow provider extension parameters as an object.
///
/// `reserved` lists request-owned fields that may be absent from the concrete
/// serialized value because of `skip_serializing_if`. `null` is treated as no
/// extension parameters. Any collision is rejected rather than silently
/// choosing caller or canonical precedence.
pub fn validated_additional_params<'a>(
    params: Option<&'a serde_json::Value>,
    reserved: &[&str],
    context: &'static str,
) -> Result<Option<&'a serde_json::Map<String, serde_json::Value>>, RequestOverlayError> {
    let Some(params) = params else {
        return Ok(None);
    };
    if params.is_null() {
        return Ok(None);
    }
    let serde_json::Value::Object(params) = params else {
        return Err(RequestOverlayError::NonObject {
            context,
            kind: json_kind(params),
        });
    };
    if let Some(key) = params
        .keys()
        .find(|key| reserved.iter().any(|reserved| key == reserved))
    {
        return Err(RequestOverlayError::Collision {
            context,
            key: key.clone(),
        });
    }
    Ok(Some(params))
}

/// Merge validated provider extension parameters into a canonical JSON object.
///
/// Present canonical keys are always protected in addition to the explicit
/// `reserved` list, so callers only need to list request-owned fields that can
/// be omitted from the concrete value.
pub fn merge_additional_params(
    canonical: serde_json::Value,
    params: Option<serde_json::Value>,
    reserved: &[&str],
    context: &'static str,
) -> Result<serde_json::Value, RequestOverlayError> {
    let serde_json::Value::Object(mut canonical) = canonical else {
        return Err(RequestOverlayError::CanonicalNotObject { context });
    };
    let Some(validated_params) = validated_additional_params(params.as_ref(), reserved, context)?
    else {
        return Ok(serde_json::Value::Object(canonical));
    };
    if let Some(key) = validated_params
        .keys()
        .find(|key| canonical.contains_key(*key))
    {
        return Err(RequestOverlayError::Collision {
            context,
            key: key.clone(),
        });
    }
    // Re-match the owner after borrowed validation so the object map can move.
    if let Some(serde_json::Value::Object(params)) = params {
        canonical.extend(params);
    }
    Ok(serde_json::Value::Object(canonical))
}

/// Deeply merge validated provider extensions into a canonical JSON object.
///
/// Object-valued fields are merged recursively, which lets a provider-native
/// nested field coexist with different request-owned leaves in the same
/// object. A collision is reported only at the first leaf both inputs own.
/// Explicit `reserved` fields still protect omitted top-level fields.
pub fn merge_additional_params_deep(
    canonical: serde_json::Value,
    params: Option<serde_json::Value>,
    reserved: &[&str],
    context: &'static str,
) -> Result<serde_json::Value, RequestOverlayError> {
    let serde_json::Value::Object(mut canonical) = canonical else {
        return Err(RequestOverlayError::CanonicalNotObject { context });
    };
    let Some(params) = validated_additional_params(params.as_ref(), reserved, context)? else {
        return Ok(serde_json::Value::Object(canonical));
    };

    merge_objects_deep(&mut canonical, params, context, None)?;
    Ok(serde_json::Value::Object(canonical))
}

fn merge_objects_deep(
    canonical: &mut serde_json::Map<String, serde_json::Value>,
    params: &serde_json::Map<String, serde_json::Value>,
    context: &'static str,
    parent_path: Option<&str>,
) -> Result<(), RequestOverlayError> {
    for (key, value) in params {
        let path = match parent_path {
            Some(parent) => format!("{parent}.{key}"),
            None => key.clone(),
        };

        match canonical.get_mut(key) {
            Some(serde_json::Value::Object(canonical_object)) => {
                let serde_json::Value::Object(params_object) = value else {
                    return Err(RequestOverlayError::Collision { context, key: path });
                };
                merge_objects_deep(canonical_object, params_object, context, Some(&path))?;
            }
            Some(_) => return Err(RequestOverlayError::Collision { context, key: path }),
            None => {
                canonical.insert(key.clone(), value.clone());
            }
        }
    }

    Ok(())
}

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

/// Extend one JSON object with another using last-writer-wins semantics.
///
/// This is a generic JSON utility, not a provider request boundary. Provider
/// extensions must use [`merge_additional_params`] so request-owned fields
/// cannot be replaced silently.
pub fn merge_inplace(a: &mut serde_json::Value, b: serde_json::Value) {
    if let (serde_json::Value::Object(a_map), serde_json::Value::Object(b_map)) = (a, b) {
        b_map.into_iter().for_each(|(key, value)| {
            a_map.insert(key, value);
        });
    }
}

/// Normalize a provider-wire field that may contain encoded JSON in a string.
///
/// This deliberately unwraps [`serde_json::Value::String`] and is only for
/// provider decoding, before a value enters Rig's canonical message model.
pub fn value_to_json_string(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Serialize a JSON value to its compact string form (with `String` scalars kept quoted).
///
/// Unlike [`value_to_json_string`], this never unwraps a `String` value — a JSON
/// string is serialized with its quotes. Used by the classic runtime when
/// canonicalizing tool-call arguments and hook-rewritten payloads.
pub fn serialize_json_value(value: &serde_json::Value) -> String {
    value.to_string()
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
    fn additional_params_reject_present_and_omitted_canonical_keys() {
        let canonical = serde_json::json!({"model": "m"});
        let present = merge_additional_params(
            canonical.clone(),
            Some(serde_json::json!({"model": "override"})),
            &["temperature"],
            "test request",
        )
        .expect_err("present canonical key must collide");
        assert!(matches!(
            present,
            RequestOverlayError::Collision { key, .. } if key == "model"
        ));

        let omitted = merge_additional_params(
            canonical,
            Some(serde_json::json!({"temperature": 1.0})),
            &["temperature"],
            "test request",
        )
        .expect_err("omitted canonical key must remain reserved");
        assert!(matches!(
            omitted,
            RequestOverlayError::Collision { key, .. } if key == "temperature"
        ));
    }

    #[test]
    fn additional_params_preserve_unrelated_provider_fields() {
        let merged = merge_additional_params(
            serde_json::json!({"model": "m"}),
            Some(serde_json::json!({"vendor_option": {"enabled": true}})),
            &["temperature"],
            "test request",
        )
        .expect("unrelated extension should merge");
        assert_eq!(merged["model"], "m");
        assert_eq!(merged["vendor_option"]["enabled"], true);
    }

    #[test]
    fn deep_additional_params_merge_distinct_nested_leaves() {
        let merged = merge_additional_params_deep(
            serde_json::json!({
                "generationConfig": {
                    "imageConfig": {"aspectRatio": "1:1"},
                    "responseModalities": ["IMAGE"]
                }
            }),
            Some(serde_json::json!({
                "generationConfig": {
                    "imageConfig": {"imageSize": "2K"},
                    "temperature": 0.4
                }
            })),
            &[],
            "test request",
        )
        .expect("distinct nested leaves should merge");

        assert_eq!(
            merged["generationConfig"]["imageConfig"]["aspectRatio"],
            "1:1"
        );
        assert_eq!(merged["generationConfig"]["imageConfig"]["imageSize"], "2K");
        assert_eq!(merged["generationConfig"]["temperature"], 0.4);
    }

    #[test]
    fn deep_additional_params_reject_exact_nested_leaf_collisions() {
        let error = merge_additional_params_deep(
            serde_json::json!({
                "generationConfig": {
                    "imageConfig": {"aspectRatio": "1:1"}
                }
            }),
            Some(serde_json::json!({
                "generationConfig": {
                    "imageConfig": {"aspectRatio": "16:9"}
                }
            })),
            &[],
            "test request",
        )
        .expect_err("the same nested leaf must collide");

        assert!(matches!(
            error,
            RequestOverlayError::Collision { key, .. }
                if key == "generationConfig.imageConfig.aspectRatio"
        ));
    }

    #[test]
    fn additional_params_require_an_object_or_null() {
        let error = validated_additional_params(
            Some(&serde_json::json!(["not", "an", "object"])),
            &[],
            "test request",
        )
        .expect_err("array must fail");
        assert!(matches!(
            error,
            RequestOverlayError::NonObject { kind: "array", .. }
        ));
        assert!(
            validated_additional_params(Some(&serde_json::Value::Null), &[], "test request")
                .expect("null means no extensions")
                .is_none()
        );

        let canonical = serde_json::json!({"model": "m"});
        let merge_error = merge_additional_params(
            canonical.clone(),
            Some(serde_json::json!(["not", "an", "object"])),
            &[],
            "test request",
        )
        .expect_err("merge must preserve non-object validation");
        assert_eq!(merge_error, error);
        assert_eq!(
            merge_additional_params(
                canonical.clone(),
                Some(serde_json::Value::Null),
                &[],
                "test request",
            )
            .expect("null means no extensions"),
            canonical,
        );
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
