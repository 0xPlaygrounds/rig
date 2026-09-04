use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize, Serializer};
use std::collections::HashMap;
use std::convert::Infallible;
use std::fmt;
use std::marker::PhantomData;
use std::str::FromStr;

/// `skip_serializing_if` helper: serde requires a `fn(&bool) -> bool`, so the
/// trivially-copy lint does not apply here.
#[allow(clippy::trivially_copy_pass_by_ref)]
pub(crate) fn is_false(value: &bool) -> bool {
    !value
}

/// Serialize a `HashMap` in sorted key order.
///
/// `HashMap` seeds its iteration order per instance, so a map serialized into a
/// request body emits its keys in a *different order on every request*. Provider
/// prompt caches are prefix matches over the exact request bytes, so a map
/// anywhere in the cacheable prefix — a tool's JSON Schema `properties`, a
/// document's metadata — makes every request a guaranteed cache miss, silently
/// and permanently.
///
/// This is invisible to almost every test one would think to write: cassette
/// replay compares key-sorted canonical JSON, and a `serde_json::Value`
/// round-trip normalizes key order too, so recorded evidence looks identical
/// while the live wire never repeats itself. It is caught by
/// `provider_request_serialization_is_deterministic` in
/// `tests/cassette_cache_prefix.rs`, which serializes the same request several
/// times and compares the raw bytes.
///
/// Sorting rather than preserving insertion order matches the deliberate choice
/// already made when rendering [`crate::completion::Document`] metadata into a
/// prompt, and needs no ordered-map dependency in a public field type. JSON
/// object key order carries no meaning to any provider API, so sorting costs
/// nothing.
pub fn serialize_map_sorted<S, V>(
    map: &HashMap<String, V>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
    V: Serialize,
{
    let mut entries: Vec<_> = map.iter().collect();
    entries.sort_by(|(left, _), (right, _)| left.cmp(right));
    serializer.collect_map(entries)
}

/// [`serialize_map_sorted`] for an optional map.
///
/// Pairs with `#[serde(skip_serializing_if = "Option::is_none")]`: serde still
/// routes `Some` through this function, and the `None` arm only runs for a field
/// that is serialized unconditionally.
pub fn serialize_optional_map_sorted<S, V>(
    map: &Option<HashMap<String, V>>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
    V: Serialize,
{
    match map {
        Some(map) => serialize_map_sorted(map, serializer),
        None => serializer.serialize_none(),
    }
}

/// `value` serialized with every object's keys sorted, at every depth.
///
/// `serde_json` keeps insertion order in a build that enables its
/// `preserve_order` feature and sorts otherwise, so text rendered from a
/// `serde_json::Value` — a schema quoted into a preamble — would differ
/// between two crates over the same value, and with it the request bytes a
/// prompt cache is keyed on. Key order carries no meaning, so the sorted
/// rendering is the rendering.
pub fn to_canonical_string(value: &serde_json::Value) -> String {
    fn sorted(value: &serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::Object(map) => {
                let mut entries: Vec<_> = map.iter().collect();
                entries.sort_by(|(left, _), (right, _)| left.cmp(right));
                let mut out = serde_json::Map::new();
                for (key, value) in entries {
                    out.insert(key.clone(), sorted(value));
                }
                serde_json::Value::Object(out)
            }
            serde_json::Value::Array(items) => {
                serde_json::Value::Array(items.iter().map(sorted).collect())
            }
            other => other.clone(),
        }
    }
    sorted(value).to_string()
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

// Only the feature-gated `image` / `audio` provider request builders call this
// now; the default feature set has no caller, so allow it to be unused there
// rather than warning on an otherwise-live utility.
#[cfg_attr(not(any(feature = "image", feature = "audio")), allow(dead_code))]
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

        /// A bare object where a list is expected is one block, not a defect —
        /// several wires spell single-block content that way. This arm comes
        /// from the removed non-empty container's `string_or_one_or_many`,
        /// whose callers (Anthropic `Message.content`, OpenAI system and user
        /// content) now share this helper.
        ///
        /// The two were not interchangeable: this one also has
        /// `visit_none`/`visit_unit`, so the migrated fields now accept `null`
        /// where they used to raise a parse error. Those arms are load-bearing
        /// for the OpenAI assistant-content field that already used this
        /// helper — OpenAI sends `"content": null` for a tool-calls-only
        /// message — so the widening is the price of sharing one helper, and it
        /// is documented in MIGRATING rather than hidden.
        fn visit_map<M>(self, map: M) -> Result<Vec<T>, M::Error>
        where
            M: MapAccess<'de>,
        {
            let item = Deserialize::deserialize(de::value::MapAccessDeserializer::new(map))?;
            Ok(vec![item])
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

/// Deserializes `T`, mapping an explicit `null` to `T::default()`.
///
/// Driven through `deserialize_option` rather than `deserialize_any`; the two
/// agree only for self-describing formats, which is all rig decodes (JSON).
pub fn null_or_default<'de, T, D>(deserializer: D) -> Result<T, D::Error>
where
    T: Deserialize<'de> + Default,
    D: Deserializer<'de>,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

#[cfg(test)]
mod tests;
