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

/// Serializes a `HashMap` in lexicographic key order, propagating serializer errors.
/// Stable ordering avoids randomized map iteration changing request bytes.
pub fn serialize_map_sorted<S, V>(
    map: &HashMap<String, V>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
    V: Serialize,
{
    let mut entries: Vec<_> = map.iter().collect();
    entries.sort_by_key(|(key, _)| *key);
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

/// Renders compact JSON with lexicographically sorted object keys at every depth,
/// independent of serde_json's `preserve_order` feature. Array order is retained.
pub fn to_canonical_string(value: &serde_json::Value) -> String {
    fn sorted(value: &serde_json::Value) -> serde_json::Value {
        match value {
            serde_json::Value::Object(map) => {
                let mut entries: Vec<_> = map.iter().collect();
                entries.sort_by_key(|(key, _)| *key);
                let mut out = serde_json::Map::new();
                for (key, value) in entries {
                    out.insert(key.clone(), sorted(value));
                }
                serde_json::Value::Object(out)
            }
            serde_json::Value::Array(items) => {
                serde_json::Value::Array(items.iter().map(sorted).collect())
            }
            serde_json::Value::Null
            | serde_json::Value::Bool(_)
            | serde_json::Value::Number(_)
            | serde_json::Value::String(_) => value.clone(),
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

/// Applies a request builder's `additional_params` call: `None` clears, and a
/// value merges over the parameters earlier calls set. Merging combines JSON
/// objects key by key; earlier parameters that are not an object are kept.
pub(crate) fn merge_params(
    existing: Option<serde_json::Value>,
    params: Option<serde_json::Value>,
) -> Option<serde_json::Value> {
    Some(match (existing, params?) {
        (Some(existing), params) => merge(existing, params),
        (None, params) => params,
    })
}

// Callers require the image or audio feature.
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

/// Serializes compact JSON, retaining quotes and escaping for string values.
pub fn serialize_json_value(value: &serde_json::Value) -> String {
    value.to_string()
}

/// Deserializes strings verbatim and other non-null values as compact JSON text.
/// Null becomes `None`; fields that may be absent need a serde default.
/// Object key order depends on serde_json's enabled features.
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

/// Serde adapters for JSON encoded inside strings. Empty or whitespace-only
/// strings deserialize to empty objects.
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

    /// Parses string contents as JSON and passes other values through unchanged.
    /// Empty or whitespace-only strings become empty objects.
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

/// Deserializes a string or object as one item, a sequence as multiple items,
/// and null as an empty list. Strings use [`FromStr`]; objects use [`Deserialize`].
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
