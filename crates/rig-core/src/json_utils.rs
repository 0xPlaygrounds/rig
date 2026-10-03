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

/// The longest object a cut-off JSON text `text` still states: open strings,
/// arrays and objects are closed and a dangling key or separator dropped, as
/// pi's streaming JSON parse does. `None` when no prefix states an object.
pub fn parse_partial_object(text: &str) -> Option<serde_json::Map<String, serde_json::Value>> {
    // Bounds the work on a large cut-off input: the last valid cut is
    // almost always near its end.
    const MAX_ATTEMPTS: usize = 256;
    let mut stack = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    // Prefix lengths a value just ended at, with the containers open there.
    let mut cuts: Vec<(usize, Vec<u8>, bool)> = Vec::new();
    let bytes = text.as_bytes();
    for (at, byte) in bytes.iter().enumerate() {
        if in_string {
            match (escaped, *byte) {
                (true, _) => escaped = false,
                (false, b'\\') => escaped = true,
                (false, b'"') => {
                    in_string = false;
                    cuts.push((at + 1, stack.clone(), false));
                }
                (false, _) => {}
            }
            continue;
        }
        match byte {
            b'"' => in_string = true,
            b'{' | b'[' => {
                stack.push(*byte);
                cuts.push((at + 1, stack.clone(), false));
            }
            b'}' | b']' => {
                stack.pop();
                cuts.push((at + 1, stack.clone(), false));
            }
            b if b.is_ascii_alphanumeric() || *b == b'.' || *b == b'-' || *b == b'+' => {
                let next = bytes.get(at + 1).copied();
                if !next.is_some_and(|n| {
                    n.is_ascii_alphanumeric() || n == b'.' || n == b'-' || n == b'+'
                }) {
                    cuts.push((at + 1, stack.clone(), false));
                }
            }
            _ => {}
        }
    }
    if in_string {
        let at = text.len() - usize::from(escaped);
        cuts.push((at, stack.clone(), true));
    }
    cuts.iter()
        .rev()
        .take(MAX_ATTEMPTS)
        .find_map(|(at, open, quote)| {
            let mut candidate = text.get(..*at)?.to_owned();
            if *quote {
                candidate.push('"');
            }
            for container in open.iter().rev() {
                candidate.push(if *container == b'{' { '}' } else { ']' });
            }
            match serde_json::from_str(&candidate) {
                Ok(serde_json::Value::Object(object)) => Some(object),
                _ => None,
            }
        })
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

/// Lenient reads of a provider reply: each accessor gives `None`, or an
/// empty slice, for a field that is missing or of another type, and never
/// fails. Decoders read replies through it, so a field no block or finish
/// needs can never fail a reply.
///
/// ```
/// use rig_core::json_utils::Lenient;
/// use serde_json::json;
///
/// let reply = json!({"id": "r1", "usage": {"input_tokens": "7"}, "output": null});
/// assert_eq!(reply.str("id"), Some("r1"));
/// assert_eq!(reply.at("/usage/input_tokens").and_then(Lenient::as_u64_lenient), Some(7));
/// assert!(reply.arr("output").is_empty());
/// ```
pub trait Lenient {
    /// The string at `key`.
    fn str(&self, key: &str) -> Option<&str>;
    /// The unsigned integer at `key`, from a number or a numeric string.
    fn u64(&self, key: &str) -> Option<u64>;
    /// The integer at `key`, from a number or a numeric string.
    fn i64(&self, key: &str) -> Option<i64>;
    /// The number at `key`, from a number or a numeric string.
    fn f64(&self, key: &str) -> Option<f64>;
    /// The boolean at `key`.
    fn bool(&self, key: &str) -> Option<bool>;
    /// The object at `key`.
    fn obj(&self, key: &str) -> Option<&serde_json::Map<String, serde_json::Value>>;
    /// The array at `key`, empty when absent or of another type.
    fn arr(&self, key: &str) -> &[serde_json::Value];
    /// The value at a JSON pointer, `None` for `null`.
    fn at(&self, pointer: &str) -> Option<&serde_json::Value>;
    /// This value as an unsigned integer, from a number or a numeric string.
    fn as_u64_lenient(&self) -> Option<u64>;
}

impl Lenient for serde_json::Value {
    fn str(&self, key: &str) -> Option<&str> {
        self.get(key)?.as_str()
    }

    fn u64(&self, key: &str) -> Option<u64> {
        self.get(key)?.as_u64_lenient()
    }

    fn i64(&self, key: &str) -> Option<i64> {
        match self.get(key)? {
            serde_json::Value::Number(number) => number.as_i64(),
            serde_json::Value::String(text) => text.trim().parse().ok(),
            _ => None,
        }
    }

    fn f64(&self, key: &str) -> Option<f64> {
        match self.get(key)? {
            serde_json::Value::Number(number) => number.as_f64(),
            serde_json::Value::String(text) => text.trim().parse().ok(),
            _ => None,
        }
    }

    fn bool(&self, key: &str) -> Option<bool> {
        self.get(key)?.as_bool()
    }

    fn obj(&self, key: &str) -> Option<&serde_json::Map<String, serde_json::Value>> {
        self.get(key)?.as_object()
    }

    fn arr(&self, key: &str) -> &[serde_json::Value] {
        self.get(key)
            .and_then(serde_json::Value::as_array)
            .map_or(&[], Vec::as_slice)
    }

    fn at(&self, pointer: &str) -> Option<&serde_json::Value> {
        self.pointer(pointer).filter(|value| !value.is_null())
    }

    fn as_u64_lenient(&self) -> Option<u64> {
        match self {
            serde_json::Value::Number(number) => number.as_u64().or_else(|| {
                number
                    .as_f64()
                    .filter(|float| *float >= 0.0 && float.fract() == 0.0)
                    .map(|float| float as u64)
            }),
            serde_json::Value::String(text) => text.trim().parse().ok(),
            _ => None,
        }
    }
}
