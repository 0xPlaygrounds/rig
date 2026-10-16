//! Classifies stream frames as known, unknown, or corrupt before interpretation.
//! Known discriminators require typed decoding; invalid known payloads must not
//! become ignorable unknown events.
//!
//! ```
//! use rig_core::providers::internal::wire::classify_untyped_line;
//! use rig_core::wire::WireEvent;
//! let event = classify_untyped_line::<serde_json::Value>(b"{}");
//! assert!(matches!(event, WireEvent::Known(_)));
//! ```

use serde::de::{DeserializeSeed, IgnoredAny, MapAccess, Visitor};
use serde_json::Value;
use serde_json::value::RawValue;

use crate::wire::WireEvent;

/// Classify JSON by its top-level `tag` string.
/// Unknown strings and non-object JSON produce `Unknown`. Known, missing, or
/// nonstring tags require typed decoding. Invalid JSON, duplicate tags, and
/// failed typed decoding produce `Corrupt`.
pub fn classify_tagged_frame<T>(
    data: &str,
    tag: &str,
    is_known_event_type: impl Fn(&str) -> bool,
) -> WireEvent<T>
where
    T: serde::de::DeserializeOwned,
{
    match scan(data, &[tag], true) {
        Err(error) => WireEvent::Corrupt(error),
        // Non-object keep-alives must not become fatal typed-decode failures.
        Ok(None) => unknown(data, String::new()),
        Ok(Some(found)) => match text(found.first().copied().flatten()) {
            Some(event_type) if !is_known_event_type(&event_type) => unknown(data, event_type),
            _ => decode_known(data),
        },
    }
}

/// A frame classified as unknown, its payload kept for raw passthrough.
fn unknown<T>(data: &str, event_type: String) -> WireEvent<T> {
    match serde_json::from_str::<Value>(data) {
        Ok(value) => WireEvent::Unknown {
            event_type,
            value: value.into(),
        },
        Err(error) => WireEvent::Corrupt(error),
    }
}

/// Classify one chat-completions SSE frame (OpenAI-compatible chat wire).
///
/// The chat wire has no `type` discriminator, so recognizability substitutes:
/// a frame saying `"object": "chat.completion.chunk"` or carrying `choices`
/// is a chunk and must pass the full typed decode (failure is `Corrupt`);
/// valid JSON that is neither is `Unknown`. A frame carrying `object` or
/// `choices` more than once is `Corrupt` outright (same duplicate-key policy
/// as [`classify_tagged_frame`]).
pub fn classify_chat_completions_frame<T>(data: &str) -> WireEvent<T>
where
    T: serde::de::DeserializeOwned,
{
    let found = match scan(data, &["object", "choices"], true) {
        Ok(Some(found)) => found,
        // Non-object JSON is unrecognized rather than corrupt.
        Ok(None) => return unknown(data, String::new()),
        Err(error) => return WireEvent::Corrupt(error),
    };
    let object = text(found.first().copied().flatten());
    let has_choices = found.get(1).is_some_and(Option::is_some);
    if object.as_deref() == Some("chat.completion.chunk") || has_choices {
        decode_known(data)
    } else {
        unknown(data, object.unwrap_or_default())
    }
}

/// Classify JSON by the presence of any top-level `marker_keys`.
/// Recognized frames require typed decoding; failures produce `Corrupt`.
/// Valid JSON without markers produces `Unknown`, named by its top-level
/// keys so the driver's warn log stays diagnosable. Duplicate markers are
/// allowed.
pub fn classify_marker_keyed_frame<T>(data: &str, marker_keys: &[&str]) -> WireEvent<T>
where
    T: serde::de::DeserializeOwned,
{
    match scan(data, marker_keys, false) {
        Err(error) => WireEvent::Corrupt(error),
        Ok(Some(found)) if found.iter().any(Option::is_some) => decode_known(data),
        Ok(_) => match serde_json::from_str::<Value>(data) {
            Ok(value) => WireEvent::Unknown {
                event_type: value
                    .as_object()
                    .map(|object| object.keys().cloned().collect::<Vec<_>>().join(","))
                    .unwrap_or_default(),
                value: value.into(),
            },
            Err(error) => WireEvent::Corrupt(error),
        },
    }
}

/// Classify one line of an undiscriminated JSON wire.
///
/// The wire has no discriminator at all: a line either decodes as the
/// response shape (`Known`) or is `Corrupt`. This family never produces
/// `Unknown`.
pub fn classify_untyped_line<T>(line: &[u8]) -> WireEvent<T>
where
    T: serde::de::DeserializeOwned,
{
    match serde_json::from_slice::<T>(line) {
        Ok(event) => WireEvent::Known(event),
        Err(error) => WireEvent::Corrupt(error),
    }
}

/// Try `then` only when `first` returns `Corrupt`.
/// Return `then`'s known event or corrupt error. If `then` returns `Unknown`,
/// preserve `first`'s error. Initial `Known` and `Unknown` results pass through.
pub fn classify_or<T>(
    data: &str,
    first: impl Fn(&str) -> WireEvent<T>,
    then: impl Fn(&str) -> WireEvent<T>,
) -> WireEvent<T> {
    match first(data) {
        WireEvent::Corrupt(first_error) => match then(data) {
            WireEvent::Known(event) => WireEvent::Known(event),
            WireEvent::Corrupt(error) => WireEvent::Corrupt(error),
            WireEvent::Unknown { .. } => WireEvent::Corrupt(first_error),
        },
        event => event,
    }
}

/// The `{ "message": ... }` error envelope some APIs answer with HTTP 200.
#[derive(serde::Deserialize)]
struct MessageEnvelope {
    #[allow(dead_code)]
    message: String,
}

/// Classify a whole reply recognized by its `reply_marker` key, or the
/// `{ "message": ... }` error envelope its API can send with HTTP 200 instead.
/// The envelope yields its body verbatim as `Err`, so the decoder reports it
/// and the driver adds the HTTP status. When neither shape decodes, the
/// reply's diagnostic stands.
pub fn classify_reply_or_message_envelope<T>(
    data: &str,
    reply_marker: &str,
) -> WireEvent<Result<T, String>>
where
    T: serde::de::DeserializeOwned,
{
    classify_or(
        data,
        |data| classify_marker_keyed_frame::<T>(data, &[reply_marker, "message"]).map(Ok),
        |data| {
            classify_marker_keyed_frame::<MessageEnvelope>(data, &["message"])
                .map(|_| Err(data.to_owned()))
        },
    )
}

/// [`classify_tagged_frame`] for a wire whose events carry `tag`, mapped by
/// `tagged`, falling back to `untagged` for an object without the tag (a
/// unary body), which is never parsed as an event first. A tagged frame that
/// fails its typed decode stays corrupt rather than passing for the
/// fallback's shape; an untagged object the fallback does not recognize
/// takes the typed decode, so its error is the event's.
pub fn classify_or_untagged<T, E>(
    data: &str,
    tag: &str,
    is_known_event_type: impl Fn(&str) -> bool,
    tagged: impl Fn(T) -> E,
    untagged: impl Fn(&str) -> WireEvent<E>,
) -> WireEvent<E>
where
    T: serde::de::DeserializeOwned,
{
    match scan(data, &[tag], true) {
        Err(error) => WireEvent::Corrupt(error),
        Ok(None) => unknown(data, String::new()),
        Ok(Some(found)) => match found.first().copied().flatten() {
            None => match untagged(data) {
                WireEvent::Unknown { .. } => decode_known(data).map(tagged),
                event => event,
            },
            Some(value) => match text(Some(value)) {
                Some(event_type) if !is_known_event_type(&event_type) => unknown(data, event_type),
                _ => decode_known(data).map(tagged),
            },
        },
    }
}

/// What `data`, when it is a JSON object, holds of each of `keys` (the
/// first occurrence), read in one pass that keeps no other value; `None`
/// for any other text, invalid JSON included.
pub(crate) fn top_level<'a>(data: &'a str, keys: &[&str]) -> Option<Found<'a>> {
    scan(data, keys, false).ok().flatten()
}

/// The first value of each scanned key, in key order, as the frame's text;
/// `None` when the object lacks the key.
type Found<'a> = Vec<Option<&'a RawValue>>;

/// `value` when it is a JSON string.
fn text(value: Option<&RawValue>) -> Option<String> {
    serde_json::from_str(value?.get()).ok()
}

/// When `data` is a JSON object, what it holds of each of `keys`, in one
/// pass that reads every other value without keeping it; `None` for any
/// other JSON. A second occurrence of one of `keys` is an error when
/// `unique`.
fn scan<'a>(
    data: &'a str,
    keys: &[&str],
    unique: bool,
) -> Result<Option<Found<'a>>, serde_json::Error> {
    let mut json = serde_json::Deserializer::from_str(data);
    let scanned = if data
        .trim_start_matches([' ', '\t', '\n', '\r'])
        .starts_with('{')
    {
        Some(serde::Deserializer::deserialize_map(&mut json, Scan(keys))?)
    } else {
        <IgnoredAny as serde::Deserialize>::deserialize(&mut json)?;
        None
    };
    json.end()?;
    match scanned {
        Some((_, Some(key))) if unique => Err(serde::de::Error::custom(format!(
            "duplicate `{key}` discriminator key in stream frame"
        ))),
        scanned => Ok(scanned.map(|(found, _)| found)),
    }
}

/// Reads an object's `keys`, and the first of them that occurs twice.
struct Scan<'k>(&'k [&'k str]);

impl<'de, 'k> Visitor<'de> for Scan<'k> {
    type Value = (Found<'de>, Option<&'k str>);

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a JSON object")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut found = vec![None; self.0.len()];
        let mut duplicate = None;
        while let Some(index) = map.next_key_seed(Key(self.0))? {
            match index.and_then(|index| found.get_mut(index).zip(self.0.get(index))) {
                Some((slot @ None, _)) => *slot = Some(map.next_value()?),
                Some((Some(_), key)) => {
                    duplicate = duplicate.or(Some(*key));
                    map.next_value::<IgnoredAny>()?;
                }
                None => {
                    map.next_value::<IgnoredAny>()?;
                }
            }
        }
        Ok((found, duplicate))
    }
}

/// An object key's position in a key list, read without allocating.
struct Key<'k>(&'k [&'k str]);

impl<'de> DeserializeSeed<'de> for Key<'_> {
    type Value = Option<usize>;

    fn deserialize<D: serde::Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_str(self)
    }
}

impl<'de> Visitor<'de> for Key<'_> {
    type Value = Option<usize>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("an object key")
    }

    fn visit_str<E>(self, key: &str) -> Result<Self::Value, E> {
        Ok(self.0.iter().position(|candidate| *candidate == key))
    }
}

fn decode_known<T>(data: &str) -> WireEvent<T>
where
    T: serde::de::DeserializeOwned,
{
    match serde_json::from_str::<T>(data) {
        Ok(event) => WireEvent::Known(event),
        Err(error) => WireEvent::Corrupt(error),
    }
}

#[cfg(test)]
mod tests;
