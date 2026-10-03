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

use serde_json::Value;

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
        Ok((value, None)) => unknown(value, String::new()),
        Ok((value, Some(found))) => match found.first().and_then(|tag| tag.as_ref()?.as_str()) {
            Some(event_type) if !is_known_event_type(event_type) => {
                let event_type = event_type.to_owned();
                unknown(value, event_type)
            }
            _ => decode_known(data),
        },
    }
}

/// A frame classified as unknown, its payload kept for raw passthrough.
fn unknown<T>(value: Value, event_type: String) -> WireEvent<T> {
    WireEvent::Unknown {
        event_type,
        value: value.into(),
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
    let (value, found) = match scan(data, &["object", "choices"], true) {
        Ok((value, Some(found))) => (value, found),
        // Non-object JSON is unrecognized rather than corrupt.
        Ok((value, None)) => return unknown(value, String::new()),
        Err(error) => return WireEvent::Corrupt(error),
    };
    let object = found
        .first()
        .and_then(|object| object.as_ref()?.as_str())
        .map(str::to_owned);
    let has_choices = found.get(1).is_some_and(Option::is_some);
    if object.as_deref() == Some("chat.completion.chunk") || has_choices {
        decode_known(data)
    } else {
        unknown(value, object.unwrap_or_default())
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
        Ok((_, Some(found))) if found.iter().any(Option::is_some) => decode_known(data),
        Ok((value, _)) => {
            let event_type = value
                .as_object()
                .map(|object| object.keys().cloned().collect::<Vec<_>>().join(","))
                .unwrap_or_default();
            unknown(value, event_type)
        }
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

/// [`classify_or`] for a wire whose events carry `tag`: only a frame without
/// the tag falls back to `then`. A tagged frame that fails its typed decode
/// stays corrupt, rather than passing for the fallback's shape.
pub fn classify_or_untagged<T>(
    data: &str,
    tag: &str,
    first: impl Fn(&str) -> WireEvent<T>,
    then: impl Fn(&str) -> WireEvent<T>,
) -> WireEvent<T> {
    classify_or(data, first, |data| match scan(data, &[tag], false) {
        // `classify_or` keeps the typed decode's error for this.
        Ok((value, Some(found))) if found.iter().any(Option::is_some) => {
            unknown(value, tag.to_owned())
        }
        _ => then(data),
    })
}

/// A JSON object's entries in their order, duplicates kept.
struct Entries(Vec<(String, Value)>);

impl<'de> serde::Deserialize<'de> for Entries {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = Entries;
            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a JSON object")
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<Entries, A::Error> {
                let mut entries = Vec::new();
                while let Some(entry) = map.next_entry()? {
                    entries.push(entry);
                }
                Ok(Entries(entries))
            }
        }
        deserializer.deserialize_map(Visitor)
    }
}

/// The frame `data` as JSON and, when it is an object, the first value of
/// each of `keys`. A second occurrence of one is an error when `unique`.
fn scan(
    data: &str,
    keys: &[&str],
    unique: bool,
) -> Result<(Value, Option<Vec<Option<Value>>>), serde_json::Error> {
    let value: Value = serde_json::from_str(data)?;
    if !value.is_object() {
        return Ok((value, None));
    }
    let Entries(entries) = serde_json::from_str(data)?;
    let mut found = vec![None; keys.len()];
    for (key, field) in entries {
        match keys
            .iter()
            .position(|candidate| *candidate == key)
            .and_then(|index| found.get_mut(index))
        {
            Some(slot @ None) => *slot = Some(field),
            Some(Some(_)) if unique => {
                return Err(serde::de::Error::custom(format!(
                    "duplicate `{key}` discriminator key in stream frame"
                )));
            }
            Some(Some(_)) | None => {}
        }
    }
    Ok((value, Some(found)))
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
