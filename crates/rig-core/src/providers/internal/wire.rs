//! Decode-then-validate classification for JSON stream wire frames.
//!
//! Each streaming wire family classifies every frame through exactly one of
//! the functions below before acting on it, so the parse policy is stated once
//! per family instead of being re-derived from serde failure modes at each
//! call site. The classification is a hand-written tag dispatch, never an
//! untagged serde fallback: a trailing `#[serde(untagged)]` variant on an
//! internally-tagged enum swallows a known tag with an invalid payload
//! (`rig-2257-code-review-findings-34ee8ba5.md` P2), which would silently
//! demote a data-level defect to an ignorable unknown event.

/// One classified wire frame.
#[derive(Debug)]
pub enum WireEvent<T> {
    /// The frame carries a discriminator this client models and its payload
    /// decoded fully.
    Known(T),
    /// Valid JSON whose discriminator this client does not model. Policy
    /// (owned by the stream driver, never per adapter): warn — structural
    /// metadata only, so payloads never leak into logs — and skip, for
    /// forward compatibility.
    Unknown {
        /// The unmodeled discriminator value.
        event_type: String,
        /// The full frame payload, for the driver's raw passthrough channel
        /// (never for its warn log — and its Debug is redacted by type).
        value: crate::streaming::UnknownPayload,
    },
    /// Not valid JSON, or a modeled discriminator whose payload failed the
    /// typed decode — a data-level defect in a known event, which must never
    /// be demoted to `Unknown`.
    Corrupt(serde_json::Error),
}

impl<T> WireEvent<T> {
    /// Map the `Known` payload, preserving the classification.
    ///
    /// This is how an adapter layers a pure event-shape mapping on top of a
    /// classifier without restating the triage: `Unknown` and `Corrupt` pass
    /// through untouched, so policy stays with the driver.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> WireEvent<U> {
        match self {
            Self::Known(event) => WireEvent::Known(f(event)),
            Self::Unknown { event_type, value } => WireEvent::Unknown { event_type, value },
            Self::Corrupt(error) => WireEvent::Corrupt(error),
        }
    }
}

/// Classify one frame of a tag-discriminated JSON wire (OpenAI Responses SSE,
/// Cohere SSE, and Anthropic use `type`; Gemini Interactions uses
/// `event_type`).
///
/// Dispatch on the envelope's `tag` field: a value outside
/// `is_known_event_type` is `Unknown`; a modeled value — or a missing tag,
/// which no modeled event omits — must pass the full typed decode, and a
/// failure there is `Corrupt`, not `Unknown`. A frame carrying the tag key
/// more than once is `Corrupt` outright: `serde_json::Value` keeps only the
/// last occurrence, so without the rejection a defective known frame could
/// masquerade as a skippable unknown one.
pub fn classify_tagged_frame<T>(
    data: &str,
    tag: &str,
    is_known_event_type: impl Fn(&str) -> bool,
) -> WireEvent<T>
where
    T: serde::de::DeserializeOwned,
{
    let scanned = match scan_discriminators(data, &[tag], true) {
        Ok(scanned) => scanned,
        Err(error) => return WireEvent::Corrupt(error),
    };
    match scanned {
        DiscriminatorScan::Object(found) => {
            match found.first().and_then(|key| key.string_value.as_deref()) {
                Some(event_type) if !is_known_event_type(event_type) => {
                    unknown_with_value(data, event_type.to_owned())
                }
                _ => decode_known(data),
            }
        }
        // Valid JSON that is not an object (a gateway keep-alive `null`, a
        // bare array or scalar) cannot be a modeled event: it is Unknown
        // (warn-and-skip), never routed into the typed decode where its
        // guaranteed failure would read as Corrupt and error the stream.
        DiscriminatorScan::NotObject => unknown_with_value(data, String::new()),
    }
}

/// Build the `Unknown` cold path: the raw channel carries the full payload,
/// parsed lazily here — the hot Known path never pays for it.
fn unknown_with_value<T>(data: &str, event_type: String) -> WireEvent<T> {
    match serde_json::from_str::<serde_json::Value>(data) {
        Ok(value) => WireEvent::Unknown {
            event_type,
            value: value.into(),
        },
        // Unreachable in practice: the scan already tokenized this text.
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
    let scanned = match scan_discriminators(data, &["object", "choices"], true) {
        Ok(scanned) => scanned,
        Err(error) => return WireEvent::Corrupt(error),
    };
    let found = match scanned {
        DiscriminatorScan::Object(found) => found,
        // Same policy as `classify_tagged_frame`: non-object valid JSON is
        // unrecognizable, so it is Unknown (warn-and-skip) — a keep-alive
        // `null` must not become a fatal Corrupt via a doomed typed decode.
        DiscriminatorScan::NotObject => return unknown_with_value(data, String::new()),
    };
    let object_value = found.first().and_then(|key| key.string_value.as_deref());
    let has_choices = found.get(1).is_some_and(|key| key.present);
    let is_chat_chunk =
        object_value.is_some_and(|object| object == "chat.completion.chunk") || has_choices;
    if !is_chat_chunk {
        return unknown_with_value(data, object_value.unwrap_or_default().to_owned());
    }

    decode_known(data)
}

/// Classify one frame of an untagged JSON wire recognized by payload keys
/// (Gemini `streamGenerateContent`).
///
/// The wire has no discriminator, so recognizability substitutes — the same
/// policy as [`classify_chat_completions_frame`]: a frame carrying any of
/// `marker_keys` at top level is the wire's chunk shape and must pass the
/// full typed decode (failure is `Corrupt`); valid JSON carrying none of them
/// is `Unknown`.
pub fn classify_marker_keyed_frame<T>(data: &str, marker_keys: &[&str]) -> WireEvent<T>
where
    T: serde::de::DeserializeOwned,
{
    // Markers are presence checks, not discriminators — historically
    // duplicate-tolerant, so the scan does not reject duplicates here.
    let scanned = match scan_discriminators(data, marker_keys, false) {
        Ok(scanned) => scanned,
        Err(error) => return WireEvent::Corrupt(error),
    };
    let recognizable = match &scanned {
        DiscriminatorScan::Object(found) => found.iter().any(|key| key.present),
        DiscriminatorScan::NotObject => false,
    };
    if !recognizable {
        // Cold path: the Unknown channel needs the payload anyway, so parse
        // it here and name the frame by its top-level keys so the driver's
        // warn log stays diagnosable.
        let value = match serde_json::from_str::<serde_json::Value>(data) {
            Ok(value) => value,
            // Unreachable in practice: the scan already tokenized this text.
            Err(error) => return WireEvent::Corrupt(error),
        };
        let event_type = value
            .as_object()
            .map(|object| object.keys().cloned().collect::<Vec<_>>().join(","))
            .unwrap_or_default();
        return WireEvent::Unknown {
            event_type,
            value: value.into(),
        };
    }

    decode_known(data)
}

/// Classify one line of an undiscriminated NDJSON wire (Ollama).
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

/// Triage of one already-deserialized event from a typed-transport wire
/// (an aws-sdk event stream, a prost/tonic gRPC stream, an in-process
/// generation channel), for [`classify_typed_event`].
#[derive(Debug)]
pub enum TypedEvent<T> {
    /// A variant this client models.
    Modeled(T),
    /// The SDK's own unknown-variant signal — aws-sdk's non-exhaustive
    /// `Unknown` union variant, a prost oneof decoding to `None`.
    Unrecognized {
        /// Discriminator for the driver's warn log.
        event_type: String,
        /// Debug rendering of the frame, for the driver's warn log.
        detail: String,
    },
    /// The SDK reported a decode failure for a modeled event — a data-level
    /// defect in a known event.
    Malformed(String),
}

/// Classify one event of a typed-transport wire (bedrock's Converse event
/// stream, gemini-grpc, candle's in-process generation).
///
/// The transport SDK already deserialized the frame, so the byte-level decode
/// step collapses and only the triage remains: modeled variants are `Known`;
/// the SDK's non-exhaustive/unrecognized variants are `Unknown`; an SDK
/// decode error for a modeled event is `Corrupt` — the same known-tag
/// strictness as the JSON classifiers, so a typed transport earns no policy
/// exemption.
pub fn classify_typed_event<T>(event: TypedEvent<T>) -> WireEvent<T> {
    match event {
        TypedEvent::Modeled(event) => WireEvent::Known(event),
        TypedEvent::Unrecognized { event_type, detail } => WireEvent::Unknown {
            event_type,
            value: serde_json::Value::String(detail).into(),
        },
        TypedEvent::Malformed(message) => {
            WireEvent::Corrupt(<serde_json::Error as serde::de::Error>::custom(message))
        }
    }
}

/// Classify a frame with a one-shot salvage step for `Corrupt` results.
///
/// Some replayed (buffered) wire bodies verifiably omit envelope bookkeeping
/// fields the typed decode requires (ChatGPT's unary Responses bodies). This
/// wrapper keeps that salvage inside the classify layer: when `classify`
/// reports `Corrupt`, `repair` may produce an amended frame that is classified
/// once more through the SAME interpreter. A frame `repair` cannot amend
/// (`None`) maps to `Corrupt(on_unrepairable(original_error))`; a repaired
/// frame that still fails maps to `Corrupt(on_still_corrupt())` — it is
/// defective in its data, not its envelope. `Known` and `Unknown` results
/// pass through untouched, so no policy is decided here.
pub fn classify_with_repair<T>(
    data: &str,
    classify: impl Fn(&str) -> WireEvent<T>,
    repair: impl FnOnce(&str) -> Option<String>,
    on_unrepairable: impl FnOnce(&serde_json::Error) -> serde_json::Error,
    on_still_corrupt: impl FnOnce() -> serde_json::Error,
) -> WireEvent<T> {
    match classify(data) {
        WireEvent::Corrupt(corrupt) => match repair(data) {
            None => WireEvent::Corrupt(on_unrepairable(&corrupt)),
            Some(repaired) => match classify(&repaired) {
                WireEvent::Known(event) => WireEvent::Known(event),
                // `Unknown` is unreachable in practice (an unknown tag never
                // classified `Corrupt` in the first pass); treat it as the
                // defect it would be.
                WireEvent::Unknown { .. } | WireEvent::Corrupt(_) => {
                    WireEvent::Corrupt(on_still_corrupt())
                }
            },
        },
        event => event,
    }
}

/// Reject a top-level object that carries any discriminator key more than
/// once.
///
/// `serde_json::Value` retains only the last occurrence of a duplicate key,
/// so a frame like `{"type":"text.delta","type":"future.event",...}` would
/// otherwise dispatch on the *last* value and demote a defective known frame
/// to a skippable `Unknown` — violating the classifier invariant that a
/// data-level defect in a known event is always `Corrupt`. This re-scans the
/// raw text with a streaming visitor that sees every key occurrence.
/// `value` (the already-parsed frame) gates the scan to top-level objects.
/// What one streaming pass learned about a frame's discriminator keys.
///
/// This is the fused form of "parse to `Value` for the discriminator lookup"
/// and "scan for duplicate discriminator keys": one tokenization pass over
/// the raw text yields both, so the hot path (Known frames) runs exactly two
/// passes — this scan plus the typed decode, the irreducible minimum. The
/// `Unknown` cold path lazily parses the `Value` it must carry anyway.
enum DiscriminatorScan {
    /// Top-level object: per requested key, whether it was present and its
    /// string value when it had one (first occurrence; a duplicate is an
    /// error before this is returned).
    Object(Vec<KeyScan>),
    /// Not a JSON object — no top-level keys exist; classification falls
    /// through to the typed decode.
    NotObject,
}

#[derive(Default, Clone)]
struct KeyScan {
    present: bool,
    string_value: Option<String>,
}

fn scan_discriminators(
    data: &str,
    keys: &[&str],
    reject_duplicates: bool,
) -> Result<DiscriminatorScan, serde_json::Error> {
    struct Scan<'a> {
        keys: &'a [&'a str],
        reject_duplicates: bool,
    }

    impl<'de> serde::de::Visitor<'de> for Scan<'_> {
        type Value = DiscriminatorScan;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("a JSON value")
        }

        fn visit_map<A>(self, mut map: A) -> Result<DiscriminatorScan, A::Error>
        where
            A: serde::de::MapAccess<'de>,
        {
            let mut found = vec![KeyScan::default(); self.keys.len()];
            while let Some(key) = map.next_key::<String>()? {
                match self.keys.iter().position(|candidate| *candidate == key) {
                    Some(index) => {
                        let entry = found.get_mut(index).ok_or_else(|| {
                            serde::de::Error::custom("discriminator index out of range")
                        })?;
                        if entry.present {
                            if self.reject_duplicates {
                                return Err(serde::de::Error::custom(format!(
                                    "duplicate `{key}` discriminator key in stream frame"
                                )));
                            }
                            // Presence-only keys tolerate duplicates; the
                            // first occurrence's value stands.
                            map.next_value::<serde::de::IgnoredAny>()?;
                            continue;
                        }
                        entry.present = true;
                        // Only string discriminators carry a value; anything
                        // else (e.g. a `choices` array) records presence.
                        entry.string_value = match map.next_value::<StringOrIgnored>()? {
                            StringOrIgnored::String(value) => Some(value),
                            StringOrIgnored::Ignored => None,
                        };
                    }
                    None => {
                        map.next_value::<serde::de::IgnoredAny>()?;
                    }
                }
            }
            Ok(DiscriminatorScan::Object(found))
        }

        // Every non-map shape falls through to the typed decode downstream.
        fn visit_bool<E>(self, _: bool) -> Result<DiscriminatorScan, E> {
            Ok(DiscriminatorScan::NotObject)
        }
        fn visit_i64<E>(self, _: i64) -> Result<DiscriminatorScan, E> {
            Ok(DiscriminatorScan::NotObject)
        }
        fn visit_u64<E>(self, _: u64) -> Result<DiscriminatorScan, E> {
            Ok(DiscriminatorScan::NotObject)
        }
        fn visit_f64<E>(self, _: f64) -> Result<DiscriminatorScan, E> {
            Ok(DiscriminatorScan::NotObject)
        }
        fn visit_str<E>(self, _: &str) -> Result<DiscriminatorScan, E> {
            Ok(DiscriminatorScan::NotObject)
        }
        fn visit_unit<E>(self) -> Result<DiscriminatorScan, E> {
            Ok(DiscriminatorScan::NotObject)
        }
        fn visit_seq<A>(self, mut seq: A) -> Result<DiscriminatorScan, A::Error>
        where
            A: serde::de::SeqAccess<'de>,
        {
            while seq.next_element::<serde::de::IgnoredAny>()?.is_some() {}
            Ok(DiscriminatorScan::NotObject)
        }
    }

    /// Captures a string value, consumes-and-ignores every other shape.
    enum StringOrIgnored {
        String(String),
        Ignored,
    }

    impl<'de> serde::Deserialize<'de> for StringOrIgnored {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            struct V;
            impl<'de> serde::de::Visitor<'de> for V {
                type Value = StringOrIgnored;
                fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    formatter.write_str("any JSON value")
                }
                fn visit_str<E>(self, value: &str) -> Result<StringOrIgnored, E> {
                    Ok(StringOrIgnored::String(value.to_owned()))
                }
                fn visit_string<E>(self, value: String) -> Result<StringOrIgnored, E> {
                    Ok(StringOrIgnored::String(value))
                }
                fn visit_bool<E>(self, _: bool) -> Result<StringOrIgnored, E> {
                    Ok(StringOrIgnored::Ignored)
                }
                fn visit_i64<E>(self, _: i64) -> Result<StringOrIgnored, E> {
                    Ok(StringOrIgnored::Ignored)
                }
                fn visit_u64<E>(self, _: u64) -> Result<StringOrIgnored, E> {
                    Ok(StringOrIgnored::Ignored)
                }
                fn visit_f64<E>(self, _: f64) -> Result<StringOrIgnored, E> {
                    Ok(StringOrIgnored::Ignored)
                }
                fn visit_unit<E>(self) -> Result<StringOrIgnored, E> {
                    Ok(StringOrIgnored::Ignored)
                }
                fn visit_map<A>(self, mut map: A) -> Result<StringOrIgnored, A::Error>
                where
                    A: serde::de::MapAccess<'de>,
                {
                    while map
                        .next_entry::<serde::de::IgnoredAny, serde::de::IgnoredAny>()?
                        .is_some()
                    {}
                    Ok(StringOrIgnored::Ignored)
                }
                fn visit_seq<A>(self, mut seq: A) -> Result<StringOrIgnored, A::Error>
                where
                    A: serde::de::SeqAccess<'de>,
                {
                    while seq.next_element::<serde::de::IgnoredAny>()?.is_some() {}
                    Ok(StringOrIgnored::Ignored)
                }
            }
            deserializer.deserialize_any(V)
        }
    }

    let mut deserializer = serde_json::Deserializer::from_str(data);
    let scanned = serde::Deserializer::deserialize_any(
        &mut deserializer,
        Scan {
            keys,
            reject_duplicates,
        },
    )?;
    deserializer.end()?;
    Ok(scanned)
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
