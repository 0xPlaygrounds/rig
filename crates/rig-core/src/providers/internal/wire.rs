//! Classifies stream frames as known, unknown, or corrupt before interpretation.
//! Known discriminators require typed decoding; invalid known payloads must not
//! become ignorable unknown events.
//!
//! ```
//! use rig_core::providers::internal::wire::{classify_untyped_line, WireEvent};
//! let event = classify_untyped_line::<serde_json::Value>(b"{}");
//! assert!(matches!(event, WireEvent::Known(_)));
//! ```

/// One classified wire frame.
#[derive(Debug)]
pub enum WireEvent<T> {
    /// The frame carries a discriminator this client models and its payload
    /// decoded fully.
    Known(T),
    /// Valid JSON not recognized by this classifier.
    /// Drivers log structural metadata only and skip interpretation.
    Unknown {
        /// The unmodeled discriminator value.
        event_type: String,
        /// Full payload for raw passthrough, never warning logs. Debug is redacted.
        value: crate::streaming::UnknownPayload,
    },
    /// Invalid JSON or a recognized frame that failed typed decoding.
    /// Must not be demoted to `Unknown`.
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
        // Non-object keep-alives must not become fatal typed-decode failures.
        DiscriminatorScan::NotObject => unknown_with_value(data, String::new()),
    }
}

/// Parse the payload for raw passthrough after classifying a frame as unknown.
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
        // Non-object JSON is unrecognized rather than corrupt.
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

/// Classify JSON by the presence of any top-level `marker_keys`.
/// Recognized frames require typed decoding; failures produce `Corrupt`.
/// Valid JSON without markers produces `Unknown`. Duplicate markers are allowed.
pub fn classify_marker_keyed_frame<T>(data: &str, marker_keys: &[&str]) -> WireEvent<T>
where
    T: serde::de::DeserializeOwned,
{
    // Presence markers permit duplicates because their values do not select a type.
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
    /// An unrecognized variant reported by the transport SDK.
    Unrecognized {
        /// Discriminator for the driver's warn log.
        event_type: String,
        /// Frame detail retained for raw passthrough, not warning logs.
        detail: String,
    },
    /// SDK decode failure for a modeled event.
    Malformed(String),
}

/// Map modeled SDK events to `Known`, unrecognized events to `Unknown`, and
/// malformed events to `Corrupt`. Unknown detail is retained for raw passthrough.
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

/// Retry classification once after repairing a `Corrupt` frame.
/// Initial `Known` and `Unknown` results pass through. No repair returns
/// `on_unrepairable`'s error; repaired frames must classify as `Known` or return
/// `on_still_corrupt`'s error.
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

/// Discriminator presence and first string values collected in one JSON scan.
enum DiscriminatorScan {
    /// Presence and first string value for each requested top-level key.
    Object(Vec<KeyScan>),
    /// Valid non-object JSON with no top-level keys.
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
