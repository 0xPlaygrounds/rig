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
        /// (never for its warn log).
        value: serde_json::Value,
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
    let value = match serde_json::from_str::<serde_json::Value>(data) {
        Ok(value) => value,
        Err(error) => return WireEvent::Corrupt(error),
    };
    if let Err(error) = reject_duplicate_discriminators(data, &value, &[tag]) {
        return WireEvent::Corrupt(error);
    }

    match value.get(tag).and_then(serde_json::Value::as_str) {
        Some(event_type) if !is_known_event_type(event_type) => WireEvent::Unknown {
            event_type: event_type.to_owned(),
            value,
        },
        _ => decode_known(data),
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
    let value = match serde_json::from_str::<serde_json::Value>(data) {
        Ok(value) => value,
        Err(error) => return WireEvent::Corrupt(error),
    };
    if let Err(error) = reject_duplicate_discriminators(data, &value, &["object", "choices"]) {
        return WireEvent::Corrupt(error);
    }

    let is_chat_chunk = value
        .get("object")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|object| object == "chat.completion.chunk")
        || value.get("choices").is_some();
    if !is_chat_chunk {
        let event_type = value
            .get("object")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_owned();
        return WireEvent::Unknown { event_type, value };
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
    let value = match serde_json::from_str::<serde_json::Value>(data) {
        Ok(value) => value,
        Err(error) => return WireEvent::Corrupt(error),
    };

    let recognizable = marker_keys.iter().any(|key| value.get(key).is_some());
    if !recognizable {
        // No tag exists on this wire; name the frame by its top-level keys so
        // the driver's warn log stays diagnosable.
        let event_type = value
            .as_object()
            .map(|object| object.keys().cloned().collect::<Vec<_>>().join(","))
            .unwrap_or_default();
        return WireEvent::Unknown { event_type, value };
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
            value: serde_json::Value::String(detail),
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
fn reject_duplicate_discriminators(
    data: &str,
    value: &serde_json::Value,
    keys: &[&str],
) -> Result<(), serde_json::Error> {
    /// Visitor that walks the top-level map and counts occurrences of the
    /// discriminator keys, ignoring every value.
    struct DuplicateKeyScan<'a> {
        keys: &'a [&'a str],
    }

    impl<'de> serde::de::Visitor<'de> for DuplicateKeyScan<'_> {
        type Value = ();

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("a JSON object without duplicate discriminator keys")
        }

        fn visit_map<A>(self, mut map: A) -> Result<(), A::Error>
        where
            A: serde::de::MapAccess<'de>,
        {
            let mut seen = vec![false; self.keys.len()];
            while let Some(key) = map.next_key::<String>()? {
                map.next_value::<serde::de::IgnoredAny>()?;
                if let Some(index) = self.keys.iter().position(|candidate| *candidate == key) {
                    match seen.get_mut(index) {
                        Some(entry) if *entry => {
                            return Err(serde::de::Error::custom(format!(
                                "duplicate `{key}` discriminator key in stream frame"
                            )));
                        }
                        Some(entry) => *entry = true,
                        // Unreachable (`seen` mirrors `keys`); nothing to
                        // record.
                        None => {}
                    }
                }
            }
            Ok(())
        }
    }

    if !value.is_object() {
        // A non-object frame has no top-level keys to duplicate; the tag
        // lookup on `value` handles it downstream.
        return Ok(());
    }
    let mut deserializer = serde_json::Deserializer::from_str(data);
    serde::Deserializer::deserialize_map(&mut deserializer, DuplicateKeyScan { keys })
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
mod tests {
    use super::{WireEvent, classify_chat_completions_frame, classify_tagged_frame};

    #[derive(Debug, serde::Deserialize)]
    #[serde(tag = "type")]
    enum TestEvent {
        #[serde(rename = "text.delta")]
        TextDelta { delta: String },
    }

    fn known(event_type: &str) -> bool {
        event_type == "text.delta"
    }

    #[derive(Debug, serde::Deserialize)]
    struct TestChunk {
        #[allow(dead_code)]
        choices: Vec<serde_json::Value>,
    }

    #[test]
    fn tagged_known_frame_decodes() {
        let event = classify_tagged_frame::<TestEvent>(
            r#"{"type":"text.delta","delta":"hi"}"#,
            "type",
            known,
        );
        assert!(matches!(event, WireEvent::Known(TestEvent::TextDelta { delta }) if delta == "hi"));
    }

    #[test]
    fn tagged_unknown_type_is_unknown() {
        let event = classify_tagged_frame::<TestEvent>(r#"{"type":"future.event"}"#, "type", known);
        assert!(matches!(
            event,
            WireEvent::Unknown { event_type, .. } if event_type == "future.event"
        ));
    }

    #[test]
    fn tagged_invalid_json_is_corrupt() {
        let event = classify_tagged_frame::<TestEvent>("{not json", "type", known);
        assert!(matches!(event, WireEvent::Corrupt(_)));
    }

    #[test]
    fn tagged_known_type_with_defective_payload_is_corrupt() {
        let event = classify_tagged_frame::<TestEvent>(
            r#"{"type":"text.delta","delta":42}"#,
            "type",
            known,
        );
        assert!(matches!(event, WireEvent::Corrupt(_)));
    }

    #[test]
    fn tagged_typeless_frame_is_corrupt() {
        let event = classify_tagged_frame::<TestEvent>("{}", "type", known);
        assert!(matches!(event, WireEvent::Corrupt(_)));
    }

    /// #2258 review probe: `serde_json::Value` keeps the last duplicate key,
    /// so without the duplicate-discriminator rejection this frame would
    /// dispatch on `future.event` and demote a defective known frame to a
    /// skippable `Unknown`.
    #[test]
    fn tagged_duplicate_discriminator_is_corrupt() {
        let event = classify_tagged_frame::<TestEvent>(
            r#"{"type":"text.delta","type":"future.event","delta":"hi"}"#,
            "type",
            known,
        );
        assert!(matches!(event, WireEvent::Corrupt(_)));
    }

    /// #2258 review probe (second-pass extension): the chat classifier's
    /// `object` recognizability key is equally spoofable via duplication.
    #[test]
    fn chat_duplicate_object_discriminator_is_corrupt() {
        let event = classify_chat_completions_frame::<TestChunk>(
            r#"{"object":"chat.completion.chunk","object":"future.thing","data":1}"#,
        );
        assert!(matches!(event, WireEvent::Corrupt(_)));
    }

    #[test]
    fn chat_duplicate_choices_key_is_corrupt() {
        let event = classify_chat_completions_frame::<TestChunk>(r#"{"choices":[],"choices":42}"#);
        assert!(matches!(event, WireEvent::Corrupt(_)));
    }

    #[test]
    fn tagged_duplicate_non_discriminator_key_still_classifies() {
        // Only discriminator duplication is rejected by the scanner; other
        // duplicate keys stay with the typed decode's own policy (an ignored
        // key duplicated is harmless).
        let event = classify_tagged_frame::<TestEvent>(
            r#"{"type":"text.delta","ignored":1,"ignored":2,"delta":"hi"}"#,
            "type",
            known,
        );
        assert!(matches!(event, WireEvent::Known(TestEvent::TextDelta { delta }) if delta == "hi"));
    }

    #[test]
    fn chat_recognizable_chunk_decodes() {
        let event = classify_chat_completions_frame::<TestChunk>(r#"{"choices":[]}"#);
        assert!(matches!(event, WireEvent::Known(_)));
    }

    #[test]
    fn chat_unrecognizable_json_is_unknown() {
        let event = classify_chat_completions_frame::<TestChunk>(r#"{"object":"ping"}"#);
        assert!(matches!(
            event,
            WireEvent::Unknown { event_type, .. } if event_type == "ping"
        ));
    }

    #[test]
    fn chat_recognizable_chunk_with_defective_payload_is_corrupt() {
        let event = classify_chat_completions_frame::<TestChunk>(r#"{"choices":42}"#);
        assert!(matches!(event, WireEvent::Corrupt(_)));
    }

    #[test]
    fn chat_invalid_json_is_corrupt() {
        let event = classify_chat_completions_frame::<TestChunk>("{not json");
        assert!(matches!(event, WireEvent::Corrupt(_)));
    }

    #[test]
    fn tagged_dispatch_honors_a_non_type_tag_name() {
        #[derive(Debug, serde::Deserialize)]
        #[serde(tag = "event_type")]
        enum EventTypeTagged {
            #[serde(rename = "step.delta")]
            StepDelta { delta: String },
        }

        let event = classify_tagged_frame::<EventTypeTagged>(
            r#"{"event_type":"step.delta","delta":"hi"}"#,
            "event_type",
            |event_type| event_type == "step.delta",
        );
        assert!(matches!(
            event,
            WireEvent::Known(EventTypeTagged::StepDelta { delta }) if delta == "hi"
        ));

        let event = classify_tagged_frame::<EventTypeTagged>(
            r#"{"event_type":"future.event"}"#,
            "event_type",
            |event_type| event_type == "step.delta",
        );
        assert!(matches!(
            event,
            WireEvent::Unknown { event_type, .. } if event_type == "future.event"
        ));
    }

    #[test]
    fn marker_keyed_recognizable_chunk_decodes() {
        let event = super::classify_marker_keyed_frame::<TestChunk>(
            r#"{"choices":[]}"#,
            &["choices", "usage"],
        );
        assert!(matches!(event, WireEvent::Known(_)));
    }

    #[test]
    fn marker_keyed_unrecognizable_json_is_unknown() {
        let event = super::classify_marker_keyed_frame::<TestChunk>(
            r#"{"noise":true,"other":1}"#,
            &["choices", "usage"],
        );
        assert!(matches!(
            event,
            WireEvent::Unknown { event_type, .. } if event_type == "noise,other"
        ));
    }

    #[test]
    fn marker_keyed_recognizable_chunk_with_defective_payload_is_corrupt() {
        let event =
            super::classify_marker_keyed_frame::<TestChunk>(r#"{"choices":42}"#, &["choices"]);
        assert!(matches!(event, WireEvent::Corrupt(_)));
    }

    #[test]
    fn marker_keyed_invalid_json_is_corrupt() {
        let event = super::classify_marker_keyed_frame::<TestChunk>("{not json", &["choices"]);
        assert!(matches!(event, WireEvent::Corrupt(_)));
    }

    #[test]
    fn typed_event_triage_maps_onto_the_shared_policy() {
        // Modeled variants pass through as Known.
        let event = super::classify_typed_event(super::TypedEvent::Modeled(7u8));
        assert!(matches!(event, WireEvent::Known(7)));

        // The SDK's unknown-variant signal (aws-sdk `Unknown`, prost oneof
        // `None`) is Unknown, carrying the debug payload for the warn log.
        let event = super::classify_typed_event::<u8>(super::TypedEvent::Unrecognized {
            event_type: "unknown".to_string(),
            detail: "FutureEvent".to_string(),
        });
        assert!(matches!(
            event,
            WireEvent::Unknown { event_type, value }
                if event_type == "unknown" && value == serde_json::Value::String("FutureEvent".into())
        ));

        // An SDK decode error for a modeled event is Corrupt, never Unknown.
        let event =
            super::classify_typed_event::<u8>(super::TypedEvent::Malformed("bad frame".into()));
        assert!(
            matches!(event, WireEvent::Corrupt(error) if error.to_string().contains("bad frame"))
        );
    }

    #[test]
    fn untyped_line_is_known_or_corrupt() {
        assert!(matches!(
            super::classify_untyped_line::<TestChunk>(br#"{"choices":[]}"#),
            WireEvent::Known(_)
        ));
        assert!(matches!(
            super::classify_untyped_line::<TestChunk>(b"{not json"),
            WireEvent::Corrupt(_)
        ));
    }
}
