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
    /// (owned by the stream driver, never per adapter): warn — with the
    /// payload, so unrecognized events are diagnosable — and skip, for
    /// forward compatibility.
    Unknown {
        /// The unmodeled discriminator value.
        event_type: String,
        /// The full frame payload, for the driver's warn log.
        value: serde_json::Value,
    },
    /// Not valid JSON, or a modeled discriminator whose payload failed the
    /// typed decode — a data-level defect in a known event, which must never
    /// be demoted to `Unknown`.
    Corrupt(serde_json::Error),
}

/// Classify one frame of a tag-discriminated JSON wire (OpenAI Responses SSE,
/// Cohere SSE, and Anthropic use `type`; Gemini Interactions uses
/// `event_type`).
///
/// Dispatch on the envelope's `tag` field: a value outside
/// `is_known_event_type` is `Unknown`; a modeled value — or a missing tag,
/// which no modeled event omits — must pass the full typed decode, and a
/// failure there is `Corrupt`, not `Unknown`.
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
/// valid JSON that is neither is `Unknown`.
pub fn classify_chat_completions_frame<T>(data: &str) -> WireEvent<T>
where
    T: serde::de::DeserializeOwned,
{
    let value = match serde_json::from_str::<serde_json::Value>(data) {
        Ok(value) => value,
        Err(error) => return WireEvent::Corrupt(error),
    };

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
