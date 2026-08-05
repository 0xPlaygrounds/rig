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

/// Classify one frame of a `type`-discriminated JSON wire (OpenAI Responses
/// SSE, Cohere SSE).
///
/// Dispatch on the envelope's `type`: a value outside `is_known_event_type`
/// is `Unknown`; a modeled value — or a missing `type`, which no modeled
/// event omits — must pass the full typed decode, and a failure there is
/// `Corrupt`, not `Unknown`.
pub fn classify_tagged_frame<T>(
    data: &str,
    is_known_event_type: impl Fn(&str) -> bool,
) -> WireEvent<T>
where
    T: serde::de::DeserializeOwned,
{
    let value = match serde_json::from_str::<serde_json::Value>(data) {
        Ok(value) => value,
        Err(error) => return WireEvent::Corrupt(error),
    };

    match value.get("type").and_then(serde_json::Value::as_str) {
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
        let event =
            classify_tagged_frame::<TestEvent>(r#"{"type":"text.delta","delta":"hi"}"#, known);
        assert!(matches!(event, WireEvent::Known(TestEvent::TextDelta { delta }) if delta == "hi"));
    }

    #[test]
    fn tagged_unknown_type_is_unknown() {
        let event = classify_tagged_frame::<TestEvent>(r#"{"type":"future.event"}"#, known);
        assert!(matches!(
            event,
            WireEvent::Unknown { event_type, .. } if event_type == "future.event"
        ));
    }

    #[test]
    fn tagged_invalid_json_is_corrupt() {
        let event = classify_tagged_frame::<TestEvent>("{not json", known);
        assert!(matches!(event, WireEvent::Corrupt(_)));
    }

    #[test]
    fn tagged_known_type_with_defective_payload_is_corrupt() {
        let event =
            classify_tagged_frame::<TestEvent>(r#"{"type":"text.delta","delta":42}"#, known);
        assert!(matches!(event, WireEvent::Corrupt(_)));
    }

    #[test]
    fn tagged_typeless_frame_is_corrupt() {
        let event = classify_tagged_frame::<TestEvent>("{}", known);
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
