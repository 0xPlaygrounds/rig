use super::{WireEvent, classify_chat_completions_frame, classify_tagged_frame};

#[derive(Debug, serde::Deserialize)]
#[serde(tag = "type")]
enum TestEvent {
    #[serde(rename = "text.delta")]
    TextDelta {
        #[allow(dead_code)]
        delta: String,
    },
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
fn chat_duplicate_choices_key_is_corrupt() {
    let event = classify_chat_completions_frame::<TestChunk>(r#"{"choices":[],"choices":42}"#);
    assert!(matches!(event, WireEvent::Corrupt(_)));
}

#[test]
fn chat_unrecognizable_json_is_unknown() {
    let event = classify_chat_completions_frame::<TestChunk>(r#"{"object":"ping"}"#);
    assert!(matches!(
        event,
        WireEvent::Unknown { event_type, .. } if event_type == "ping"
    ));
}

/// Valid JSON that is not an object — a gateway keep-alive `null`, a
/// bare array or scalar — is Unknown (warn-and-skip) on every
/// classifier, never routed into a typed decode whose guaranteed
/// failure would fatal the stream as Corrupt (#2258 B5).
#[test]
fn non_object_json_is_unknown_never_corrupt() {
    for frame in ["null", "[]", "42", r#""ping""#] {
        let event = classify_chat_completions_frame::<TestChunk>(frame);
        assert!(
            matches!(event, WireEvent::Unknown { .. }),
            "chat classifier must skip {frame}, got {event:?}"
        );
        let event = classify_tagged_frame::<TestEvent>(frame, "type", known);
        assert!(
            matches!(event, WireEvent::Unknown { .. }),
            "tagged classifier must skip {frame}, got {event:?}"
        );
    }
}
