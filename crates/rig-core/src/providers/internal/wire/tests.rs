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
        classify_tagged_frame::<TestEvent>(r#"{"type":"text.delta","delta":"hi"}"#, "type", known);
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
    let event =
        classify_tagged_frame::<TestEvent>(r#"{"type":"text.delta","delta":42}"#, "type", known);
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
    let event =
        super::classify_marker_keyed_frame::<TestChunk>(r#"{"choices":[]}"#, &["choices", "usage"]);
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
    let event = super::classify_marker_keyed_frame::<TestChunk>(r#"{"choices":42}"#, &["choices"]);
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
            if event_type == "unknown" && value.value() == &serde_json::Value::String("FutureEvent".into())
    ));

    // An SDK decode error for a modeled event is Corrupt, never Unknown.
    let event = super::classify_typed_event::<u8>(super::TypedEvent::Malformed("bad frame".into()));
    assert!(matches!(event, WireEvent::Corrupt(error) if error.to_string().contains("bad frame")));
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
