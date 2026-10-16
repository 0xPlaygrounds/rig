use super::{
    Key, Scan, WireEvent, classify_chat_completions_frame, classify_marker_keyed_frame,
    classify_or_untagged, classify_tagged_frame,
};

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

/// A lone surrogate escape is not valid JSON text, though the key scan
/// skips over it: a frame holding one is corrupt, never passed through as
/// unknown.
#[test]
fn an_unknown_frame_that_is_not_valid_json_text_is_corrupt() {
    let event =
        classify_tagged_frame::<TestEvent>(r#"{"type":"ping","note":"\ud800"}"#, "type", known);
    assert!(matches!(event, WireEvent::Corrupt(_)), "{event:?}");
    let event = classify_marker_keyed_frame::<TestChunk>(r#"{"note":"\ud800"}"#, &["choices"]);
    assert!(matches!(event, WireEvent::Corrupt(_)), "{event:?}");
}

/// A marker key may repeat; the frame still takes the typed decode.
#[test]
fn a_repeated_marker_key_is_recognized() {
    let event = classify_marker_keyed_frame::<serde_json::Value>(
        r#"{"choices":[1],"choices":[]}"#,
        &["choices"],
    );
    assert!(matches!(event, WireEvent::Known(_)), "{event:?}");
}

/// Only a frame without the tag falls back: a tagged frame that fails its
/// typed decode stays corrupt rather than passing for the fallback's shape.
#[test]
fn only_an_untagged_frame_falls_back() {
    let classify = |frame: &str| {
        classify_or_untagged(
            frame,
            "type",
            known,
            |event: TestEvent| event,
            |_| {
                WireEvent::Known(TestEvent::TextDelta {
                    delta: "fallback".to_owned(),
                })
            },
        )
    };
    let event = classify(r#"{"type":"text.delta","delta":5}"#);
    assert!(matches!(event, WireEvent::Corrupt(_)), "{event:?}");
    let event = classify(r#"{"delta":"hi"}"#);
    assert!(matches!(event, WireEvent::Known(_)), "{event:?}");
}

/// The key scan says what it expected when handed something else.
#[test]
fn the_key_scan_names_what_it_expected() {
    use serde::de::{DeserializeSeed, Deserializer};
    let error = serde_json::Value::from(1)
        .deserialize_map(Scan(&["type"]))
        .expect_err("a number is no object");
    assert!(
        error.to_string().contains("expected a JSON object"),
        "{error}"
    );
    let error = Key(&["type"])
        .deserialize(serde_json::Value::from(1))
        .expect_err("a number is no key");
    assert!(
        error.to_string().contains("expected an object key"),
        "{error}"
    );
}
