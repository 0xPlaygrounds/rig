use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::completion::CompletionRequest;
use crate::message::{
    AdditionalParams, AssistantContent, Extension, Fingerprint, Issuer, Message, Opaque, Sealed,
    Text, canonical_streamed_choice, turn_delivered_no_answer,
};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Phase {
    phase: String,
}

impl Extension for Phase {
    const KEY: &'static str = "test_phase";
    const LEGACY_KEYS: &'static [&'static str] = &["phase_v0"];
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
struct Item(serde_json::Value);

impl Extension for Item {
    const KEY: &'static str = "test_item";
}

fn phase(value: &str) -> Phase {
    Phase {
        phase: value.to_owned(),
    }
}

fn opaque(issuer: &'static str, value: serde_json::Value) -> AssistantContent {
    AssistantContent::Opaque(Sealed::new(
        issuer,
        Opaque::of(&Item(value)).expect("serializes"),
    ))
}

#[test]
fn a_typed_value_reads_back_by_type_and_keeps_the_stored_shape() {
    let text = Text::new("hi").with_extension(&phase("final")).unwrap();
    assert_eq!(text.extension::<Phase>().unwrap(), Some(phase("final")));
    // An absent type is absent, not an error.
    assert_eq!(text.extension::<Item>().unwrap(), None);
    // The stored form is the plain object earlier releases wrote.
    assert_eq!(
        serde_json::to_value(&text).unwrap(),
        json!({"text": "hi", "additional_params": {"test_phase": {"phase": "final"}}})
    );
}

#[test]
fn a_malformed_stored_value_is_an_error_not_absence() {
    let params = AdditionalParams::new(
        json!({"test_phase": {"phase": 7}})
            .as_object()
            .cloned()
            .unwrap_or_default(),
    )
    .expect("non-empty");
    let error = params
        .extension::<Phase>()
        .expect_err("a number is not a phase");
    assert_eq!(error.key, "test_phase");
}

#[test]
fn legacy_keys_read_and_the_current_key_writes() {
    let mut params = AdditionalParams::new(
        json!({"phase_v0": {"phase": "old"}, "other": 1})
            .as_object()
            .cloned()
            .unwrap_or_default(),
    )
    .expect("non-empty");
    assert_eq!(params.extension::<Phase>().unwrap(), Some(phase("old")));
    params.insert_extension(&phase("new")).unwrap();
    assert_eq!(
        params.clone().into_value(),
        json!({"test_phase": {"phase": "new"}, "other": 1})
    );
    let rest = params
        .without_extension::<Phase>()
        .expect("`other` is left");
    assert_eq!(rest.into_value(), json!({"other": 1}));
}

#[test]
fn fingerprints_are_fnv1a_and_serialize_as_hex() {
    // FNV-1a 64 reference vectors.
    assert_eq!(
        serde_json::to_value(Fingerprint::of("")).unwrap(),
        json!("cbf29ce484222325")
    );
    assert_eq!(
        serde_json::to_value(Fingerprint::of("a")).unwrap(),
        json!("af63dc4c8601ec8c")
    );
    let read: Fingerprint = serde_json::from_value(json!("af63dc4c8601ec8c")).unwrap();
    assert!(read.matches("a"));
    assert!(!read.matches("b"));
}

#[test]
fn an_opaque_item_round_trips_through_history_serde() {
    let message = Message::Assistant {
        id: None,
        content: vec![opaque("vendor", json!({"type": "new_item", "x": [1, 2]}))],
    };
    let stored = serde_json::to_value(&message).unwrap();
    assert_eq!(
        stored["content"][0],
        json!({
            "type": "opaque",
            "issuer": "vendor",
            "extensions": {"test_item": {"type": "new_item", "x": [1, 2]}},
        })
    );
    let loaded: Message = serde_json::from_value(stored).unwrap();
    assert_eq!(loaded, message);
}

#[test]
fn an_opaque_item_opens_only_to_its_issuer() {
    let AssistantContent::Opaque(item) = opaque("vendor", json!({"type": "x"})) else {
        unreachable!()
    };
    assert_eq!(
        item.extension_for::<Item>(&[Issuer::from("vendor")])
            .unwrap(),
        Some(Item(json!({"type": "x"})))
    );
    assert_eq!(
        item.extension_for::<Item>(&[Issuer::from("other")])
            .unwrap(),
        None
    );
}

/// The cross-dialect rule at the request boundary: an assistant message
/// holding only items no replay issuer opens is left out, like one holding
/// only foreign reasoning.
#[test]
fn foreign_opaque_only_turns_are_left_out_of_a_request() {
    let request = CompletionRequest::new(Message::user("hi"))
        .message(Message::Assistant {
            id: None,
            content: vec![opaque("vendor", json!({"type": "x"}))],
        })
        .message(Message::user("again"));
    let kept = request
        .clone()
        .replayable_to(&[Issuer::from("other")])
        .unwrap();
    assert_eq!(kept.chat_history.len(), 2);
    let kept = request.replayable_to(&[Issuer::from("vendor")]).unwrap();
    assert_eq!(kept.chat_history.len(), 3);
}

/// Stream-fold regrouping keeps an opaque item in its place among the text,
/// because a hosted step and the text citing it must stay in order.
#[test]
fn regrouping_keeps_opaque_items_with_text_in_order() {
    let step = opaque("vendor", json!({"type": "search"}));
    let choice = vec![
        step.clone(),
        AssistantContent::text("answer"),
        AssistantContent::tool_call(
            "call_1",
            crate::message::ToolName::new("t").unwrap(),
            json!({}),
        ),
        AssistantContent::reasoning("vendor", "thinking"),
    ];
    let regrouped = canonical_streamed_choice(choice);
    assert!(matches!(regrouped[0], AssistantContent::Reasoning(_)));
    assert_eq!(regrouped[1], step);
    assert!(matches!(regrouped[2], AssistantContent::Text(_)));
    assert!(matches!(regrouped[3], AssistantContent::ToolCall(_)));
}

#[test]
fn an_opaque_item_alone_is_not_an_answer() {
    assert!(turn_delivered_no_answer(&[opaque(
        "vendor",
        json!({"type": "x"})
    )]));
}
