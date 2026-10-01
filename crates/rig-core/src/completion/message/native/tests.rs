use super::*;
use crate::message::{AssistantContent, Issuer, Message, Sealed};
use serde_json::json;

struct Alpha;
impl NativeDialect for Alpha {
    const FORMAT: WireFormat = WireFormat::from_static("alpha.v1");
    type Item = serde_json::Value;
}

struct Beta;
impl NativeDialect for Beta {
    const FORMAT: WireFormat = WireFormat::from_static("beta.v1");
    type Item = serde_json::Value;
}

#[test]
fn decodes_only_as_its_own_format() {
    let item = Native::new::<Alpha>(&json!({"type": "novel", "x": 1})).expect("json");
    assert_eq!(
        item.decode::<Alpha>().transpose().expect("json"),
        Some(json!({"type": "novel", "x": 1}))
    );
    assert!(item.decode::<Beta>().is_none());
    assert_eq!(item.kind(), Some("novel"));
}

#[test]
fn serializes_sealed_with_issuer_and_format() {
    let part = AssistantContent::Native(Sealed::new(
        Issuer::from("alpha"),
        Native::new::<Alpha>(&json!({"type": "novel"})).expect("json"),
    ));
    let value = serde_json::to_value(&part).expect("json");
    assert_eq!(
        value,
        json!({"type": "native", "issuer": "alpha", "format": "alpha.v1", "item": {"type": "novel"}})
    );
    let back: AssistantContent = serde_json::from_value(value).expect("json");
    assert_eq!(back, part);
}

#[test]
fn foreign_native_only_turn_does_not_replay() {
    let message = Message::Assistant {
        id: None,
        content: vec![AssistantContent::Native(Sealed::new(
            Issuer::from("alpha"),
            Native::new::<Alpha>(&json!({"type": "novel"})).expect("json"),
        ))],
    };
    let alpha = Issuer::from("alpha");
    assert!(message.replays_to(std::slice::from_ref(&alpha), Some(&Alpha::FORMAT)));
    // Another issuer, another format, or a wire with no native format.
    assert!(!message.replays_to(&[Issuer::from("beta")], Some(&Alpha::FORMAT)));
    assert!(!message.replays_to(std::slice::from_ref(&alpha), Some(&Beta::FORMAT)));
    assert!(!message.replays_to(std::slice::from_ref(&alpha), None));
}

#[test]
fn stream_fold_regrouping_keeps_native_items_in_place_with_text() {
    use crate::message::{Reasoning, ToolCall, ToolFunction, ToolName, canonical_streamed_choice};
    let native = |id: &str| {
        AssistantContent::Native(Sealed::new(
            Issuer::from("alpha"),
            Native::new::<Alpha>(&json!({"type": "server_tool_use", "id": id}))
                .expect("serializes"),
        ))
    };
    let call = AssistantContent::ToolCall(ToolCall::from_wire(
        "call_1",
        ToolFunction::new(ToolName::new("f").expect("name"), json!({})),
    ));
    let reasoning = AssistantContent::Reasoning(Reasoning::new("r").sealed("alpha"));
    let choice = vec![
        native("a"),
        native("b"),
        AssistantContent::text("cited"),
        call.clone(),
        reasoning.clone(),
    ];
    assert_eq!(
        canonical_streamed_choice(choice),
        vec![
            reasoning,
            native("a"),
            native("b"),
            AssistantContent::text("cited"),
            call
        ]
    );
}

#[test]
fn a_history_saved_before_native_items_still_loads() {
    // The tunnelled hosted-tool block of an older history is an empty text
    // block with provider metadata: it loads unchanged.
    let old = json!({
        "role": "assistant", "id": null,
        "content": [
            {"type": "text", "text": "", "additional_params": {"anthropic_content": {"type": "server_tool_use", "id": "srvtoolu_1"}}},
            {"type": "reasoning", "issuer": "anthropic", "id": null, "content": [{"type": "text", "content": {"text": "r"}}]},
            {"type": "text", "text": "answer"}
        ]
    });
    let message: Message = serde_json::from_value(old.clone()).expect("json");
    assert_eq!(
        crate::message::keys_lost_in_round_trip(
            &old,
            &serde_json::to_value(&message).expect("json")
        ),
        Vec::<String>::new()
    );
}
