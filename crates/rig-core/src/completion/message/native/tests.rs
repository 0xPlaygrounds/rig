use serde_json::json;

use crate::message::{
    AssistantContent, AssistantMessage, CallId, Fingerprint, Message, Opaque, Origin, StopReason,
    Text, ToolCall, ToolFunction, ToolName,
};

fn call(arguments: serde_json::Value) -> AssistantContent {
    let name = ToolName::new("lookup").expect("a tool name");
    AssistantContent::ToolCall(ToolCall::new(
        CallId::from_wire("call_1"),
        ToolFunction::new(name, arguments),
    ))
}

#[test]
fn fingerprints_survive_a_serde_round_trip() {
    // Key order and float digits must survive, or a stored block would
    // never match its item again after a history is saved and loaded.
    let arguments = json!({"zeta": 1, "alpha": {"y": 0.1, "x": [2.718_281_828_459_1, 1e-7]}});
    let block = call(arguments).with_native(json!({"type": "function_call", "id": "fc_1"}));
    let saved = serde_json::to_string(&block).expect("serialize");
    let loaded: AssistantContent = serde_json::from_str(&saved).expect("deserialize");
    assert_eq!(loaded.fingerprint(), block.fingerprint());
    assert!(loaded.native_item().is_some());
}

#[test]
fn opaque_items_carry_no_separate_native() {
    let opaque = AssistantContent::Opaque(Opaque {
        item: json!({"type": "web_search_call", "id": "ws_1"}),
        replay: true,
    });
    assert_eq!(opaque.clone().with_native(json!({"ignored": true})), opaque);
    assert_eq!(opaque.native_item(), None);
}

#[test]
fn an_assistant_message_serializes_flat_under_its_role() {
    let turn = AssistantMessage {
        content: vec![AssistantContent::Text(Text::new("hi"))],
        origin: Some(Origin::new("anthropic.messages", "anthropic", "claude")),
        stop: Some(StopReason::Error("refused".into())),
    };
    let value = serde_json::to_value(Message::Assistant(turn.clone())).expect("serialize");
    assert_eq!(
        value,
        json!({
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
            "origin": {"api": "anthropic.messages", "provider": "anthropic", "model": "claude"},
            "stop": {"error": "refused"},
        })
    );
    assert_eq!(
        serde_json::from_value::<Message>(value).expect("deserialize"),
        Message::Assistant(turn)
    );
}

#[test]
fn a_store_that_writes_whole_numbers_as_integers_keeps_the_item_current() {
    let block = call(json!({"limit": 20.0, "scale": 0.5}))
        .with_native(json!({"type": "function_call", "id": "fc_1"}));
    let stored = serde_json::to_string(&block)
        .expect("a block serializes")
        .replace("20.0", "20");
    let loaded: AssistantContent = serde_json::from_str(&stored).expect("a stored block loads");
    assert!(loaded.native_item().is_some(), "{stored}");
    assert_ne!(
        Fingerprint::of(&json!({"scale": 0.5})),
        Fingerprint::of(&json!({"scale": 1}))
    );
}

#[test]
fn a_stored_item_that_cannot_be_read_loads_as_no_item() {
    for fingerprint in [json!(1234), json!("not hex"), json!(null)] {
        let block: AssistantContent = serde_json::from_value(json!({
            "type": "text",
            "text": "hi",
            "native": {"item": {"id": "msg_1"}, "fingerprint": fingerprint}
        }))
        .expect("the block loads");
        assert_eq!(block, AssistantContent::text("hi"), "{fingerprint}");
    }
    let current = AssistantContent::text("hi").with_native(json!({"id": "msg_1"}));
    let stored = serde_json::to_value(&current).expect("serializes");
    assert_eq!(
        serde_json::from_value::<AssistantContent>(stored).expect("loads"),
        current
    );
}
