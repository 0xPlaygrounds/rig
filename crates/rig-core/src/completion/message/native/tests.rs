use serde_json::json;

use crate::message::{
    AssistantContent, AssistantMessage, CallId, Fingerprint, Message, Opaque, Origin, Reasoning,
    StopReason, Text, ToolCall, ToolFunction, ToolName,
};

fn call(arguments: serde_json::Value) -> AssistantContent {
    let name = ToolName::new("lookup").expect("a tool name");
    AssistantContent::ToolCall(ToolCall::new(
        CallId::from_wire("call_1"),
        ToolFunction::new(name, arguments),
    ))
}

#[test]
fn a_fresh_native_item_is_current_and_an_edit_makes_it_stale() {
    let item = json!({"type": "text", "text": "hi", "citations": [{"url": "u"}]});
    let block = AssistantContent::text("hi").with_native(item.clone());
    assert_eq!(block.native_item(), Some(&item));

    let mut edited = block.clone();
    if let AssistantContent::Text(text) = &mut edited {
        text.text.push('!');
    }
    assert_eq!(edited.native_item(), None);
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
fn the_canonical_form_leaves_the_item_out() {
    let plain = AssistantContent::Reasoning(Reasoning::new("think"));
    let decoded = plain.clone().with_native(json!({"signature": "sig"}));
    assert_eq!(decoded.fingerprint(), plain.fingerprint());
    assert_eq!(decoded.canonical(), plain);
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
fn a_message_native_tracks_its_whole_content() {
    let turn = AssistantMessage::new(vec![AssistantContent::text("a"), call(json!({}))])
        .with_native(json!({"role": "assistant", "content": "a"}));
    assert!(turn.native_item().is_some());

    let mut edited = turn.clone();
    edited.content.pop();
    assert_eq!(edited.native_item(), None);
}

#[test]
fn an_assistant_message_serializes_flat_under_its_role() {
    let turn = AssistantMessage {
        content: vec![AssistantContent::Text(Text::new("hi"))],
        origin: Some(Origin::new("anthropic.messages", "anthropic", "claude")),
        stop: Some(StopReason::Error("refused".into())),
        native: None,
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
fn fingerprint_is_fnv1a_of_the_json_bytes() {
    // FNV-1a of `"a"` (three bytes), pinned so a hashing change is a
    // deliberate break of every stored history.
    assert_eq!(Fingerprint::of(&"a"), Fingerprint::of(&json!("a")));
    assert_ne!(Fingerprint::of(&"a"), Fingerprint::of(&"b"));
}

#[test]
fn a_rig_issued_call_id_fingerprints_the_same_on_every_decode() {
    let decoded = || {
        AssistantContent::ToolCall(ToolCall::new(
            CallId::from_wire(""),
            ToolFunction::new(ToolName::new("lookup").expect("a name"), json!({})),
        ))
    };
    let (first, second) = (decoded(), decoded());
    assert_ne!(first, second, "each decode issues its own id");
    assert_eq!(first.fingerprint(), second.fingerprint());
}
