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

/// A native item records the first of its wire's slots that holds a
/// blob, finds the blob from that record after a round trip through
/// storage, and records none when every slot is empty.
#[test]
fn a_reasoning_blob_is_read_where_its_wire_put_it() {
    use crate::message::{Blob, Native};
    let slots = ["/signature", "/data"];
    let mut redacted = Native {
        item: json!({"type": "redacted_thinking", "data": "r"}),
        fingerprint: Fingerprint::of(&"x"),
        blob_slot: None,
    };
    redacted.find_blob(&slots);
    let stored = serde_json::to_value(&redacted).expect("serializes");
    let stored: Native = serde_json::from_value(stored).expect("loads");
    let found = stored.reasoning_blob();
    let mut unsigned = Native {
        item: json!({"type": "thinking", "thinking": "t", "signature": ""}),
        ..redacted.clone()
    };
    unsigned.find_blob(&slots);
    assert_eq!(
        found,
        Some(Blob {
            key: "/data",
            value: "r"
        })
    );
    assert_eq!(
        (&unsigned.blob_slot, unsigned.reasoning_blob()),
        (&None, None)
    );
}

/// Stored native items carry these values, so a change to the projection or
/// to how it is hashed would stale every saved item.
#[test]
fn fingerprints_are_pinned() {
    use crate::completion::ToolDefinition;
    use crate::message::{
        DocumentSourceKind, Image, ImageDetail, ImageMediaType, Reasoning, ToolCall, ToolFunction,
    };
    let tool = |name: &str, parameters| {
        ToolDefinition::new(
            ToolName::new(name).expect("a tool name"),
            format!("{name} things"),
            parameters,
        )
    };
    let tools = [
        tool(
            "search",
            json!({"type": "object", "properties": {"q": {"type": "string"}}}),
        ),
        tool(
            "add",
            json!({"properties": {"b": {"maximum": 1.0}}, "type": "object"}),
        ),
    ];
    let arguments = json!({"b": {"z": 1, "a": 20.0}, "a": [1.5, 2.0, -0.0, 1e20]});
    let local = AssistantContent::ToolCall(ToolCall::new(
        CallId::from_wire(""),
        ToolFunction::new(ToolName::new("lookup").expect("a tool name"), json!({})),
    ));
    let image = AssistantContent::Image(Image {
        data: DocumentSourceKind::base64("aGk="),
        media_type: Some(ImageMediaType::PNG),
        detail: Some(ImageDetail::High),
        native: None,
    });
    let opaque = AssistantContent::Opaque(Opaque {
        item: json!({"type": "web_search_call"}),
        replay: true,
    });
    let reasoning = AssistantContent::Reasoning(Reasoning {
        redacted: true,
        ..Reasoning::new("plan")
    });
    let hex = |fingerprint: Fingerprint| format!("{:016x}", fingerprint.0);
    let pinned = [
        hex(AssistantContent::text("hello").fingerprint()),
        hex(reasoning.fingerprint()),
        hex(call(arguments).fingerprint()),
        hex(local.fingerprint()),
        hex(image.fingerprint()),
        hex(opaque.fingerprint()),
        hex(Fingerprint::of(&json!(tools))),
    ];
    assert_eq!(
        pinned,
        [
            "be5cdb8169d835fd", // text
            "c2dc7b0f4e836ea1", // reasoning
            "3df7dcbe7a89ecc8", // call with a provider id
            "7e2d775ebdb398fc", // call with a local id
            "37e9c9a0147c5070", // image
            "9604486c7b29d4c5", // opaque
            "2fc10d4b16d1b11c", // tools, as a request's context hashes them
        ]
    );
}
