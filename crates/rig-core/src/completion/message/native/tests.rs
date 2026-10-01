use serde_json::json;

use super::{NativeItem, same_wire_value};
use crate::message::{AssistantContent, Issuer, Message, Sealed, Text};

#[test]
fn absent_members_compare_equal() {
    assert!(same_wire_value(
        &json!({"type": "text", "text": "hi", "citations": null, "annotations": []}),
        &json!({"text": "hi", "type": "text"}),
    ));
    assert!(same_wire_value(
        &json!({"type": "text", "text": ""}),
        &json!({"type": "text"})
    ));
    assert!(!same_wire_value(
        &json!({"type": "text", "text": "hi", "phase": "final_answer"}),
        &json!({"type": "text", "text": "hi"}),
    ));
    assert!(!same_wire_value(&json!([1, 2]), &json!([1])));
}

#[test]
fn a_native_opens_only_for_its_dialect_and_issuer() {
    let sealed = Sealed::new(
        Issuer::from("anthropic"),
        NativeItem::new("anthropic.messages", json!({"type": "compaction"})),
    );
    let anthropic = [Issuer::from("anthropic")];
    assert!(
        sealed
            .open_native("anthropic.messages", &anthropic)
            .is_some()
    );
    // Bedrock Claude stamps the same issuer but speaks another wire.
    assert!(sealed.open_native("bedrock.converse", &anthropic).is_none());
    assert!(
        sealed
            .open_native("anthropic.messages", &[Issuer::from("zai")])
            .is_none()
    );
}

#[test]
fn native_blocks_and_residue_round_trip_through_serde() {
    let native = Sealed::new(
        Issuer::from("openai"),
        NativeItem::new(
            "openai.responses",
            json!({"type": "compaction", "id": "cmp_1"}),
        ),
    );
    let message = Message::Assistant {
        id: None,
        content: vec![
            AssistantContent::Native(native.clone()),
            AssistantContent::Text(Text {
                native: Some(native),
                ..Text::new("hi")
            }),
        ],
    };
    let json = serde_json::to_value(&message).expect("serializes");
    assert_eq!(json["content"][0]["type"], "native");
    assert_eq!(json["content"][0]["dialect"], "openai.responses");
    assert_eq!(json["content"][0]["issuer"], "openai");
    let back: Message = serde_json::from_value(json).expect("deserializes");
    assert_eq!(back, message);
}

#[test]
fn histories_without_natives_keep_their_shape() {
    let old = json!({"role": "assistant", "id": null, "content": [{"type": "text", "text": "hi"}]});
    let message: Message = serde_json::from_value(old.clone()).expect("an older history loads");
    assert_eq!(serde_json::to_value(&message).expect("serializes"), old);
}

#[test]
fn foreign_natives_alone_do_not_replay() {
    let message = Message::Assistant {
        id: None,
        content: vec![AssistantContent::Native(Sealed::new(
            Issuer::from("anthropic"),
            NativeItem::new("anthropic.messages", json!({"type": "server_tool_use"})),
        ))],
    };
    assert!(message.replays_to(&[Issuer::from("anthropic")]));
    assert!(!message.replays_to(&[Issuer::from("openai")]));
}
