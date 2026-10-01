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

/// Stream-fold regrouping moves blocks, and the provider items they carry
/// move with them; an item with no canonical form keeps its place among
/// the text.
#[test]
fn regrouping_moves_items_with_their_blocks() {
    use crate::message::{Reasoning, ToolCall, ToolFunction, ToolName, canonical_streamed_choice};
    let item = |kind: &str| {
        Sealed::new(
            Issuer::from("openai"),
            NativeItem::new("openai.responses", json!({"type": kind})),
        )
    };
    let call = AssistantContent::ToolCall(ToolCall {
        native: Some(item("function_call")),
        ..ToolCall::from_wire(
            "call_1",
            ToolFunction::new(ToolName::new("f").expect("a name"), json!({})),
        )
    });
    let text = AssistantContent::Text(Text {
        native: Some(item("message")),
        ..Text::new("hi")
    });
    let native = AssistantContent::Native(item("compaction"));
    let reasoning = AssistantContent::Reasoning(Reasoning::new("think").sealed("openai"));
    let regrouped = canonical_streamed_choice(vec![
        call.clone(),
        native.clone(),
        text.clone(),
        reasoning.clone(),
    ]);
    assert_eq!(regrouped, [reasoning, native, text, call]);
}

/// A block's canonical form sets aside the item it carries, and two blocks
/// are the same canonically whatever items or reasoning issuers they carry.
#[test]
fn canonical_form_sets_items_aside() {
    use crate::message::Reasoning;
    let text = AssistantContent::Text(Text {
        native: Some(Sealed::new(
            Issuer::from("openai"),
            NativeItem::new("openai.responses", json!({"type": "message"})),
        )),
        ..Text::new("hi")
    });
    assert_eq!(text.canonical(), AssistantContent::text("hi"));
    assert!(NativeItem::same_canonical(
        &text,
        &AssistantContent::text("hi")
    ));
    assert!(!NativeItem::same_canonical(
        &text,
        &AssistantContent::text("ho")
    ));
    assert!(NativeItem::same_canonical(
        &AssistantContent::Reasoning(Reasoning::new("r").sealed("openai")),
        &AssistantContent::Reasoning(Reasoning::new("r").sealed("anthropic")),
    ));
}

/// Replay form: members that state nothing or a documented default are
/// omitted unless the canonical encoding states them; without a canonical
/// encoding only defaults are.
#[test]
fn replay_form_omits_only_what_states_nothing() {
    use super::replay_form;
    let is_default =
        |member: &str, value: &serde_json::Value| member == "status" && value == "completed";
    let item = json!({"type": "message", "status": "completed", "phase": "x",
                      "content": [{"type": "output_text", "text": "hi", "annotations": []}]});
    let canonical = json!({"type": "message", "status": "completed",
                           "content": [{"type": "output_text", "text": "hi"}]});
    assert_eq!(
        replay_form(&item, Some(&canonical), is_default),
        json!({"type": "message", "status": "completed", "phase": "x",
               "content": [{"type": "output_text", "text": "hi"}]})
    );
    let unmodelled = json!({"type": "result", "status": "completed", "content": []});
    assert_eq!(
        replay_form(&unmodelled, None, is_default),
        json!({"type": "result", "content": []})
    );
}
