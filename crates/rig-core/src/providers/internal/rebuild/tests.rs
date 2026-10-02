use serde_json::json;

use super::{Piece, Rebuilt, call_item};
use crate::message::{
    AssistantContent, AssistantMessage, CallId, Image, Opaque, Reasoning, Text, ToolCall,
    ToolFunction, ToolName,
};

fn call(id: &str, arguments: serde_json::Value) -> ToolCall {
    let name = ToolName::new("lookup").unwrap_or_else(|_| panic!("a tool name"));
    ToolCall::new(CallId::from_wire(id), ToolFunction::new(name, arguments))
}

#[test]
fn text_joins_with_nothing_and_reasoning_with_a_newline_per_field() {
    let turn = AssistantMessage::new(vec![
        AssistantContent::Reasoning(Reasoning::new("first"))
            .with_native(json!({"thinking": "first"})),
        AssistantContent::Text(Text::new("a")),
        AssistantContent::Text(Text::new("  ")),
        AssistantContent::Reasoning(Reasoning::new("second"))
            .with_native(json!({"thinking": "second"})),
        AssistantContent::Text(Text::new("b")),
        AssistantContent::Reasoning(Reasoning::new("unsigned")),
    ]);
    let rebuilt = Rebuilt::of(&turn);
    assert_eq!(rebuilt.text(), "ab");
    assert_eq!(
        rebuilt.reasoning(None),
        vec![("thinking".to_owned(), "first\nsecond".to_owned())]
    );
    assert_eq!(
        rebuilt.reasoning(Some("thinking")),
        vec![("thinking".to_owned(), "first\nsecond\nunsigned".to_owned())]
    );
    assert!(!rebuilt.has_parts());
}

#[test]
fn reasoning_fields_beside_its_text_are_kept_verbatim() {
    let details = json!([{"type": "reasoning.encrypted", "data": "sig"}]);
    let turn = AssistantMessage::new(vec![
        AssistantContent::Reasoning(Reasoning::new("why"))
            .with_native(json!({"reasoning": "why", "reasoning_details": details})),
    ]);
    let rebuilt = Rebuilt::of(&turn);
    assert_eq!(
        rebuilt.reasoning(None),
        vec![("reasoning".to_owned(), "why".to_owned())]
    );
    assert_eq!(rebuilt.fields.get("reasoning_details"), Some(&details));
}

#[test]
fn content_parts_opaque_items_and_images() {
    let thinking = json!({"type": "thinking", "thinking": [{"type": "text", "text": "hm"}]});
    let turn = AssistantMessage::new(vec![
        AssistantContent::Reasoning(Reasoning::new("hm")).with_native(thinking.clone()),
        AssistantContent::Image(Image::default()),
        AssistantContent::Opaque(Opaque {
            item: json!({"audio": {"id": "audio_1"}}),
            replay: true,
        }),
    ]);
    let rebuilt = Rebuilt::of(&turn);
    assert!(rebuilt.has_parts());
    assert!(rebuilt.reasoning(None).is_empty());
    let [
        Piece::Reasoning {
            part: Some(part), ..
        },
        Piece::Opaque(item),
    ] = rebuilt.pieces.as_slice()
    else {
        panic!("the image is left out and the rest keep their order");
    };
    assert_eq!(part, &thinking);
    assert_eq!(item, &json!({"audio": {"id": "audio_1"}}));
}

#[test]
fn a_call_keeps_its_item_with_canonical_arguments() {
    let item = json!({
        "id": "call_1",
        "type": "function",
        "function": {"name": "lookup", "arguments": {"q": "object form"}},
        "x_rig_field": true,
    });
    let turn = AssistantMessage::new(vec![
        AssistantContent::ToolCall(call("call_1", json!("{\"q\":\"rig\"}"))).with_native(item),
    ]);
    let rebuilt = Rebuilt::of(&turn);
    let [(call, item)] = rebuilt.calls.as_slice() else {
        panic!("one call");
    };
    assert_eq!(
        call_item(call, item.clone(), Some("call_1".to_owned()), true),
        json!({
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{\"q\":\"rig\"}"},
            "x_rig_field": true,
        })
    );
    assert_eq!(
        call_item(call, None, None, false),
        json!({"type": "function", "function": {"name": "lookup", "arguments": {"q": "rig"}}})
    );
}

#[test]
fn a_custom_call_takes_its_input_back() {
    let item =
        json!({"id": "call_c", "type": "custom", "custom": {"name": "lookup", "input": "x"}});
    let custom = call("call_c", json!({"input": "grep rig"}));
    assert_eq!(
        call_item(
            &custom,
            item.as_object().cloned(),
            Some("call_c".to_owned()),
            true
        ),
        json!({"id": "call_c", "type": "custom", "custom": {"name": "lookup", "input": "grep rig"}})
    );
}
