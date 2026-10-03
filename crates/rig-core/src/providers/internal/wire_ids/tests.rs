//! Synthetic transcript tests cover identities that cannot be requested reliably
//! from a live provider. Adapter request-boundary tests cover their use.

use super::*;
use crate::message::{ToolCall, ToolFunction, ToolResult};

fn call(id: CallId) -> Message {
    Message::Assistant(crate::message::AssistantMessage::new(vec![
        AssistantContent::ToolCall(ToolCall::new(
            id,
            ToolFunction::new(
                crate::message::ToolName::new("test").expect("tool name"),
                serde_json::json!({}),
            ),
        )),
    ]))
}

fn result(id: CallId) -> Message {
    Message::User {
        content: vec![UserContent::ToolResult(ToolResult {
            is_error: false,
            call: id,
            name: crate::message::ToolName::new("possibly_repaired").expect("tool name"),
            content: vec![crate::message::ToolResultContent::text("")],
        })],
    }
}

fn provider(id: &str) -> CallId {
    CallId::from_wire(id)
}

fn local() -> CallId {
    CallId::from_wire("")
}

#[test]
fn wire_slot_assignment_is_atomic_and_uses_original_content_order() {
    let function = || {
        ToolFunction::new(
            crate::message::ToolName::new("test").expect("tool name"),
            serde_json::json!({}),
        )
    };
    let history = vec![Message::Assistant(crate::message::AssistantMessage::new(
        vec![
            AssistantContent::text("before"),
            AssistantContent::ToolCall(ToolCall::new(provider("first"), function())),
            AssistantContent::text("between"),
            AssistantContent::ToolCall(ToolCall::new(provider("second"), function())),
        ],
    ))];
    let ids = WireIds::new(&history);
    for count in [1, 3] {
        let mut slots = vec!["unchanged".to_owned(); count];
        assert!(ids.apply(0, slots.iter_mut()).is_err());
        assert!(slots.iter().all(|slot| slot == "unchanged"));
    }
    let mut slots = [String::new(), String::new()];
    ids.apply(0, slots.iter_mut()).expect("two slots");
    assert_eq!(slots, ["first", "second"]);
}

#[test]
fn a_rig_issued_id_is_one_alias_that_no_provider_id_takes() {
    let issued = local();
    let history = vec![
        call(provider("tool-0")),
        call(issued.clone()),
        result(issued),
        call(local()),
    ];
    let ids = WireIds::new(&history);
    assert_eq!(ids.get(0, 0), Some("tool-0"));
    assert_eq!(ids.get(1, 0), Some("tool-1"));
    assert_eq!(ids.get(1, 0), ids.get(2, 0), "a result answers its call");
    assert_eq!(ids.get(3, 0), Some("tool-2"));
}

#[test]
fn a_provider_result_without_its_call_keeps_the_providers_id() {
    let history = vec![result(provider("real"))];
    assert_eq!(WireIds::new(&history).get(0, 0), Some("real"));
}
