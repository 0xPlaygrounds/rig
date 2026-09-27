use rig_core::message::AssistantContent;
use rig_core::streaming::{
    BlockClose, BlockId, BlockKind, Delta, MintKind, StreamEvent, ToolCallEnd, UnparseableToolInput,
};

use super::{completed_call_id, completed_calls, delivered_prefix};

/// A call a handler streamed as raw events, its end carrying no block.
fn raw_call(id: &BlockId, name: &str, arguments: &str) -> Vec<StreamEvent> {
    vec![
        StreamEvent::BlockStart {
            id: id.clone(),
            kind: BlockKind::ToolCall,
        },
        StreamEvent::BlockDelta {
            id: id.clone(),
            delta: Delta::ToolName {
                name: name.to_owned(),
            },
        },
        StreamEvent::BlockDelta {
            id: id.clone(),
            delta: Delta::ToolArguments {
                arguments: arguments.to_owned(),
            },
        },
        StreamEvent::BlockEnd {
            id: id.clone(),
            end: BlockClose::ToolCall(ToolCallEnd::new(UnparseableToolInput::Error)),
            block: None,
        },
    ]
}

/// A handler need not write its stream through the sink: a call whose end
/// carries no block resolves to the call its events assemble, so a repair
/// or an ignore of it reaches later prefixes.
#[test]
fn a_raw_call_end_resolves_to_the_call_its_events_assemble() {
    let call = BlockId::wire("call_1");
    let events = raw_call(&call, "wrong", r#"{"q":1}"#);
    let completed: Vec<_> = completed_calls(&events)
        .into_iter()
        .map(|call| call.map(|(_, call)| (call.function.name, call.function.arguments)))
        .collect();
    assert_eq!(
        completed,
        vec![
            None,
            None,
            None,
            Some(("wrong".to_owned(), serde_json::json!({"q": 1})))
        ],
        "only the end completes the call, as its events assembled it"
    );
    let resolved = completed_call_id(&events, 1);
    assert!(resolved.is_some(), "the raw end completes the call");
    let mut after = events.clone();
    after.extend(raw_call(&BlockId::wire("call_2"), "other", "{}"));
    let first = delivered_prefix(&after).and_then(|prefix| {
        prefix.into_iter().find_map(|part| match part {
            AssistantContent::ToolCall(call) if call.function.name == "wrong" => Some(call),
            _ => None,
        })
    });
    assert_eq!(first.as_ref().map(|call| call.id.clone()), resolved);
    assert_eq!(
        first.map(|call| call.function.arguments),
        Some(serde_json::json!({"q": 1}))
    );
}

/// Text the model was still writing when an invalid call's name arrived
/// is part of the delivered prefix, although its block has not ended.
#[test]
fn the_delivered_prefix_keeps_a_block_still_open() {
    let text = MintKind::Text.for_wire_index(0);
    let call = BlockId::wire("call_1");
    let events = [
        StreamEvent::BlockStart {
            id: text.clone(),
            kind: BlockKind::Text {
                additional_params: None,
            },
        },
        StreamEvent::text(text, "before call"),
        StreamEvent::BlockStart {
            id: call.clone(),
            kind: BlockKind::ToolCall,
        },
        StreamEvent::BlockDelta {
            id: call,
            delta: Delta::ToolName {
                name: "wrong".to_owned(),
            },
        },
    ];
    assert_eq!(
        delivered_prefix(&events),
        Some(vec![AssistantContent::text("before call")]),
        "the open text is delivered; the unfinished call is not"
    );
}
