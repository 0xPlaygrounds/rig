use rig_core::message::AssistantContent;
use rig_core::streaming::{BlockId, BlockKind, Delta, MintKind, StreamEvent};

use super::delivered_prefix;

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
