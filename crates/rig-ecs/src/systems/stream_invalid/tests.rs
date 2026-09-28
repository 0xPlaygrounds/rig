use rig_core::message::AssistantContent;
use rig_core::streaming::Transcript;

use super::delivered_prefix;

/// Text the model was still writing when an invalid call arrived is part of
/// the delivered prefix, although its part has not ended; a call that has
/// not ended is not.
#[test]
fn the_delivered_prefix_keeps_a_text_part_still_open() {
    let items = Transcript::parse_prefix(serde_json::json!([
        {"item": "event", "value": {"event": "start", "part": 0, "kind": "text"}},
        {"item": "event", "value": {"event": "text", "part": 0, "text": "before call"}},
        {"item": "event", "value": {"event": "start", "part": 1, "kind": "tool_call"}},
        {"item": "event", "value": {"event": "arguments", "part": 1, "json": "{}"}},
    ]))
    .expect("a stream prefix in order")
    .into_items();
    assert_eq!(
        delivered_prefix(&items),
        vec![AssistantContent::text("before call")],
        "the open text is delivered; the unfinished call is not"
    );
}
