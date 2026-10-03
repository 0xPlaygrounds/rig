#![allow(clippy::expect_used)]

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
        (vec![AssistantContent::text("before call")], true),
        "the open text is delivered; the unfinished call is not"
    );
}

/// A part still open cuts the prefix: what ended after it keeps no provider
/// item, so a call never replays without the reasoning it follows.
#[test]
fn blocks_after_an_unfinished_part_keep_no_provider_item() {
    let text = AssistantContent::text("hi").with_native(serde_json::json!({"id": "msg_1"}));
    let stream = |reasoning: bool| {
        let mut events = Vec::new();
        if reasoning {
            events.push(serde_json::json!(
                {"item": "event", "value": {"event": "start", "part": 0, "kind": "reasoning"}}
            ));
        }
        let part = events.len();
        events.push(serde_json::json!(
            {"item": "event", "value": {"event": "start", "part": part, "kind": "text"}}
        ));
        events.push(serde_json::json!(
            {"item": "event", "value": {"event": "end", "part": part, "content": text}}
        ));
        Transcript::parse_prefix(serde_json::Value::Array(events))
            .expect("a stream prefix in order")
            .into_items()
    };
    assert_eq!(
        delivered_prefix(&stream(false)),
        (vec![text.clone()], false)
    );
    let (prefix, cut) = delivered_prefix(&stream(true));
    assert!(cut);
    assert_eq!(prefix, vec![AssistantContent::text("hi")]);
}
