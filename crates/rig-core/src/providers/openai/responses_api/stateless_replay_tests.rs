//! rig#2269: items a stateless Responses client must send back unchanged.
//!
//! - `compaction` items round-trip verbatim on both the output and the input
//!   side of the wire;
//! - an output message's `phase` survives the trip through rig history and
//!   is re-sent on the assistant input item, never leaked onto a text block;

use super::*;
use crate::completion;
use crate::message::{self, Text};
use serde_json::json;

#[test]
fn compaction_output_item_round_trips_verbatim() {
    let wire = json!({
        "type": "compaction",
        "id": "cmp_123",
        "encrypted_content": "opaque-bytes",
        "status": "completed",
        "future_field": {"nested": [1, 2, 3]}
    });
    let output: Output = serde_json::from_value(wire.clone()).expect("compaction decodes");
    let Output::Compaction(fields) = &output else {
        panic!("expected Output::Compaction, got {output:?}");
    };
    assert_eq!(fields.get("id"), Some(&json!("cmp_123")));
    assert!(
        fields.get("type").is_none(),
        "the tag must not be duplicated inside the payload"
    );

    let back = serde_json::to_value(&output).expect("compaction re-serializes");
    assert_eq!(back, wire);
}

#[test]
fn compaction_input_item_round_trips_verbatim() {
    let wire = json!({
        "type": "compaction",
        "id": "cmp_123",
        "encrypted_content": "opaque-bytes"
    });
    let item: InputItem = serde_json::from_value(wire.clone()).expect("compaction input decodes");
    assert!(matches!(item.input, InputContent::Compaction(_)));
    let back = serde_json::to_value(&item).expect("compaction input re-serializes");
    assert_eq!(back, wire);
}

/// The exact window `/responses/compact` returns — regular items around an
/// opaque compaction item — decodes as `output[]` without dropping anything.
#[test]
fn compacted_window_decodes_with_every_item_typed() {
    let output: Vec<Output> = serde_json::from_value(json!([
        {"type": "compaction", "id": "cmp_1", "encrypted_content": "..."},
        {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": "hi", "annotations": []}]},
        {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "f",
         "arguments": "{}", "status": "completed"}
    ]))
    .expect("window decodes");
    assert!(matches!(output[0], Output::Compaction(_)));
    assert!(matches!(output[1], Output::Message(_)));
    assert!(matches!(output[2], Output::FunctionCall(_)));
}

#[test]
fn output_message_phase_decodes_and_is_absent_by_default() {
    let with: OutputMessage = serde_json::from_value(json!({
        "id": "msg_1", "role": "assistant", "status": "completed",
        "content": [], "phase": "final_answer"
    }))
    .expect("decodes");
    assert_eq!(with.phase.as_deref(), Some("final_answer"));
    let without: OutputMessage = serde_json::from_value(json!({
        "id": "msg_1", "role": "assistant", "status": "completed", "content": []
    }))
    .expect("decodes");
    assert_eq!(without.phase, None);
    let back = serde_json::to_value(&without).expect("serializes");
    assert!(
        back.get("phase").is_none(),
        "absent phase must not serialize as null"
    );
}

/// `Output::Message` → rig history → assistant input item: `phase` arrives on
/// the item and not on the text block.
#[test]
fn phase_survives_history_and_is_resent_on_the_assistant_item() {
    let output = Output::Message(OutputMessage {
        id: "msg_1".to_string(),
        role: OutputRole::Assistant,
        status: ResponseStatus::Completed,
        content: vec![AssistantContent::OutputText(OutputText::new("the answer"))],
        phase: Some("final_answer".to_string()),
    });
    let content = super::tests::folded_choice(vec![output]);
    let history = completion::Message::Assistant {
        id: Some("msg_1".to_string()),
        content,
    };

    let items = Vec::<InputItem>::try_from(history).expect("history converts");
    assert_eq!(items.len(), 1);
    let InputContent::Message(Message::Assistant {
        phase, content, id, ..
    }) = &items[0].input
    else {
        panic!("expected an assistant input item, got {:?}", items[0].input);
    };
    assert_eq!(id, "msg_1");
    assert_eq!(phase.as_deref(), Some("final_answer"));

    // Never on the block: the flatten would put it beside `text`.
    let block = serde_json::to_value(&content[0]).expect("block serializes");
    assert!(
        block.get("phase").is_none(),
        "phase leaked onto the text block: {block}"
    );

    let wire = serde_json::to_value(&items[0]).expect("item serializes");
    assert_eq!(wire["phase"], "final_answer");
    assert_eq!(wire["id"], "msg_1");
}

/// A message without a phase replays exactly as before: no `phase` key.
#[test]
fn history_without_phase_replays_without_the_key() {
    let history = completion::Message::Assistant {
        id: Some("msg_1".to_string()),
        content: vec![message::AssistantContent::Text(Text::new("plain"))],
    };
    let items = Vec::<InputItem>::try_from(history).expect("history converts");
    let wire = serde_json::to_value(&items[0]).expect("item serializes");
    assert!(wire.get("phase").is_none(), "{wire}");
}
