//! Item-shaped history checks on recorded Converse replies: a recorded whole
//! reply and the same reply restated as Converse stream events fold into one
//! turn, and that turn goes back to its model as the blocks Bedrock sent.
//! A response's `raw` is the body Bedrock sent, so the recorded blocks are
//! read from it.

use rig::bedrock::completion::{Converse, ConverseFrame};
use rig::completion::{CompletionRequest, CompletionResponse, Message};
use rig_core::operation::Completion;
use rig_core::test_utils::history::assert_restated_agrees;
use rig_core::wire::{Mode, Operation, Wire};
use serde_json::{Value, json};

/// `content` as the Converse stream events that carry it.
fn restated(content: &[Value], stop_reason: &str) -> Vec<ConverseFrame> {
    let mut events = Vec::new();
    for (index, block) in content.iter().enumerate() {
        let delta = |delta: Value| json!({ "contentBlockDelta": { "contentBlockIndex": index, "delta": delta } });
        if let Some(text) = block.get("text") {
            events.push(delta(json!({ "text": text })));
        } else if let Some(call) = block.get("toolUse") {
            let opened = json!({ "toolUseId": call["toolUseId"], "name": call["name"] });
            events.push(json!({ "contentBlockStart": {
                "contentBlockIndex": index, "start": { "toolUse": opened },
            } }));
            events.push(delta(
                json!({ "toolUse": { "input": call["input"].to_string() } }),
            ));
        } else if let Some(reasoning) = block.get("reasoningContent") {
            let reasoning = reasoning.get("reasoningText").unwrap_or(reasoning);
            events.push(delta(json!({ "reasoningContent": reasoning })));
        } else {
            panic!("no recorded reply holds {block}");
        }
        events.push(json!({ "contentBlockStop": { "contentBlockIndex": index } }));
    }
    events.push(json!({ "messageStop": { "stopReason": stop_reason } }));
    events.push(json!({ "metadata": {} }));
    events.into_iter().map(ConverseFrame::Event).collect()
}

/// Assert `response`, a recorded unary reply from `model`, folds into the
/// same turn restated as a stream, and that the turn goes back to `model`
/// as the blocks Bedrock sent, blank text aside.
pub(super) fn assert_recorded_history(model: &str, response: &CompletionResponse) {
    let content: Vec<Value> = response
        .raw
        .pointer("/output/message/content")
        .and_then(Value::as_array)
        .expect("raw is the recorded body")
        .clone();
    let stop_reason = response.raw["stopReason"].as_str().expect("a stop reason");
    let wire = Converse::new(model);
    assert_restated_agrees(
        &wire,
        [ConverseFrame::Whole(response.raw.clone())],
        restated(&content, stop_reason),
    );

    let turn = response.message().expect("the reply is a turn");
    let mut request = CompletionRequest::new("again").messages([Message::user("hi"), turn]);
    // The request declares the tools the turn calls, as a tool loop does.
    request.tools = response
        .tool_calls()
        .map(|call| rig::completion::ToolDefinition {
            name: call.function.name.clone(),
            description: String::new(),
            parameters: json!({ "type": "object" }),
        })
        .collect();
    let request = Completion::prepare(request, &wire.describe()).expect("prepares");
    let body = wire.encode(request, Mode::Unary).expect("encodes").body;
    let recorded: Vec<Value> = content
        .into_iter()
        .filter(|block| {
            !block["text"]
                .as_str()
                .is_some_and(|text| text.trim().is_empty())
        })
        .collect();
    assert_eq!(
        body["messages"][1]["content"],
        Value::Array(recorded),
        "the same model gets its blocks back"
    );
}
