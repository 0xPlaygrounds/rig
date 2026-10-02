//! Item-shaped history checks on recorded Converse replies: a recorded whole
//! reply and the same reply restated as Converse stream events fold into one
//! turn, and that turn goes back to its model as the blocks Bedrock sent.
//! A response's `raw` is the body Bedrock sent, so the recorded blocks are
//! read from it.

use aws_sdk_bedrockruntime::operation::converse::ConverseOutput;
use aws_sdk_bedrockruntime::types as aws_bedrock;
use base64::{Engine, prelude::BASE64_STANDARD};
use rig::bedrock::completion::{Converse, ConverseFrame};
use rig::completion::{CompletionRequest, CompletionResponse, Message};
use rig_core::operation::Completion;
use rig_core::test_utils::history::assert_restated_agrees;
use rig_core::wire::{Mode, Operation, Wire};
use serde_json::Value;

fn document(value: &Value) -> aws_smithy_types::Document {
    use aws_smithy_types::{Document, Number};
    match value {
        Value::Null => Document::Null,
        Value::Bool(flag) => Document::Bool(*flag),
        Value::Number(number) => match (number.as_u64(), number.as_i64()) {
            (Some(unsigned), _) => Document::Number(Number::PosInt(unsigned)),
            (None, Some(signed)) => Document::Number(Number::NegInt(signed)),
            (None, None) => Document::Number(Number::Float(number.as_f64().unwrap_or_default())),
        },
        Value::String(text) => Document::String(text.clone()),
        Value::Array(values) => Document::Array(values.iter().map(document).collect()),
        Value::Object(fields) => Document::Object(
            fields
                .iter()
                .map(|(key, value)| (key.clone(), document(value)))
                .collect(),
        ),
    }
}

/// The SDK block of a recorded Converse block.
fn block(value: &Value) -> aws_bedrock::ContentBlock {
    if let Some(text) = value.get("text").and_then(Value::as_str) {
        return aws_bedrock::ContentBlock::Text(text.to_owned());
    }
    if let Some(call) = value.get("toolUse") {
        return aws_bedrock::ContentBlock::ToolUse(
            aws_bedrock::ToolUseBlock::builder()
                .tool_use_id(call["toolUseId"].as_str().expect("an id"))
                .name(call["name"].as_str().expect("a name"))
                .input(document(&call["input"]))
                .build()
                .expect("tool use builds"),
        );
    }
    let reasoning = value
        .get("reasoningContent")
        .unwrap_or_else(|| panic!("no recorded reply holds {value}"));
    if let Some(redacted) = reasoning.get("redactedContent").and_then(Value::as_str) {
        let bytes = BASE64_STANDARD.decode(redacted).expect("base64");
        return aws_bedrock::ContentBlock::ReasoningContent(
            aws_bedrock::ReasoningContentBlock::RedactedContent(aws_smithy_types::Blob::new(bytes)),
        );
    }
    let text = &reasoning["reasoningText"];
    aws_bedrock::ContentBlock::ReasoningContent(aws_bedrock::ReasoningContentBlock::ReasoningText(
        aws_bedrock::ReasoningTextBlock::builder()
            .text(text["text"].as_str().unwrap_or_default())
            .set_signature(text["signature"].as_str().map(str::to_owned))
            .build()
            .expect("reasoning builds"),
    ))
}

fn delta(index: i32, delta: aws_bedrock::ContentBlockDelta) -> aws_bedrock::ConverseStreamOutput {
    aws_bedrock::ConverseStreamOutput::ContentBlockDelta(
        aws_bedrock::ContentBlockDeltaEvent::builder()
            .content_block_index(index)
            .delta(delta)
            .build()
            .expect("delta builds"),
    )
}

/// `content` as the Converse stream events that carry it.
fn restated(content: &[aws_bedrock::ContentBlock], stop_reason: &str) -> Vec<ConverseFrame> {
    use aws_bedrock::{ContentBlockDelta as Delta, ReasoningContentBlockDelta as Thought};
    let mut events = Vec::new();
    for (index, block) in content.iter().enumerate() {
        let index = i32::try_from(index).expect("small reply");
        match block {
            aws_bedrock::ContentBlock::Text(text) => {
                events.push(delta(index, Delta::Text(text.clone())));
            }
            aws_bedrock::ContentBlock::ToolUse(call) => {
                events.push(aws_bedrock::ConverseStreamOutput::ContentBlockStart(
                    aws_bedrock::ContentBlockStartEvent::builder()
                        .content_block_index(index)
                        .start(aws_bedrock::ContentBlockStart::ToolUse(
                            aws_bedrock::ToolUseBlockStart::builder()
                                .tool_use_id(&call.tool_use_id)
                                .name(&call.name)
                                .build()
                                .expect("tool start builds"),
                        ))
                        .build()
                        .expect("start builds"),
                ));
                let input = aws_bedrock::ToolUseBlockDelta::builder()
                    .input(json(&call.input).to_string())
                    .build()
                    .expect("tool delta builds");
                events.push(delta(index, Delta::ToolUse(input)));
            }
            aws_bedrock::ContentBlock::ReasoningContent(
                aws_bedrock::ReasoningContentBlock::ReasoningText(reasoning),
            ) => {
                let text = Thought::Text(reasoning.text.clone());
                events.push(delta(index, Delta::ReasoningContent(text)));
                if let Some(signature) = &reasoning.signature {
                    let signature = Thought::Signature(signature.clone());
                    events.push(delta(index, Delta::ReasoningContent(signature)));
                }
            }
            aws_bedrock::ContentBlock::ReasoningContent(
                aws_bedrock::ReasoningContentBlock::RedactedContent(blob),
            ) => {
                let bytes = Thought::RedactedContent(blob.clone());
                events.push(delta(index, Delta::ReasoningContent(bytes)));
            }
            other => panic!("no recorded reply holds {other:?}"),
        }
        events.push(aws_bedrock::ConverseStreamOutput::ContentBlockStop(
            aws_bedrock::ContentBlockStopEvent::builder()
                .content_block_index(index)
                .build()
                .expect("stop builds"),
        ));
    }
    events.push(aws_bedrock::ConverseStreamOutput::MessageStop(
        aws_bedrock::MessageStopEvent::builder()
            .stop_reason(aws_bedrock::StopReason::from(stop_reason))
            .build()
            .expect("message stop builds"),
    ));
    events.push(aws_bedrock::ConverseStreamOutput::Metadata(
        aws_bedrock::ConverseStreamMetadataEvent::builder().build(),
    ));
    events.into_iter().map(ConverseFrame::Event).collect()
}

fn json(document: &aws_smithy_types::Document) -> Value {
    use aws_smithy_types::{Document, Number};
    match document {
        Document::Null => Value::Null,
        Document::Bool(flag) => Value::Bool(*flag),
        Document::Number(Number::PosInt(number)) => Value::from(*number),
        Document::Number(Number::NegInt(number)) => Value::from(*number),
        Document::Number(Number::Float(number)) => Value::from(*number),
        Document::String(text) => Value::String(text.clone()),
        Document::Array(values) => Value::Array(values.iter().map(json).collect()),
        Document::Object(fields) => Value::Object(
            fields
                .iter()
                .map(|(key, value)| (key.clone(), json(value)))
                .collect(),
        ),
    }
}

/// Assert `response`, a recorded unary reply from `model`, folds into the
/// same turn restated as a stream, and that the turn goes back to `model`
/// as the blocks Bedrock sent, blank text aside.
pub(super) fn assert_recorded_history(model: &str, response: &CompletionResponse) {
    let content: Vec<aws_bedrock::ContentBlock> = response
        .raw
        .pointer("/output/message/content")
        .and_then(Value::as_array)
        .expect("raw is the recorded body")
        .iter()
        .map(block)
        .collect();
    let stop_reason = response.raw["stopReason"].as_str().expect("a stop reason");
    let output = ConverseOutput::builder()
        .output(aws_bedrock::ConverseOutput::Message(
            aws_bedrock::Message::builder()
                .role(aws_bedrock::ConversationRole::Assistant)
                .set_content(Some(content.clone()))
                .build()
                .expect("message builds"),
        ))
        .stop_reason(aws_bedrock::StopReason::from(stop_reason))
        .build()
        .expect("output builds");
    let wire = Converse::new(model);
    let opened = || ConverseFrame::Opened { request_id: None };
    let streamed = std::iter::once(opened()).chain(restated(&content, stop_reason));
    assert_restated_agrees(
        &wire,
        [opened(), ConverseFrame::Whole(Box::new(output))],
        streamed,
    );

    let turn = response.message().expect("the reply is a turn");
    let request = CompletionRequest::new("again").messages([Message::user("hi"), turn]);
    let request = Completion::prepare(request, &wire.describe()).expect("prepares");
    let messages = wire
        .encode(request, Mode::Unary)
        .expect("encodes")
        .request
        .messages()
        .expect("converts");
    let recorded: Vec<aws_bedrock::ContentBlock> = content
        .into_iter()
        .filter(|block| {
            !matches!(block, aws_bedrock::ContentBlock::Text(text) if text.trim().is_empty())
        })
        .collect();
    assert_eq!(
        messages[1].content, recorded,
        "the same model gets its blocks back"
    );
}
