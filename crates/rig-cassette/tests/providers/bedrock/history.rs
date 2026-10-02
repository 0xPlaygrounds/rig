//! Item-shaped history checks on recorded Converse replies: a recorded whole
//! reply and the same reply restated as Converse stream events fold into one
//! turn, and that turn goes back to its model as the blocks Bedrock sent.

use aws_sdk_bedrockruntime::types as aws_bedrock;
use rig::bedrock::completion::{Converse, ConverseFrame};
use rig::bedrock::types::converse_output::{
    CitationGeneratedContent, ContentBlock, ConverseOutput, InternalConverseOutput,
    ReasoningContentBlock,
};
use rig::completion::{CompletionRequest, CompletionResponse, Message};
use rig_core::operation::Completion;
use rig_core::test_utils::history::assert_restated_agrees;
use rig_core::wire::{Mode, Operation, Wire};
use serde::Deserialize;

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
fn restated(content: &[ContentBlock], output: &InternalConverseOutput) -> Vec<ConverseFrame> {
    use aws_bedrock::{ContentBlockDelta as Delta, ReasoningContentBlockDelta as Thought};
    let mut events = Vec::new();
    for (index, block) in content.iter().enumerate() {
        let index = i32::try_from(index).expect("small reply");
        match block {
            ContentBlock::Text(text) => events.push(delta(index, Delta::Text(text.clone()))),
            ContentBlock::CitationsContent(cited) => {
                for content in cited.content.iter().flatten() {
                    if let CitationGeneratedContent::Text(text) = content {
                        events.push(delta(index, Delta::Text(text.clone())));
                    }
                }
            }
            ContentBlock::ToolUse(call) => {
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
                    .input(call.input.to_string())
                    .build()
                    .expect("tool delta builds");
                events.push(delta(index, Delta::ToolUse(input)));
            }
            ContentBlock::ReasoningContent(ReasoningContentBlock::ReasoningText(reasoning)) => {
                let text = Thought::Text(reasoning.text.clone());
                events.push(delta(index, Delta::ReasoningContent(text)));
                if let Some(signature) = &reasoning.signature {
                    let signature = Thought::Signature(signature.clone());
                    events.push(delta(index, Delta::ReasoningContent(signature)));
                }
            }
            ContentBlock::ReasoningContent(ReasoningContentBlock::RedactedContent(blob)) => {
                let bytes = aws_smithy_types::Blob::new(blob.inner.clone());
                events.push(delta(
                    index,
                    Delta::ReasoningContent(Thought::RedactedContent(bytes)),
                ));
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
            .stop_reason(aws_bedrock::StopReason::from(output.stop_reason.as_str()))
            .build()
            .expect("message stop builds"),
    ));
    events.push(aws_bedrock::ConverseStreamOutput::Metadata(
        aws_bedrock::ConverseStreamMetadataEvent::builder().build(),
    ));
    events.into_iter().map(ConverseFrame::Event).collect()
}

/// Assert `response`, a recorded unary reply from `model`, folds into the
/// same turn restated as a stream, and that the turn goes back to `model`
/// as the blocks Bedrock sent, blank text aside.
pub(super) fn assert_recorded_history(model: &str, response: &CompletionResponse) {
    let output = InternalConverseOutput::deserialize(&response.raw).expect("raw is the reply");
    let content = match &output.output {
        Some(ConverseOutput::Message(message)) => message.content.clone(),
        other => panic!("a recorded message: {other:?}"),
    };
    let wire = Converse::new(model);
    let opened = || ConverseFrame::Opened { request_id: None };
    let streamed = std::iter::once(opened()).chain(restated(&content, &output));
    assert_restated_agrees(
        &wire,
        [opened(), ConverseFrame::Whole(Box::new(output.clone()))],
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
    let sent: Vec<ContentBlock> = messages[1]
        .content
        .iter()
        .cloned()
        .map(|block| ContentBlock::try_from(block).expect("mirrors"))
        .collect();
    let recorded: Vec<ContentBlock> = content
        .into_iter()
        .filter(|block| !matches!(block, ContentBlock::Text(text) if text.trim().is_empty()))
        .collect();
    assert_eq!(sent, recorded, "the same model gets its blocks back");
}
