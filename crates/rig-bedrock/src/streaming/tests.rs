use super::*;
use crate::completion::{Converse, ConverseRequest};
use crate::types::assistant_content::normalize_usage;
use aws_smithy_types::Blob;
use futures::StreamExt;
use rig_core::completion::{CompletionResponse, FinishReason};
use rig_core::driver::{Exchange, Model, Opened, Opening, Transport};
use rig_core::message::{AssistantContent, Opaque, StopReason as Stop};
use rig_core::streaming::{Item, StreamEvent};
use rig_core::test_utils::history::{assert_every_variant, assert_restated_agrees, decode};
use rig_core::wire::Mode;
use serde_json::json;

const NOVA: &str = "amazon.nova-lite-v1:0";

/// Replays scripted Converse events after the frame naming the model.
#[derive(Clone)]
struct Scripted(std::sync::Arc<std::sync::Mutex<Vec<aws_bedrock::ConverseStreamOutput>>>);

impl Transport<Converse> for Scripted {
    fn send(&self, _payload: ConverseRequest, _exchange: Exchange) -> Opening<ConverseFrame> {
        let events = std::mem::take(&mut *self.0.lock().expect("script lock"));
        let opened = ConverseFrame::Opened { request_id: None };
        Opening::ready(Opened::new(futures::stream::iter(
            std::iter::once(opened)
                .chain(events.into_iter().map(ConverseFrame::Event))
                .map(Ok),
        )))
    }
}

/// The stream events and outcome of scripted `events`, through the driver.
async fn driven(
    events: Vec<aws_bedrock::ConverseStreamOutput>,
) -> (Vec<StreamEvent>, Result<CompletionResponse, ProviderError>) {
    let transport = Scripted(std::sync::Arc::new(std::sync::Mutex::new(events)));
    let mut stream = Model::new(Converse::new(NOVA), transport)
        .stream(rig_core::completion::CompletionRequest::new("hi"))
        .expect("the stream opens");
    let mut seen = Vec::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(Item::Event(event)) => seen.push(event),
            Ok(Item::Unknown(_)) => {}
            Err(_) => break,
        }
    }
    (seen, stream.finish().await)
}

fn streamed(
    events: Vec<aws_bedrock::ConverseStreamOutput>,
) -> Result<CompletionResponse, ProviderError> {
    let opened = ConverseFrame::Opened { request_id: None };
    let frames = std::iter::once(opened).chain(events.into_iter().map(ConverseFrame::Event));
    decode(&Converse::new(NOVA), Mode::Streaming, frames)
}

/// The response the decoder folds the whole reply `output` into.
pub(crate) fn unary_as(
    model: &str,
    output: ConverseOutput,
) -> Result<CompletionResponse, ProviderError> {
    let frames = [
        ConverseFrame::Opened { request_id: None },
        ConverseFrame::Whole(Box::new(output)),
    ];
    decode(&Converse::new(model), Mode::Unary, frames)
}

fn usage(input: i32, output: i32) -> aws_bedrock::TokenUsage {
    aws_bedrock::TokenUsage::builder()
        .input_tokens(input)
        .output_tokens(output)
        .total_tokens(input + output)
        .build()
        .expect("usage builds")
}

/// A whole assistant reply holding `content`.
pub(crate) fn reply_of(
    content: Vec<aws_bedrock::ContentBlock>,
    stop_reason: &str,
) -> ConverseOutput {
    let message = aws_bedrock::Message::builder()
        .role(aws_bedrock::ConversationRole::Assistant)
        .set_content(Some(content))
        .build()
        .expect("message builds");
    ConverseOutput::builder()
        .output(aws_bedrock::ConverseOutput::Message(message))
        .stop_reason(aws_bedrock::StopReason::from(stop_reason))
        .usage(usage(3, 1))
        .build()
        .expect("output builds")
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

fn start(index: i32, start: aws_bedrock::ContentBlockStart) -> aws_bedrock::ConverseStreamOutput {
    aws_bedrock::ConverseStreamOutput::ContentBlockStart(
        aws_bedrock::ContentBlockStartEvent::builder()
            .content_block_index(index)
            .start(start)
            .build()
            .expect("start builds"),
    )
}

fn text(index: i32, text: &str) -> aws_bedrock::ConverseStreamOutput {
    delta(index, aws_bedrock::ContentBlockDelta::Text(text.to_owned()))
}

fn thought(
    index: i32,
    thought: aws_bedrock::ReasoningContentBlockDelta,
) -> aws_bedrock::ConverseStreamOutput {
    delta(
        index,
        aws_bedrock::ContentBlockDelta::ReasoningContent(thought),
    )
}

fn thinking(index: i32, text: &str) -> aws_bedrock::ConverseStreamOutput {
    thought(
        index,
        aws_bedrock::ReasoningContentBlockDelta::Text(text.to_owned()),
    )
}

fn signature(index: i32, signature: &str) -> aws_bedrock::ConverseStreamOutput {
    thought(
        index,
        aws_bedrock::ReasoningContentBlockDelta::Signature(signature.to_owned()),
    )
}

fn redacted(index: i32, bytes: &[u8]) -> aws_bedrock::ConverseStreamOutput {
    thought(
        index,
        aws_bedrock::ReasoningContentBlockDelta::RedactedContent(Blob::new(bytes.to_vec())),
    )
}

fn tool_start(
    index: i32,
    id: &str,
    name: &str,
    kind: Option<aws_bedrock::ToolUseType>,
) -> aws_bedrock::ConverseStreamOutput {
    start(
        index,
        aws_bedrock::ContentBlockStart::ToolUse(
            aws_bedrock::ToolUseBlockStart::builder()
                .tool_use_id(id)
                .name(name)
                .set_type(kind)
                .build()
                .expect("tool start builds"),
        ),
    )
}

fn tool_input(index: i32, input: &str) -> aws_bedrock::ConverseStreamOutput {
    delta(
        index,
        aws_bedrock::ContentBlockDelta::ToolUse(
            aws_bedrock::ToolUseBlockDelta::builder()
                .input(input)
                .build()
                .expect("tool delta builds"),
        ),
    )
}

fn stop(index: i32) -> aws_bedrock::ConverseStreamOutput {
    aws_bedrock::ConverseStreamOutput::ContentBlockStop(
        aws_bedrock::ContentBlockStopEvent::builder()
            .content_block_index(index)
            .build()
            .expect("stop builds"),
    )
}

fn message_stop(reason: &str) -> aws_bedrock::ConverseStreamOutput {
    aws_bedrock::ConverseStreamOutput::MessageStop(
        aws_bedrock::MessageStopEvent::builder()
            .stop_reason(aws_bedrock::StopReason::from(reason))
            .build()
            .expect("message stop builds"),
    )
}

fn metadata(input: i32, output: i32) -> aws_bedrock::ConverseStreamOutput {
    aws_bedrock::ConverseStreamOutput::Metadata(
        aws_bedrock::ConverseStreamMetadataEvent::builder()
            .usage(usage(input, output))
            .build(),
    )
}

/// `events` followed by an `end_turn` stop and the terminal metadata.
fn ended(
    mut events: Vec<aws_bedrock::ConverseStreamOutput>,
) -> Vec<aws_bedrock::ConverseStreamOutput> {
    events.extend([message_stop("end_turn"), metadata(3, 1)]);
    events
}

/// The events Converse streams for the whole block `block` at `index`.
#[allow(clippy::wildcard_enum_match_arm)]
fn restated_block(
    index: i32,
    block: &aws_bedrock::ContentBlock,
) -> Vec<aws_bedrock::ConverseStreamOutput> {
    use aws_bedrock::{ContentBlock as Content, ContentBlockDelta as Delta};
    let mut events = Vec::new();
    match block {
        Content::Text(body) => events.push(text(index, body)),
        Content::CitationsContent(cited) => {
            for content in cited.content.iter().flatten() {
                if let aws_bedrock::CitationGeneratedContent::Text(body) = content {
                    events.push(text(index, body));
                }
            }
            for citation in cited.citations.iter().flatten() {
                let citation = aws_bedrock::CitationsDelta::builder()
                    .set_title(citation.title.clone())
                    .set_location(citation.location.clone())
                    .build();
                events.push(delta(index, Delta::Citation(citation)));
            }
        }
        Content::ToolUse(call) => {
            events.push(tool_start(
                index,
                &call.tool_use_id,
                &call.name,
                call.r#type.clone(),
            ));
            let input = json::to_value(call.input.clone()).to_string();
            events.push(tool_input(index, &input));
        }
        Content::ReasoningContent(aws_bedrock::ReasoningContentBlock::ReasoningText(reasoning)) => {
            events.push(thinking(index, &reasoning.text));
            if let Some(sig) = &reasoning.signature {
                let (head, tail) = sig.split_at(sig.len() / 2);
                events.extend([signature(index, head), signature(index, tail)]);
            }
        }
        Content::ReasoningContent(aws_bedrock::ReasoningContentBlock::RedactedContent(blob)) => {
            let (head, tail) = blob.as_ref().split_at(blob.as_ref().len() / 2);
            events.extend([redacted(index, head), redacted(index, tail)]);
        }
        Content::Image(image) => {
            events.push(start(
                index,
                aws_bedrock::ContentBlockStart::Image(
                    aws_bedrock::ImageBlockStart::builder()
                        .format(image.format.clone())
                        .build()
                        .expect("image start builds"),
                ),
            ));
            events.push(delta(
                index,
                Delta::Image(
                    aws_bedrock::ImageBlockDelta::builder()
                        .set_source(image.source.clone())
                        .build(),
                ),
            ));
        }
        Content::ToolResult(result) => {
            events.push(start(
                index,
                aws_bedrock::ContentBlockStart::ToolResult(
                    aws_bedrock::ToolResultBlockStart::builder()
                        .tool_use_id(&result.tool_use_id)
                        .set_status(result.status.clone())
                        .set_type(result.r#type.clone())
                        .build()
                        .expect("tool result start builds"),
                ),
            ));
            let parts = result
                .content
                .iter()
                .map(|part| match part {
                    aws_bedrock::ToolResultContentBlock::Text(body) => {
                        aws_bedrock::ToolResultBlockDelta::Text(body.clone())
                    }
                    aws_bedrock::ToolResultContentBlock::Json(value) => {
                        aws_bedrock::ToolResultBlockDelta::Json(value.clone())
                    }
                    other => panic!("no stream carries {other:?}"),
                })
                .collect();
            events.push(delta(index, Delta::ToolResult(parts)));
        }
        other => panic!("{other:?} never streams"),
    }
    events.push(stop(index));
    events
}

/// `content` as the stream of events Converse sends for it.
pub(crate) fn restated(
    content: &[aws_bedrock::ContentBlock],
    stop_reason: &str,
) -> Vec<aws_bedrock::ConverseStreamOutput> {
    let mut events: Vec<_> = content
        .iter()
        .enumerate()
        .flat_map(|(index, block)| restated_block(i32::try_from(index).expect("small"), block))
        .collect();
    events.extend([message_stop(stop_reason), metadata(3, 1)]);
    events
}

/// Assert `content` decoded whole and restated as a stream fold into the
/// same turn for `model`.
pub(crate) fn assert_agrees(model: &str, content: &[aws_bedrock::ContentBlock], stop_reason: &str) {
    let opened = || ConverseFrame::Opened { request_id: None };
    let stream = std::iter::once(opened()).chain(
        restated(content, stop_reason)
            .into_iter()
            .map(ConverseFrame::Event),
    );
    assert_restated_agrees(
        &Converse::new(model),
        [
            opened(),
            ConverseFrame::Whole(Box::new(reply_of(content.to_vec(), stop_reason))),
        ],
        stream,
    );
}

pub(crate) fn reasoning_text(text: &str, signature: Option<&str>) -> aws_bedrock::ContentBlock {
    aws_bedrock::ContentBlock::ReasoningContent(aws_bedrock::ReasoningContentBlock::ReasoningText(
        aws_bedrock::ReasoningTextBlock::builder()
            .text(text)
            .set_signature(signature.map(str::to_owned))
            .build()
            .expect("reasoning builds"),
    ))
}

pub(crate) fn tool_use(
    id: &str,
    name: &str,
    input: serde_json::Value,
    kind: Option<aws_bedrock::ToolUseType>,
) -> aws_bedrock::ContentBlock {
    aws_bedrock::ContentBlock::ToolUse(
        aws_bedrock::ToolUseBlock::builder()
            .tool_use_id(id)
            .name(name)
            .input(json::to_document(input))
            .set_type(kind)
            .build()
            .expect("tool use builds"),
    )
}

const REDACTED: &[u8] = b"\x00opaque-ciphertext\xff\x01";

fn redacted_block() -> aws_bedrock::ContentBlock {
    aws_bedrock::ContentBlock::ReasoningContent(
        aws_bedrock::ReasoningContentBlock::RedactedContent(Blob::new(REDACTED.to_vec())),
    )
}

fn image_block() -> aws_bedrock::ContentBlock {
    aws_bedrock::ContentBlock::Image(
        aws_bedrock::ImageBlock::builder()
            .format(aws_bedrock::ImageFormat::Png)
            .source(aws_bedrock::ImageSource::Bytes(Blob::new(
                b"png-bytes".to_vec(),
            )))
            .build()
            .expect("image builds"),
    )
}

fn hosted_result() -> aws_bedrock::ContentBlock {
    aws_bedrock::ContentBlock::ToolResult(
        aws_bedrock::ToolResultBlock::builder()
            .tool_use_id("srv_1")
            .content(aws_bedrock::ToolResultContentBlock::Text(
                "found".to_owned(),
            ))
            .content(aws_bedrock::ToolResultContentBlock::Json(
                json::to_document(json!({ "hits": 2 })),
            ))
            .status(aws_bedrock::ToolResultStatus::Success)
            .build()
            .expect("tool result builds"),
    )
}

fn cited_block() -> aws_bedrock::ContentBlock {
    aws_bedrock::ContentBlock::CitationsContent(
        aws_bedrock::CitationsContentBlock::builder()
            .content(aws_bedrock::CitationGeneratedContent::Text(
                "The token is violet.".to_owned(),
            ))
            .citations(
                aws_bedrock::Citation::builder()
                    .title("note")
                    .location(aws_bedrock::CitationLocation::DocumentChar(
                        aws_bedrock::DocumentCharLocation::builder()
                            .document_index(0)
                            .start(0)
                            .end(20)
                            .build(),
                    ))
                    .build(),
            )
            .build(),
    )
}

fn server() -> Option<aws_bedrock::ToolUseType> {
    Some(aws_bedrock::ToolUseType::ServerToolUse)
}

/// Every block kind Converse replies with, signed and redacted reasoning
/// and a hosted tool included.
fn rich() -> Vec<aws_bedrock::ContentBlock> {
    vec![
        reasoning_text("let me think", Some("sig-abc-123")),
        redacted_block(),
        cited_block(),
        aws_bedrock::ContentBlock::Text("Calling.".to_owned()),
        image_block(),
        tool_use(
            "srv_1",
            "nova_grounding",
            json!({ "q": "violet" }),
            server(),
        ),
        hosted_result(),
        tool_use(
            "call_a",
            "get_weather",
            json!({ "location": "Paris" }),
            None,
        ),
        tool_use("call_b", "ping", json!({}), None),
    ]
}

/// Reasoning, cited text and a hosted tool's use and result keep their
/// Converse JSON; text, images and client calls keep none.
#[test]
fn kept_blocks_hold_their_converse_json() {
    let claude = crate::completion::ANTHROPIC_CLAUDE_SONNET_4_6;
    let response = unary_as(claude, reply_of(rich(), "tool_use")).expect("decodes");
    let items: Vec<_> = response
        .choice
        .iter()
        .map(|block| match block {
            AssistantContent::Opaque(opaque) => Some((opaque.item.clone(), opaque.replay)),
            block => block.native_item().cloned().map(|item| (item, true)),
        })
        .collect();
    assert_eq!(
        items,
        [
            Some((
                json!({ "reasoningContent": { "reasoningText": {
                    "text": "let me think", "signature": "sig-abc-123",
                } } }),
                true
            )),
            Some((
                json!({ "reasoningContent": {
                    "redactedContent": BASE64_STANDARD.encode(REDACTED),
                } }),
                true
            )),
            Some((
                json!({ "citationsContent": {
                    "content": [{ "text": "The token is violet." }],
                    "citations": [{
                        "title": "note",
                        "location": { "documentChar": { "documentIndex": 0, "start": 0, "end": 20 } },
                    }],
                } }),
                true
            )),
            None,
            None,
            Some((
                json!({ "toolUse": {
                    "toolUseId": "srv_1", "name": "nova_grounding",
                    "input": { "q": "violet" }, "type": "server_tool_use",
                } }),
                true
            )),
            Some((
                json!({ "toolResult": {
                    "toolUseId": "srv_1",
                    "content": [{ "text": "found" }, { "json": { "hits": 2 } }],
                    "status": "success",
                } }),
                true
            )),
            None,
            None,
        ]
    );
    assert_eq!(
        response.choice[2],
        AssistantContent::text("The token is violet.")
            .with_native(items[2].clone().expect("cited").0)
    );
    assert!(matches!(&response.choice[4], AssistantContent::Image(_)));
    assert_eq!(response.finish_reason(), Some(FinishReason::ToolCalls));
}

/// A whole reply and the same reply as a stream fold into one turn.
#[test]
fn whole_and_streamed_replies_agree() {
    assert_agrees(
        crate::completion::ANTHROPIC_CLAUDE_SONNET_4_6,
        &rich(),
        "tool_use",
    );
    assert_agrees(
        NOVA,
        &[
            aws_bedrock::ContentBlock::Text("<thinking>sum</thinking>".to_owned()),
            tool_use("tooluse_1", "add", json!({ "x": 2, "y": 5 }), None),
        ],
        "tool_use",
    );
    assert_agrees(
        NOVA,
        &[
            reasoning_text("", Some("only-signature")),
            aws_bedrock::ContentBlock::Text("4".into()),
        ],
        "max_tokens",
    );
}

/// A hosted tool's use and result are opaque items that replay to the same
/// model, whole or streamed; the agent never sees a call it should run.
#[test]
fn a_hosted_tool_use_is_not_a_client_call() {
    let content = [
        tool_use(
            "srv_1",
            "nova_grounding",
            json!({ "q": "violet" }),
            server(),
        ),
        hosted_result(),
        aws_bedrock::ContentBlock::Text("Violet.".to_owned()),
    ];
    let whole = unary_as(NOVA, reply_of(content.to_vec(), "end_turn")).expect("decodes");
    let stream = streamed(restated(&content, "end_turn")).expect("decodes");
    for response in [whole, stream] {
        assert_eq!(response.tool_calls().count(), 0, "{:?}", response.choice);
        assert!(
            matches!(
                &response.choice[..],
                [
                    AssistantContent::Opaque(Opaque { replay: true, .. }),
                    AssistantContent::Opaque(Opaque { replay: true, .. }),
                    AssistantContent::Text(_),
                ]
            ),
            "{:?}",
            response.choice
        );
        assert_eq!(response.stop(), Stop::Stop);
    }
}

/// A stream's citations stay on their text block's item, as a whole reply
/// states them.
#[test]
fn streamed_citations_are_kept() {
    assert_agrees(NOVA, &[cited_block()], "end_turn");
    let response = streamed(ended(vec![
        text(0, "The token "),
        delta(
            0,
            aws_bedrock::ContentBlockDelta::Citation(
                aws_bedrock::CitationsDelta::builder().title("note").build(),
            ),
        ),
        text(0, "is violet."),
        stop(0),
    ]))
    .expect("decodes");
    assert_eq!(
        response.choice[0].native_item(),
        Some(&json!({ "citationsContent": {
            "content": [{ "text": "The token is violet." }],
            "citations": [{ "title": "note" }],
        } }))
    );
}

/// A block is complete only at its stop: one the stream never stopped
/// keeps no provider item, and a hosted call stops replaying.
#[test]
fn a_block_keeps_its_item_only_at_its_stop() {
    let response = streamed(vec![
        thinking(0, "partial"),
        signature(0, "sig"),
        tool_start(1, "srv_1", "nova_grounding", server()),
        tool_input(1, "{}"),
        message_stop("end_turn"),
        metadata(3, 1),
    ])
    .expect("decodes");
    assert_eq!(response.choice[0], AssistantContent::reasoning("partial"));
    assert!(response.choice[0].native_item().is_none());
    assert!(
        matches!(
            &response.choice[1],
            AssistantContent::Opaque(Opaque { replay: false, .. })
        ),
        "{:?}",
        response.choice[1]
    );
}

/// Signature fragments concatenate into the reasoning's item.
#[test]
fn signed_thinking_keeps_its_signature() {
    let response = streamed(ended(vec![
        thinking(0, "I am "),
        thinking(0, "thinking"),
        signature(0, "sig-"),
        signature(0, "abc"),
        stop(0),
    ]))
    .expect("decodes");
    assert_eq!(
        response.choice,
        [AssistantContent::reasoning("I am thinking").with_native(
            json!({ "reasoningContent": { "reasoningText": { "text": "I am thinking", "signature": "sig-abc" } } })
        )]
    );
}

/// Adaptive thinking can sign a block with no text; the signature must
/// reach the next turn, so the block is kept.
#[test]
fn signature_only_thinking_is_kept() {
    let response = streamed(ended(vec![signature(0, "sig-only"), stop(0)])).expect("decodes");
    assert_eq!(
        response.choice,
        [AssistantContent::reasoning("").with_native(
            json!({ "reasoningContent": { "reasoningText": { "text": "", "signature": "sig-only" } } })
        )]
    );
}

/// A thinking block that streamed nothing is no content.
#[test]
fn empty_thinking_is_no_content() {
    let response = streamed(ended(vec![thinking(0, ""), stop(0)])).expect("decodes");
    assert!(response.choice.is_empty());
}

/// Redacted chunks are encoded once whole, so chunk boundaries that are
/// not multiples of three still give the base64 of the bytes Bedrock sent.
#[test]
fn redacted_reasoning_encodes_its_bytes_once() {
    let response = streamed(ended(vec![
        redacted(0, &REDACTED[..4]),
        redacted(0, &REDACTED[4..]),
        stop(0),
        text(1, "done"),
        stop(1),
    ]))
    .expect("decodes");
    let AssistantContent::Reasoning(reasoning) = &response.choice[0] else {
        panic!("{:?}", response.choice);
    };
    assert!(reasoning.redacted);
    assert_eq!(
        response.choice[0].native_item(),
        Some(
            &json!({ "reasoningContent": { "redactedContent": BASE64_STANDARD.encode(REDACTED) } })
        )
    );
    assert_eq!(response.choice[1], AssistantContent::text("done"));
}

#[test]
fn parallel_tool_calls_end_in_a_tool_use_finish() {
    let response = streamed(vec![
        tool_start(0, "call_a", "get_weather", None),
        tool_input(0, "{\"location\":"),
        tool_input(0, "\"Paris\"}"),
        stop(0),
        tool_start(1, "call_b", "ping", None),
        stop(1),
        message_stop("tool_use"),
        metadata(3, 1),
    ])
    .expect("decodes");
    assert_eq!(response.finish_reason(), Some(FinishReason::ToolCalls));
    let calls: Vec<_> = response
        .tool_calls()
        .map(|call| (call.id.to_string(), call.function.arguments_value()))
        .collect();
    assert_eq!(
        calls,
        [
            ("call_a".to_owned(), json!({ "location": "Paris" })),
            ("call_b".to_owned(), json!({})),
        ]
    );
}

/// A call ends at its block stop, before the reply's end.
#[tokio::test]
async fn a_call_ends_at_its_block_stop() {
    let (events, outcome) = driven(vec![
        tool_start(0, "call_a", "get_weather", None),
        tool_input(0, "{}"),
        stop(0),
    ])
    .await;
    assert!(events.iter().any(|event| matches!(
        event,
        StreamEvent::End {
            content: AssistantContent::ToolCall(_),
            ..
        }
    )));
    assert!(
        matches!(outcome, Err(ProviderError::Truncated)),
        "{outcome:?}"
    );
}

/// A stream that omits block stops still delivers every call at its end.
#[test]
fn calls_missing_a_block_stop_end_with_the_reply() {
    let response = streamed(vec![
        tool_start(0, "call_a", "get_weather", None),
        tool_input(0, "{\"location\":\"Paris\"}"),
        tool_start(1, "call_b", "get_time", None),
        tool_input(1, "{\"zone\":\"UTC\"}"),
        message_stop("tool_use"),
        metadata(3, 1),
    ])
    .expect("decodes");
    let ids: Vec<_> = response
        .tool_calls()
        .map(|call| call.id.to_string())
        .collect();
    assert_eq!(ids, ["call_a", "call_b"]);
}

/// A call whose input was cut off mid-JSON keeps what its input states.
#[test]
fn a_call_cut_off_mid_input_keeps_what_it_states() {
    let response = streamed(vec![
        tool_start(0, "call_a", "get_weather", None),
        tool_input(0, "{\"location\":\"Par"),
        message_stop("max_tokens"),
        metadata(3, 1),
    ])
    .expect("decodes");
    let calls: Vec<_> = response
        .tool_calls()
        .map(|call| call.function.arguments_value())
        .collect();
    assert_eq!(calls, [serde_json::json!({"location": "Par"})]);
    assert_eq!(response.finish_reason(), Some(FinishReason::Length));
}

/// Input that is not JSON never fails the reply: the call keeps its text.
#[test]
fn malformed_tool_json_is_kept_with_its_text() {
    let response = streamed(ended(vec![
        tool_start(0, "call_a", "get_weather", None),
        tool_input(0, "{\"location\": not-json"),
        stop(0),
    ]))
    .expect("malformed input does not fail the reply");
    let call = response.tool_calls().next().expect("the call is kept");
    assert_eq!(
        call.function.invalid_arguments.as_deref(),
        Some("{\"location\": not-json")
    );
}

/// A call that names no tool is dropped, not a failed reply.
#[test]
fn a_nameless_call_is_dropped() {
    let response = streamed(ended(vec![
        tool_start(0, "call_a", "", None),
        tool_input(0, "{}"),
        stop(0),
        text(1, "done"),
        stop(1),
    ]))
    .expect("a nameless call does not fail the reply");
    assert_eq!(response.choice, [AssistantContent::text("done")]);
}

/// Images and tool results arrive in pieces and become one block at their
/// stop; a delta for a block that never started is an error.
#[test]
fn image_and_tool_result_deltas_assemble_at_their_stop() {
    let response =
        streamed(restated(&[image_block(), hosted_result()], "end_turn")).expect("decodes");
    assert_eq!(
        response.choice[0],
        AssistantContent::image_base64(
            BASE64_STANDARD.encode(b"png-bytes"),
            Some(ImageMediaType::PNG),
            None
        )
    );
    assert!(matches!(
        &response.choice[1],
        AssistantContent::Opaque(Opaque { replay: true, .. })
    ));

    let source = aws_bedrock::ImageSource::Bytes(Blob::new(b"x".to_vec()));
    let orphan = delta(
        0,
        aws_bedrock::ContentBlockDelta::Image(
            aws_bedrock::ImageBlockDelta::builder()
                .source(source)
                .build(),
        ),
    );
    assert!(streamed(ended(vec![orphan])).is_err());
}

/// Every `stopReason` Converse documents maps explicitly, and only an end
/// of turn, a stop sequence, a tool use and a length stop are successes.
#[test]
#[allow(clippy::wildcard_enum_match_arm)]
fn every_stop_reason_maps() {
    use aws_bedrock::StopReason as Reason;
    let cases = [
        (Reason::EndTurn, FinishReason::Stop, Stop::Stop),
        (Reason::StopSequence, FinishReason::Stop, Stop::Stop),
        (Reason::ToolUse, FinishReason::ToolCalls, Stop::ToolUse),
        (Reason::MaxTokens, FinishReason::Length, Stop::Length),
        (
            Reason::ModelContextWindowExceeded,
            FinishReason::Length,
            Stop::Length,
        ),
        (
            Reason::GuardrailIntervened,
            FinishReason::ContentFilter,
            Stop::Error("Provider finish_reason: content_filter".to_owned()),
        ),
        (
            Reason::ContentFiltered,
            FinishReason::ContentFilter,
            Stop::Error("Provider finish_reason: content_filter".to_owned()),
        ),
        (
            Reason::MalformedModelOutput,
            FinishReason::Other("malformed_model_output".to_owned()),
            Stop::Error("Provider stopped with: malformed_model_output".to_owned()),
        ),
        (
            Reason::MalformedToolUse,
            FinishReason::Other("malformed_tool_use".to_owned()),
            Stop::Error("Provider stopped with: malformed_tool_use".to_owned()),
        ),
        (
            Reason::from("x_rig_invented"),
            FinishReason::Other("x_rig_invented".to_owned()),
            Stop::Error("Provider stopped with: x_rig_invented".to_owned()),
        ),
    ];
    let index = |reason: &Reason| match reason {
        Reason::ContentFiltered => 0,
        Reason::EndTurn => 1,
        Reason::GuardrailIntervened => 2,
        Reason::MalformedModelOutput => 3,
        Reason::MalformedToolUse => 4,
        Reason::MaxTokens => 5,
        Reason::ModelContextWindowExceeded => 6,
        Reason::StopSequence => 7,
        Reason::ToolUse => 8,
        _ => 9,
    };
    let reasons: Vec<Reason> = cases.iter().map(|(reason, ..)| reason.clone()).collect();
    assert_every_variant(&reasons, index, 10);
    for (reason, finish, stopped) in cases {
        let content = vec![aws_bedrock::ContentBlock::Text("x".into())];
        let whole = unary_as(NOVA, reply_of(content, reason.as_str())).expect("decodes");
        let stream = streamed(vec![
            text(0, "x"),
            stop(0),
            message_stop(reason.as_str()),
            metadata(1, 1),
        ])
        .expect("decodes");
        for response in [whole, stream] {
            assert_eq!(response.finish_reason(), Some(finish.clone()), "{reason:?}");
            assert_eq!(response.stop(), stopped, "{reason:?}");
        }
    }
}

/// An empty text block is no content, whole or streamed.
#[test]
fn an_empty_text_block_is_no_content() {
    let whole = unary_as(
        NOVA,
        reply_of(
            vec![aws_bedrock::ContentBlock::Text(String::new())],
            "end_turn",
        ),
    )
    .expect("decodes");
    assert!(whole.choice.is_empty());
    let stream = streamed(ended(vec![text(0, ""), stop(0)])).expect("decodes");
    assert!(stream.choice.is_empty());
}

/// A stream's `raw` is the JSON Bedrock sent for its message-level events:
/// the stop with its model-specific fields, and the metadata with usage,
/// metrics, trace, performance configuration and service tier.
#[test]
fn a_streams_raw_is_bedrocks_json() {
    let message_stop_json = json!({ "messageStop": {
        "stopReason": "end_turn",
        "additionalModelResponseFields": { "stop_sequence": null },
    } });
    let metadata_json = json!({ "metadata": {
        "usage": { "inputTokens": 3, "outputTokens": 1, "totalTokens": 4 },
        "metrics": { "latencyMs": 120 },
        "trace": { "promptRouter": { "invokedModelId": "amazon.nova-lite-v1:0" } },
        "performanceConfig": { "latency": "optimized" },
        "serviceTier": { "type": "priority" },
    } });
    let frames = [
        ConverseFrame::Opened { request_id: None },
        ConverseFrame::Event(text(0, "hi")),
        ConverseFrame::Event(stop(0)),
        ConverseFrame::Raw(message_stop_json.clone()),
        ConverseFrame::Event(message_stop("end_turn")),
        ConverseFrame::Raw(metadata_json.clone()),
        ConverseFrame::Event(metadata(3, 1)),
    ];
    let response = decode(&Converse::new(NOVA), Mode::Streaming, frames).expect("decodes");
    assert_eq!(
        response.raw,
        json!({
            "messageStop": message_stop_json["messageStop"],
            "metadata": metadata_json["metadata"],
        })
    );
    assert_eq!(response.usage.total_tokens, Some(4));
}

/// Rig's input counts Bedrock's cache reads and writes, as `totalTokens`
/// does.
#[test]
fn usage_counts_cache_reads_and_writes_in_input() {
    let usage = aws_bedrock::TokenUsage::builder()
        .input_tokens(200)
        .output_tokens(75)
        .total_tokens(325)
        .cache_read_input_tokens(40)
        .cache_write_input_tokens(10)
        .build()
        .expect("usage builds");
    assert_eq!(
        normalize_usage(&usage),
        rig_core::completion::Usage {
            input_tokens: Some(250),
            output_tokens: Some(75),
            total_tokens: Some(325),
            cached_input_tokens: Some(40),
            cache_creation_input_tokens: Some(10),
            tool_use_prompt_tokens: None,
            reasoning_tokens: None,
        }
    );
}

/// Opening a Converse stream sends nothing until the first poll, so a send
/// that fails is the stream's first item rather than an error from `stream`.
#[tokio::test]
async fn a_converse_stream_whose_send_fails_reports_it_in_band() {
    use aws_sdk_bedrockruntime::config::{
        BehaviorVersion, Credentials, Region, retry::RetryConfig,
    };
    let config = aws_sdk_bedrockruntime::Config::builder()
        .behavior_version(BehaviorVersion::latest())
        .region(Region::new("us-east-1"))
        .credentials_provider(Credentials::new("id", "secret", None, None, "test"))
        // Nothing listens on port 1, so the send fails without a network.
        .endpoint_url("http://127.0.0.1:1")
        .retry_config(RetryConfig::disabled())
        .build();
    let runtime =
        crate::client::BedrockRuntime::from(aws_sdk_bedrockruntime::Client::from_conf(config));
    let mut stream = Model::new(Converse::new(NOVA), runtime)
        .stream(rig_core::completion::CompletionRequest::new("hi"))
        .expect("opening a stream sends nothing");
    let first = stream.next().await.expect("the stream yields the failure");
    assert!(
        first.is_err(),
        "the failed send is the first item: {first:?}"
    );
}
