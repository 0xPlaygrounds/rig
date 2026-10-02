use super::*;
use crate::completion::{Converse, ConverseRequest};
use crate::types::assistant_content::normalize_usage;
use crate::types::converse_output::{
    CachePointBlock, CachePointType, Citation, CitationsContentBlock, ConverseOutput,
    DocumentBlock, DocumentFormat, DocumentSource, GuardrailConverseContentBlock,
    GuardrailConverseTextBlock, Message, ReasoningTextBlock, ToolUseBlock, VideoBlock, VideoFormat,
};
use futures::StreamExt;
use rig_core::completion::{CompletionResponse, FinishReason};
use rig_core::driver::{Exchange, Model, Opened, Opening, Transport};
use rig_core::message::{Opaque, StopReason as Stop};
use rig_core::streaming::{Item, StreamEvent};
use rig_core::test_utils::history::{assert_every_variant, assert_restated_agrees, decode};
use rig_core::wire::Mode;
use serde_json::json;

const NOVA: &str = "amazon.nova-lite-v1:0";

/// Replays scripted Converse events after the frame naming the model.
#[derive(Clone)]
struct Scripted(std::sync::Arc<std::sync::Mutex<Vec<aws_bedrock::ConverseStreamOutput>>>);

impl Transport<Converse> for Scripted {
    fn send(&self, payload: ConverseRequest, _exchange: Exchange) -> Opening<ConverseFrame> {
        let events = std::mem::take(&mut *self.0.lock().expect("script lock"));
        let opened = ConverseFrame::Opened {
            model: payload.model,
            request_id: None,
        };
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

/// The response `model`'s decoder folds streamed `events` into.
fn streamed_as(
    model: &str,
    events: Vec<aws_bedrock::ConverseStreamOutput>,
) -> Result<CompletionResponse, ProviderError> {
    let opened = ConverseFrame::Opened {
        model: model.to_owned(),
        request_id: None,
    };
    let frames = std::iter::once(opened).chain(events.into_iter().map(ConverseFrame::Event));
    decode(&Converse::new(model), Mode::Streaming, frames)
}

fn streamed(
    events: Vec<aws_bedrock::ConverseStreamOutput>,
) -> Result<CompletionResponse, ProviderError> {
    streamed_as(NOVA, events)
}

/// The response the decoder folds the whole reply `output` into.
pub(crate) fn unary_as(
    model: &str,
    output: InternalConverseOutput,
) -> Result<CompletionResponse, ProviderError> {
    let frames = [
        ConverseFrame::Opened {
            model: model.to_owned(),
            request_id: None,
        },
        ConverseFrame::Whole(Box::new(output)),
    ];
    decode(&Converse::new(model), Mode::Unary, frames)
}

/// A whole assistant reply holding `content`.
pub(crate) fn reply_of(
    content: Vec<ContentBlock>,
    stop_reason: StopReason,
) -> InternalConverseOutput {
    InternalConverseOutput {
        output: Some(ConverseOutput::Message(Message {
            role: ConversationRole::Assistant,
            content,
        })),
        stop_reason,
        usage: Some(TokenUsage {
            input_tokens: 3,
            output_tokens: 1,
            total_tokens: 4,
            cache_read_input_tokens: None,
            cache_write_input_tokens: None,
        }),
        metrics: None,
        additional_model_response_fields: None,
        request_id: None,
        trace: None,
        performance_config: None,
        service_tier: None,
    }
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
        aws_bedrock::ReasoningContentBlockDelta::RedactedContent(aws_smithy_types::Blob::new(
            bytes.to_vec(),
        )),
    )
}

fn tool_start(index: i32, id: &str, name: &str) -> aws_bedrock::ConverseStreamOutput {
    start(
        index,
        aws_bedrock::ContentBlockStart::ToolUse(
            aws_bedrock::ToolUseBlockStart::builder()
                .tool_use_id(id)
                .name(name)
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
            .usage(
                aws_bedrock::TokenUsage::builder()
                    .input_tokens(input)
                    .output_tokens(output)
                    .total_tokens(input + output)
                    .build()
                    .expect("usage builds"),
            )
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

/// `output` as the stream of events Converse sends for it. Documents,
/// videos, cache points and guard content only ever arrive whole.
pub(crate) fn restated(output: &InternalConverseOutput) -> Vec<aws_bedrock::ConverseStreamOutput> {
    use aws_bedrock::ContentBlockDelta as Delta;
    let mut events = Vec::new();
    let content = match &output.output {
        Some(ConverseOutput::Message(message)) => message.content.as_slice(),
        _ => &[],
    };
    for (index, block) in content.iter().enumerate() {
        let index = i32::try_from(index).expect("small reply");
        match block {
            ContentBlock::Text(body) => events.push(text(index, body)),
            ContentBlock::CitationsContent(cited) => {
                for content in cited.content.iter().flatten() {
                    if let CitationGeneratedContent::Text(body) = content {
                        events.push(text(index, body));
                    }
                }
                for citation in cited.citations.iter().flatten() {
                    let citation = aws_bedrock::CitationsDelta::builder()
                        .set_title(citation.title.clone())
                        .build();
                    events.push(delta(index, Delta::Citation(citation)));
                }
            }
            ContentBlock::ToolUse(call) => {
                events.push(tool_start(index, &call.tool_use_id, &call.name));
                events.push(tool_input(index, &call.input.to_string()));
            }
            ContentBlock::ReasoningContent(ReasoningContentBlock::ReasoningText(reasoning)) => {
                events.push(thinking(index, &reasoning.text));
                if let Some(sig) = &reasoning.signature {
                    let (head, tail) = sig.split_at(sig.len() / 2);
                    events.extend([signature(index, head), signature(index, tail)]);
                }
            }
            ContentBlock::ReasoningContent(ReasoningContentBlock::RedactedContent(blob)) => {
                let (head, tail) = blob.inner.split_at(blob.inner.len() / 2);
                events.extend([redacted(index, head), redacted(index, tail)]);
            }
            ContentBlock::Image(image) => {
                let format = match image.format {
                    ImageFormat::Gif => "gif",
                    ImageFormat::Jpeg => "jpeg",
                    ImageFormat::Png => "png",
                    ImageFormat::Webp => "webp",
                    ImageFormat::Unknown(_) => "tiff",
                };
                events.push(start(
                    index,
                    aws_bedrock::ContentBlockStart::Image(
                        aws_bedrock::ImageBlockStart::builder()
                            .format(aws_bedrock::ImageFormat::from(format))
                            .build()
                            .expect("image start builds"),
                    ),
                ));
                if let Some(ImageSource::Bytes(blob)) = &image.source {
                    let source = aws_bedrock::ImageSource::Bytes(aws_smithy_types::Blob::new(
                        blob.inner.clone(),
                    ));
                    events.push(delta(
                        index,
                        Delta::Image(
                            aws_bedrock::ImageBlockDelta::builder()
                                .source(source)
                                .build(),
                        ),
                    ));
                }
            }
            ContentBlock::ToolResult(result) => {
                events.push(start(
                    index,
                    aws_bedrock::ContentBlockStart::ToolResult(
                        aws_bedrock::ToolResultBlockStart::builder()
                            .tool_use_id(&result.tool_use_id)
                            .build()
                            .expect("tool result start builds"),
                    ),
                ));
                let parts = result
                    .content
                    .iter()
                    .filter_map(|part| match part {
                        ToolResultContentBlock::Text(body) => {
                            Some(aws_bedrock::ToolResultBlockDelta::Text(body.clone()))
                        }
                        ToolResultContentBlock::Json(value) => {
                            Some(aws_bedrock::ToolResultBlockDelta::Json(json::to_document(
                                value.clone(),
                            )))
                        }
                        _ => None,
                    })
                    .collect();
                events.push(delta(index, Delta::ToolResult(parts)));
            }
            ContentBlock::ReasoningContent(ReasoningContentBlock::Unknown)
            | ContentBlock::Unknown => continue,
            ContentBlock::CachePoint(_)
            | ContentBlock::Document(_)
            | ContentBlock::GuardContent(_)
            | ContentBlock::Video(_) => panic!("{block:?} never streams"),
        }
        events.push(stop(index));
    }
    events.push(message_stop(output.stop_reason.as_str()));
    let usage = output.usage.as_ref().map(|usage| {
        aws_bedrock::TokenUsage::builder()
            .input_tokens(usage.input_tokens)
            .output_tokens(usage.output_tokens)
            .total_tokens(usage.total_tokens)
            .set_cache_read_input_tokens(usage.cache_read_input_tokens)
            .set_cache_write_input_tokens(usage.cache_write_input_tokens)
            .build()
            .expect("usage builds")
    });
    events.push(aws_bedrock::ConverseStreamOutput::Metadata(
        aws_bedrock::ConverseStreamMetadataEvent::builder()
            .set_usage(usage)
            .build(),
    ));
    events
}

/// Assert `output` decoded whole and restated as a stream fold into the
/// same turn for `model`.
pub(crate) fn assert_agrees(model: &str, output: &InternalConverseOutput) {
    let opened = || ConverseFrame::Opened {
        model: model.to_owned(),
        request_id: None,
    };
    let stream =
        std::iter::once(opened()).chain(restated(output).into_iter().map(ConverseFrame::Event));
    assert_restated_agrees(
        &Converse::new(model),
        [opened(), ConverseFrame::Whole(Box::new(output.clone()))],
        stream,
    );
}

fn reasoning_text(text: &str, signature: Option<&str>) -> ContentBlock {
    ContentBlock::ReasoningContent(ReasoningContentBlock::ReasoningText(ReasoningTextBlock {
        text: text.to_owned(),
        signature: signature.map(str::to_owned),
    }))
}

fn tool_use(id: &str, name: &str, input: serde_json::Value) -> ContentBlock {
    ContentBlock::ToolUse(ToolUseBlock {
        tool_use_id: id.to_owned(),
        name: name.to_owned(),
        input,
    })
}

const REDACTED: &[u8] = b"\x00opaque-ciphertext\xff\x01";

fn redacted_block() -> ContentBlock {
    ContentBlock::ReasoningContent(ReasoningContentBlock::RedactedContent(
        crate::types::converse_output::Blob {
            inner: REDACTED.to_vec(),
        },
    ))
}

fn image_block() -> ContentBlock {
    ContentBlock::Image(ImageBlock {
        format: ImageFormat::Png,
        source: Some(ImageSource::Bytes(crate::types::converse_output::Blob {
            inner: b"png-bytes".to_vec(),
        })),
    })
}

fn tool_result_block() -> ContentBlock {
    ContentBlock::ToolResult(ToolResultBlock {
        tool_use_id: "server_1".to_owned(),
        content: vec![
            ToolResultContentBlock::Text("found".to_owned()),
            ToolResultContentBlock::Json(json!({ "hits": 2 })),
        ],
        status: None,
    })
}

fn cited_block() -> ContentBlock {
    ContentBlock::CitationsContent(CitationsContentBlock {
        content: Some(vec![
            CitationGeneratedContent::Text("The token ".to_owned()),
            CitationGeneratedContent::Text("is violet.".to_owned()),
        ]),
        citations: Some(vec![Citation {
            title: Some("note".to_owned()),
            source_content: None,
            location: None,
        }]),
    })
}

/// A Claude reply with every streamable item kind, signed and redacted
/// reasoning included.
fn rich_reply() -> InternalConverseOutput {
    reply_of(
        vec![
            reasoning_text("let me think", Some("sig-abc-123")),
            redacted_block(),
            cited_block(),
            ContentBlock::Text("Calling.".to_owned()),
            image_block(),
            tool_result_block(),
            tool_use("call_a", "get_weather", json!({ "location": "Paris" })),
            tool_use("call_b", "ping", json!({})),
        ],
        StopReason::ToolUse,
    )
}

/// Only reasoning keeps a provider item: its signature, or its redacted
/// bytes as base64. Everything else is canonical.
#[test]
fn only_reasoning_keeps_a_provider_item() {
    let claude = crate::completion::ANTHROPIC_CLAUDE_SONNET_4_6;
    let response = unary_as(claude, rich_reply()).expect("decodes");
    let natives: Vec<_> = response
        .choice
        .iter()
        .map(|block| block.native_item().cloned())
        .collect();
    assert_eq!(
        natives,
        [
            Some(json!({ "signature": "sig-abc-123" })),
            Some(json!({ "redacted": BASE64_STANDARD.encode(REDACTED) })),
            None,
            None,
            None,
            None,
            None,
            None,
        ]
    );
    let AssistantContent::Reasoning(redacted) = &response.choice[1] else {
        panic!("{:?}", response.choice[1]);
    };
    assert!(redacted.redacted && redacted.text.is_empty());
    assert_eq!(
        response.choice[2],
        AssistantContent::text("The token is violet.")
    );
    assert!(matches!(&response.choice[4], AssistantContent::Image(_)));
    assert_eq!(
        response.choice[5],
        AssistantContent::Opaque(Opaque {
            item: serde_json::to_value(tool_result_block()).expect("serializes"),
            replay: false,
        })
    );
    assert_eq!(response.finish_reason(), Some(FinishReason::ToolCalls));
}

/// A whole reply and the same reply as a stream fold into one turn.
#[test]
fn whole_and_streamed_replies_agree() {
    assert_agrees(
        crate::completion::ANTHROPIC_CLAUDE_SONNET_4_6,
        &rich_reply(),
    );
    assert_agrees(
        NOVA,
        &reply_of(
            vec![
                ContentBlock::Text("<thinking>sum</thinking>".to_owned()),
                tool_use("tooluse_1", "add", json!({ "x": 2, "y": 5 })),
            ],
            StopReason::ToolUse,
        ),
    );
    assert_agrees(
        NOVA,
        &reply_of(
            vec![
                reasoning_text("", Some("only-signature")),
                ContentBlock::Text("4".into()),
            ],
            StopReason::MaxTokens,
        ),
    );
}

/// Every Converse content block decodes without failing the reply; the
/// ones a stream can carry agree whole and streamed. Unknown has no payload
/// and is dropped.
#[test]
#[deny(clippy::wildcard_enum_match_arm)]
fn every_content_block_decodes() {
    let samples = vec![
        ContentBlock::CachePoint(CachePointBlock {
            kind: CachePointType::Default,
        }),
        cited_block(),
        ContentBlock::Document(DocumentBlock {
            format: DocumentFormat::Txt,
            name: "doc".to_owned(),
            source: Some(DocumentSource::Text("body".to_owned())),
            context: None,
            citations: None,
        }),
        ContentBlock::GuardContent(GuardrailConverseContentBlock::Text(
            GuardrailConverseTextBlock {
                text: "guarded".to_owned(),
                qualifiers: None,
            },
        )),
        image_block(),
        reasoning_text("thinking", Some("sig")),
        ContentBlock::Text("hello".to_owned()),
        tool_result_block(),
        tool_use("call_1", "lookup", json!({ "q": 1 })),
        ContentBlock::Video(VideoBlock {
            format: VideoFormat::Mp4,
            source: None,
        }),
        ContentBlock::Unknown,
    ];
    let variant_index = |block: &ContentBlock| match block {
        ContentBlock::CachePoint(_) => 0,
        ContentBlock::CitationsContent(_) => 1,
        ContentBlock::Document(_) => 2,
        ContentBlock::GuardContent(_) => 3,
        ContentBlock::Image(_) => 4,
        ContentBlock::ReasoningContent(_) => 5,
        ContentBlock::Text(_) => 6,
        ContentBlock::ToolResult(_) => 7,
        ContentBlock::ToolUse(_) => 8,
        ContentBlock::Video(_) => 9,
        ContentBlock::Unknown => 10,
    };
    assert_every_variant(&samples, variant_index, 11);
    for block in samples {
        let whole_only = matches!(
            block,
            ContentBlock::CachePoint(_)
                | ContentBlock::Document(_)
                | ContentBlock::GuardContent(_)
                | ContentBlock::Video(_)
        );
        let unknown = block == ContentBlock::Unknown;
        let output = reply_of(vec![block], StopReason::EndTurn);
        let response = unary_as(NOVA, output.clone()).expect("every block decodes");
        if unknown {
            assert!(response.choice.is_empty());
        } else if whole_only {
            assert!(
                matches!(
                    &response.choice[..],
                    [AssistantContent::Opaque(Opaque { replay: false, .. })]
                ),
                "{:?}",
                response.choice
            );
        } else {
            assert_agrees(NOVA, &output);
        }
    }
    let reasoning = [
        reasoning_text("x", None),
        redacted_block(),
        ContentBlock::ReasoningContent(ReasoningContentBlock::Unknown),
    ];
    let reasoning_index = |block: &ContentBlock| match block {
        ContentBlock::ReasoningContent(ReasoningContentBlock::ReasoningText(_)) => 0,
        ContentBlock::ReasoningContent(ReasoningContentBlock::RedactedContent(_)) => 1,
        ContentBlock::ReasoningContent(ReasoningContentBlock::Unknown) => 2,
        ContentBlock::CachePoint(_)
        | ContentBlock::CitationsContent(_)
        | ContentBlock::Document(_)
        | ContentBlock::GuardContent(_)
        | ContentBlock::Image(_)
        | ContentBlock::Text(_)
        | ContentBlock::ToolResult(_)
        | ContentBlock::ToolUse(_)
        | ContentBlock::Video(_)
        | ContentBlock::Unknown => 3,
    };
    assert_every_variant(&reasoning, reasoning_index, 3);
    for block in reasoning {
        let output = reply_of(vec![block], StopReason::EndTurn);
        assert_agrees(NOVA, &output);
    }
}

/// An item and a field this SDK version does not model are gone before the
/// decoder sees them: a field the mirror does not name is ignored, and the
/// SDK's `Unknown` carries nothing to keep, so the reply still decodes.
#[test]
fn an_invented_item_and_field_do_not_fail_the_reply() {
    let mut raw = serde_json::to_value(reply_of(
        vec![
            ContentBlock::Unknown,
            tool_use("call_1", "lookup", json!({})),
        ],
        StopReason::ToolUse,
    ))
    .expect("serializes");
    raw["output"]["Message"]["content"][1]["ToolUse"]["invented_field"] = json!(true);
    let output: InternalConverseOutput = serde_json::from_value(raw).expect("deserializes");
    let response = unary_as(NOVA, output).expect("decodes");
    assert_eq!(response.tool_calls().count(), 1);
    assert_eq!(response.choice.len(), 1);
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
        [AssistantContent::reasoning("I am thinking")
            .with_native(json!({ "signature": "sig-abc" }))]
    );
}

/// Adaptive thinking can sign a block with no text; the signature must
/// reach the next turn, so the block is kept.
#[test]
fn signature_only_thinking_is_kept() {
    let response = streamed(ended(vec![signature(0, "sig-only"), stop(0)])).expect("decodes");
    assert_eq!(
        response.choice,
        [AssistantContent::reasoning("").with_native(json!({ "signature": "sig-only" }))]
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
        Some(&json!({ "redacted": BASE64_STANDARD.encode(REDACTED) }))
    );
    assert_eq!(response.choice[1], AssistantContent::text("done"));
}

/// Citations are not kept; their text is.
#[test]
fn cited_text_streams_as_text() {
    let citation = aws_bedrock::CitationsDelta::builder().title("note").build();
    let response = streamed(ended(vec![
        text(0, "The token "),
        delta(0, aws_bedrock::ContentBlockDelta::Citation(citation)),
        text(0, "is violet."),
        stop(0),
    ]))
    .expect("decodes");
    assert_eq!(
        response.choice,
        [AssistantContent::text("The token is violet.")]
    );
}

#[test]
fn parallel_tool_calls_end_in_a_tool_use_finish() {
    let response = streamed(vec![
        tool_start(0, "call_a", "get_weather"),
        tool_input(0, "{\"location\":"),
        tool_input(0, "\"Paris\"}"),
        stop(0),
        tool_start(1, "call_b", "ping"),
        stop(1),
        message_stop("tool_use"),
        metadata(3, 1),
    ])
    .expect("decodes");
    assert_eq!(response.finish_reason(), Some(FinishReason::ToolCalls));
    let calls: Vec<_> = response
        .tool_calls()
        .map(|call| (call.id.to_string(), call.function.arguments.clone()))
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
        tool_start(0, "call_a", "get_weather"),
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
        tool_start(0, "call_a", "get_weather"),
        tool_input(0, "{\"location\":\"Paris\"}"),
        tool_start(1, "call_b", "get_time"),
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

/// A call whose input was cut off mid-JSON never arrives.
#[test]
fn a_call_cut_off_mid_input_is_dropped() {
    let response = streamed(vec![
        tool_start(0, "call_a", "get_weather"),
        tool_input(0, "{\"location\":\"Par"),
        message_stop("max_tokens"),
        metadata(3, 1),
    ])
    .expect("decodes");
    assert_eq!(response.tool_calls().count(), 0);
    assert_eq!(response.finish_reason(), Some(FinishReason::Length));
}

/// Input the provider said was complete must parse.
#[test]
fn malformed_tool_json_fails_the_reply() {
    let error = streamed(ended(vec![
        tool_start(0, "call_a", "get_weather"),
        tool_input(0, "{\"location\": not-json"),
        stop(0),
    ]))
    .expect_err("malformed input fails");
    assert_eq!(error.kind(), rig_core::error::ErrorKind::Response);
    assert!(error.to_string().contains("get_weather"), "{error}");
}

/// Images and tool results arrive in pieces and become one block at their
/// stop; a delta for a block that never started is an error.
#[test]
fn image_and_tool_result_deltas_assemble_at_their_stop() {
    let output = reply_of(
        vec![image_block(), tool_result_block()],
        StopReason::EndTurn,
    );
    let response = streamed(restated(&output)).expect("decodes");
    assert_eq!(
        response.choice[0],
        AssistantContent::image_base64(
            BASE64_STANDARD.encode(b"png-bytes"),
            Some(ImageMediaType::PNG),
            None
        )
    );
    assert!(matches!(&response.choice[1], AssistantContent::Opaque(_)));

    let source = aws_bedrock::ImageSource::Bytes(aws_smithy_types::Blob::new(b"x".to_vec()));
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

/// pi's stop mapping: an exceeded context window is a length stop, and a
/// malformed output or a reason this crate does not know fails the turn.
#[test]
fn newer_stop_reasons_map_and_fail_the_turn_when_malformed() {
    let stopped = |reason: &str| {
        let response = streamed(vec![
            text(0, "x"),
            stop(0),
            message_stop(reason),
            metadata(1, 1),
        ])
        .expect("decodes");
        (response.finish_reason(), response.stop())
    };
    assert_eq!(
        stopped("model_context_window_exceeded"),
        (Some(FinishReason::Length), Stop::Length)
    );
    assert_eq!(
        stopped("malformed_tool_use"),
        (
            Some(FinishReason::Other("malformed_tool_use".to_owned())),
            Stop::Error("Provider stopped with: malformed_tool_use".to_owned())
        )
    );
    assert_eq!(
        stopped("pause_turn").1,
        Stop::Error("Provider stopped with: pause_turn".to_owned())
    );
    let whole = unary_as(
        NOVA,
        reply_of(
            vec![ContentBlock::Text("x".into())],
            StopReason::MalformedModelOutput,
        ),
    )
    .expect("decodes");
    assert_eq!(
        whole.stop(),
        Stop::Error("Provider stopped with: malformed_model_output".to_owned())
    );
}

/// An empty text block is no content, whole or streamed.
#[test]
fn an_empty_text_block_is_no_content() {
    let whole = unary_as(
        NOVA,
        reply_of(vec![ContentBlock::Text(String::new())], StopReason::EndTurn),
    )
    .expect("decodes");
    assert!(whole.choice.is_empty());
    let stream = streamed(ended(vec![text(0, ""), stop(0)])).expect("decodes");
    assert!(stream.choice.is_empty());
}

/// A stream's `raw` is Bedrock's own terminal record, and its usage counts
/// cache reads and writes in input as Bedrock's `totalTokens` does.
#[test]
fn a_streams_raw_round_trips_into_the_terminal_type() {
    let response = streamed(ended(vec![text(0, "hi"), stop(0)])).expect("decodes");
    let typed: BedrockStreamingResponse =
        serde_json::from_value(response.raw.clone()).expect("raw deserializes");
    assert_eq!(
        serde_json::to_value(&typed).expect("serializes"),
        response.raw
    );
    assert_eq!(typed.stop_reason, Some(StopReason::EndTurn));
    assert_eq!(response.usage.total_tokens, Some(4));

    let usage = TokenUsage {
        input_tokens: 200,
        output_tokens: 75,
        total_tokens: 325,
        cache_read_input_tokens: Some(40),
        cache_write_input_tokens: Some(10),
    };
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
