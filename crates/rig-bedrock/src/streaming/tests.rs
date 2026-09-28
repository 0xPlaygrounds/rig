use super::*;
use crate::completion::{Converse, ConverseRequest};
use futures::StreamExt;
use rig_core::completion::CompletionResponse;
use rig_core::driver::{Exchange, Model, Opened, Opening, Transport};
use rig_core::message::{AssistantContent, Reasoning, ToolCall};
use rig_core::streaming::{Item, StreamEvent};

// ---- Event-seam helpers: a transport that replays scripted events ----

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

/// The stream `model`'s Converse endpoint yields for scripted `events`.
fn stream_of(
    model: &str,
    events: Vec<aws_bedrock::ConverseStreamOutput>,
) -> rig_core::streaming::Streamed<rig_core::operation::Completion> {
    let request = rig_core::completion::CompletionRequest::new("hi");
    Model::new(
        Converse::new(model),
        Scripted(std::sync::Arc::new(std::sync::Mutex::new(events))),
    )
    .stream(request)
    .expect("the stream opens")
}

/// What one scripted reply yields: its events, then the response or the
/// error that ended it.
struct Replied {
    events: Vec<StreamEvent>,
    outcome: Result<CompletionResponse, ProviderError>,
}

impl Replied {
    fn response(&self) -> &CompletionResponse {
        self.outcome.as_ref().expect("the reply finishes")
    }

    fn reasoning(&self) -> Vec<ReasoningContent> {
        self.response()
            .choice
            .iter()
            .filter_map(|part| match part {
                AssistantContent::Reasoning(reasoning) => {
                    reasoning.open(reasoning.issuer()).cloned()
                }
                _ => None,
            })
            .flat_map(|reasoning: Reasoning| reasoning.content)
            .collect()
    }

    fn calls(&self) -> Vec<ToolCall> {
        self.response().tool_calls().cloned().collect()
    }
}

async fn reply(events: Vec<aws_bedrock::ConverseStreamOutput>) -> Replied {
    let mut stream = stream_of("amazon.nova-lite-v1:0", events);
    let mut seen = Vec::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(Item::Event(event)) => seen.push(event),
            Ok(Item::Unknown(_)) => {}
            Err(_) => break,
        }
    }
    Replied {
        events: seen,
        outcome: stream.finish().await,
    }
}

fn reasoning_text_delta(index: i32, text: &str) -> aws_bedrock::ConverseStreamOutput {
    aws_bedrock::ConverseStreamOutput::ContentBlockDelta(
        aws_bedrock::ContentBlockDeltaEvent::builder()
            .content_block_index(index)
            .delta(aws_bedrock::ContentBlockDelta::ReasoningContent(
                aws_bedrock::ReasoningContentBlockDelta::Text(text.to_string()),
            ))
            .build()
            .expect("reasoning text delta should build"),
    )
}

fn reasoning_signature_delta(index: i32, signature: &str) -> aws_bedrock::ConverseStreamOutput {
    aws_bedrock::ConverseStreamOutput::ContentBlockDelta(
        aws_bedrock::ContentBlockDeltaEvent::builder()
            .content_block_index(index)
            .delta(aws_bedrock::ContentBlockDelta::ReasoningContent(
                aws_bedrock::ReasoningContentBlockDelta::Signature(signature.to_string()),
            ))
            .build()
            .expect("reasoning signature delta should build"),
    )
}

fn reasoning_redacted_delta(index: i32, blob: &[u8]) -> aws_bedrock::ConverseStreamOutput {
    aws_bedrock::ConverseStreamOutput::ContentBlockDelta(
        aws_bedrock::ContentBlockDeltaEvent::builder()
            .content_block_index(index)
            .delta(aws_bedrock::ContentBlockDelta::ReasoningContent(
                aws_bedrock::ReasoningContentBlockDelta::RedactedContent(
                    aws_smithy_types::Blob::new(blob.to_vec()),
                ),
            ))
            .build()
            .expect("redacted reasoning delta should build"),
    )
}

fn block_stop(index: i32) -> aws_bedrock::ConverseStreamOutput {
    aws_bedrock::ConverseStreamOutput::ContentBlockStop(
        aws_bedrock::ContentBlockStopEvent::builder()
            .content_block_index(index)
            .build()
            .expect("content block stop should build"),
    )
}

fn terminal() -> Vec<aws_bedrock::ConverseStreamOutput> {
    vec![
        aws_bedrock::ConverseStreamOutput::MessageStop(
            aws_bedrock::MessageStopEvent::builder()
                .stop_reason(aws_bedrock::StopReason::EndTurn)
                .build()
                .expect("message stop should build"),
        ),
        aws_bedrock::ConverseStreamOutput::Metadata(
            aws_bedrock::ConverseStreamMetadataEvent::builder().build(),
        ),
    ]
}

/// Ordinary extended-thinking shape: thinking deltas, the block's close at
/// `contentBlockStop`, then visible text.
#[tokio::test]
async fn thinking_then_text_streams_through_the_driver() {
    let mut events = vec![
        reasoning_text_delta(0, "let me think"),
        block_stop(0),
        text_delta_event(1, "the answer"),
        block_stop_event(1),
    ];
    events.extend(terminal());
    let replied = reply(events).await;
    assert_eq!(replied.response().text(), "the answer");
    assert_eq!(
        replied.reasoning(),
        vec![ReasoningContent::Text {
            text: "let me think".to_string(),
            signature: None,
        }],
        "an unsigned block closes carrying just its accumulated text"
    );
}

const REDACTED_BLOB: &[u8] = b"\x00opaque-stream-ciphertext\xff";

/// #2258 F2(a): the redacted delta used to hit `_ => {}` and vanish.
#[tokio::test]
async fn redacted_reasoning_delta_reaches_the_consumer() {
    let mut events = vec![reasoning_redacted_delta(0, REDACTED_BLOB), block_stop(0)];
    events.extend(terminal());

    let replied = reply(events).await;

    assert_eq!(
        replied.reasoning(),
        vec![ReasoningContent::Redacted {
            data: BASE64_STANDARD.encode(REDACTED_BLOB),
        }]
    );
}

/// The redacted block must land BESIDE an open thinking block, not replace
/// it: both come from one Converse block.
#[tokio::test]
async fn redacted_reasoning_is_a_sibling_of_the_open_thinking_block() {
    let mut events = vec![
        reasoning_text_delta(0, "visible thinking"),
        reasoning_signature_delta(0, "sig_1"),
        reasoning_redacted_delta(0, REDACTED_BLOB),
        block_stop(0),
    ];
    events.extend(terminal());

    let replied = reply(events).await;

    assert_eq!(
        replied.reasoning(),
        vec![
            ReasoningContent::Text {
                text: "visible thinking".to_string(),
                signature: Some("sig_1".to_string()),
            },
            ReasoningContent::Redacted {
                data: BASE64_STANDARD.encode(REDACTED_BLOB),
            },
        ]
    );
}

/// #2258 H5: a non-`ToolUse` `ContentBlockStart` used to fail the whole
/// stream with `ProviderError("Stream is empty")`.
#[tokio::test]
async fn non_tool_use_content_block_start_is_skipped_not_failed() {
    let mut events = vec![
        aws_bedrock::ConverseStreamOutput::ContentBlockStart(
            aws_bedrock::ContentBlockStartEvent::builder()
                .content_block_index(0)
                .start(aws_bedrock::ContentBlockStart::ToolResult(
                    aws_bedrock::ToolResultBlockStart::builder()
                        .tool_use_id("tool_1")
                        .build()
                        .expect("tool result start should build"),
                ))
                .build()
                .expect("content block start should build"),
        ),
        block_stop(0),
    ];
    events.extend(terminal());

    let replied = reply(events).await;

    assert!(
        replied.outcome.is_ok(),
        "an unmodeled ContentBlockStart must not fail the stream: {:?}",
        replied.outcome
    );
}

#[test]
fn test_bedrock_usage_creation() {
    let usage = TokenUsage {
        input_tokens: 100,
        output_tokens: 50,
        total_tokens: 150,
        cache_read_input_tokens: None,
        cache_write_input_tokens: None,
    };

    assert_eq!(usage.input_tokens, 100);
    assert_eq!(usage.output_tokens, 50);
    assert_eq!(usage.total_tokens, 150);
}

#[test]
fn test_bedrock_streaming_response_with_usage() {
    let response = BedrockStreamingResponse {
        usage: Some(TokenUsage {
            input_tokens: 200,
            output_tokens: 75,
            total_tokens: 275,
            cache_read_input_tokens: Some(40),
            cache_write_input_tokens: Some(10),
        }),
        stop_reason: None,
        provider_request_id: None,
    };

    assert_eq!(
        rig_core::completion::Usage::from(&response),
        rig_core::completion::Usage {
            input_tokens: Some(200),
            output_tokens: Some(75),
            total_tokens: Some(275),
            cached_input_tokens: Some(40),
            cache_creation_input_tokens: Some(10),
            tool_use_prompt_tokens: None,
            reasoning_tokens: None,
        }
    );
}

#[test]
fn test_bedrock_streaming_response_without_usage() {
    let response = BedrockStreamingResponse {
        usage: None,
        stop_reason: None,
        provider_request_id: None,
    };

    // No wire usage means no reported counter.
    assert!(!rig_core::completion::Usage::from(&response).is_reported());
}

#[test]
fn test_streaming_response_normalizes_usage() {
    let response = BedrockStreamingResponse {
        usage: Some(TokenUsage {
            input_tokens: 448,
            output_tokens: 68,
            total_tokens: 516,
            cache_read_input_tokens: Some(80),
            cache_write_input_tokens: Some(20),
        }),
        stop_reason: None,
        provider_request_id: None,
    };

    // The streaming response normalizes into rig's usage record.
    assert_eq!(
        rig_core::completion::Usage::from(&response),
        rig_core::completion::Usage {
            input_tokens: Some(448),
            output_tokens: Some(68),
            total_tokens: Some(516),
            cached_input_tokens: Some(80),
            cache_creation_input_tokens: Some(20),
            tool_use_prompt_tokens: None,
            reasoning_tokens: None,
        }
    );
}

#[test]
fn test_bedrock_usage_serde() {
    let usage = TokenUsage {
        input_tokens: 100,
        output_tokens: 50,
        total_tokens: 150,
        cache_read_input_tokens: Some(25),
        cache_write_input_tokens: Some(5),
    };

    // Test serialization
    let json = serde_json::to_string(&usage).expect("Should serialize");
    assert!(json.contains("\"input_tokens\":100"));
    assert!(json.contains("\"output_tokens\":50"));
    assert!(json.contains("\"total_tokens\":150"));

    // Test deserialization
    let deserialized: TokenUsage = serde_json::from_str(&json).expect("Should deserialize");
    assert_eq!(deserialized.input_tokens, usage.input_tokens);
    assert_eq!(deserialized.output_tokens, usage.output_tokens);
    assert_eq!(deserialized.total_tokens, usage.total_tokens);
    assert_eq!(
        deserialized.cache_read_input_tokens,
        usage.cache_read_input_tokens
    );
    assert_eq!(
        deserialized.cache_write_input_tokens,
        usage.cache_write_input_tokens
    );
}

#[test]
fn test_bedrock_streaming_response_serde() {
    let response = BedrockStreamingResponse {
        usage: Some(TokenUsage {
            input_tokens: 200,
            output_tokens: 75,
            total_tokens: 275,
            cache_read_input_tokens: Some(30),
            cache_write_input_tokens: Some(15),
        }),
        stop_reason: None,
        provider_request_id: None,
    };

    // Test serialization
    let json = serde_json::to_string(&response).expect("Should serialize");
    assert!(json.contains("\"input_tokens\":200"));

    // Test deserialization
    let deserialized: BedrockStreamingResponse =
        serde_json::from_str(&json).expect("Should deserialize");
    assert!(deserialized.usage.is_some());
    let usage = deserialized.usage.unwrap();
    assert_eq!(usage.input_tokens, 200);
    assert_eq!(usage.output_tokens, 75);
    assert_eq!(usage.total_tokens, 275);
    assert_eq!(usage.cache_read_input_tokens, Some(30));
    assert_eq!(usage.cache_write_input_tokens, Some(15));
}

/// A signed thinking block closes with its signature attached to the
/// text assembled from the deltas — the exact shape the next turn must
/// replay to Bedrock.
#[tokio::test]
async fn signed_thinking_block_closes_with_its_signature() {
    let mut events = vec![
        reasoning_text_delta(0, "I am "),
        reasoning_text_delta(0, "thinking"),
        reasoning_signature_delta(0, "sig-abc"),
        block_stop(0),
    ];
    events.extend(terminal());

    let replied = reply(events).await;

    assert_eq!(
        replied.reasoning(),
        vec![ReasoningContent::Text {
            text: "I am thinking".to_string(),
            signature: Some("sig-abc".to_string()),
        }]
    );
}

/// Adaptive thinking on Bedrock can produce a `Signature` delta with no
/// non-empty `Text` delta. The signature is replay-required provider
/// state, so a signature-only block must still reach the consumer —
/// dropping it fails the next turn with
/// `messages.N.content.0.thinking.signature: Field required`.
#[tokio::test]
async fn signature_only_thinking_block_still_reaches_the_consumer() {
    let mut events = vec![reasoning_signature_delta(0, "sig-only"), block_stop(0)];
    events.extend(terminal());

    let replied = reply(events).await;

    assert_eq!(
        replied.reasoning(),
        vec![ReasoningContent::Text {
            text: String::new(),
            signature: Some("sig-only".to_string()),
        }]
    );
}

/// A block that streamed nothing at all — an empty `Text` delta and no
/// signature — says nothing at its stop: the payload-less end must not
/// conjure an empty reasoning part.
#[tokio::test]
async fn wholly_empty_thinking_block_emits_nothing() {
    let mut events = vec![reasoning_text_delta(0, ""), block_stop(0)];
    events.extend(terminal());

    let replied = reply(events).await;

    assert!(replied.reasoning().is_empty());
    assert!(replied.response().choice.is_empty());
}

fn tool_start_event(index: i32, id: &str, name: &str) -> aws_bedrock::ConverseStreamOutput {
    aws_bedrock::ConverseStreamOutput::ContentBlockStart(
        aws_bedrock::ContentBlockStartEvent::builder()
            .content_block_index(index)
            .start(aws_bedrock::ContentBlockStart::ToolUse(
                aws_bedrock::ToolUseBlockStart::builder()
                    .tool_use_id(id)
                    .name(name)
                    .build()
                    .expect("tool use start should build"),
            ))
            .build()
            .expect("content block start should build"),
    )
}

fn tool_delta_event(index: i32, input: &str) -> aws_bedrock::ConverseStreamOutput {
    aws_bedrock::ConverseStreamOutput::ContentBlockDelta(
        aws_bedrock::ContentBlockDeltaEvent::builder()
            .content_block_index(index)
            .delta(aws_bedrock::ContentBlockDelta::ToolUse(
                aws_bedrock::ToolUseBlockDelta::builder()
                    .input(input)
                    .build()
                    .expect("tool use delta should build"),
            ))
            .build()
            .expect("content block delta should build"),
    )
}

fn text_delta_event(index: i32, text: &str) -> aws_bedrock::ConverseStreamOutput {
    aws_bedrock::ConverseStreamOutput::ContentBlockDelta(
        aws_bedrock::ContentBlockDeltaEvent::builder()
            .content_block_index(index)
            .delta(aws_bedrock::ContentBlockDelta::Text(text.to_string()))
            .build()
            .expect("content block delta should build"),
    )
}

fn block_stop_event(index: i32) -> aws_bedrock::ConverseStreamOutput {
    aws_bedrock::ConverseStreamOutput::ContentBlockStop(
        aws_bedrock::ContentBlockStopEvent::builder()
            .content_block_index(index)
            .build()
            .expect("content block stop should build"),
    )
}

fn message_stop_event(reason: aws_bedrock::StopReason) -> aws_bedrock::ConverseStreamOutput {
    aws_bedrock::ConverseStreamOutput::MessageStop(
        aws_bedrock::MessageStopEvent::builder()
            .stop_reason(reason)
            .build()
            .expect("message stop should build"),
    )
}

#[tokio::test]
async fn parallel_tool_calls_all_emitted_with_tool_use_terminal() {
    // Two tool-use blocks in one message: both survive, and the stop reason
    // maps to a tool-use finish.
    let replied = reply(vec![
        tool_start_event(0, "call_a", "get_weather"),
        tool_delta_event(0, "{\"location\":"),
        tool_delta_event(0, "\"Paris\"}"),
        block_stop_event(0),
        tool_start_event(1, "call_b", "get_time"),
        tool_delta_event(1, "{\"zone\":\"UTC\"}"),
        block_stop_event(1),
        message_stop_event(aws_bedrock::StopReason::ToolUse),
        metadata_event_with_usage(3, 1),
    ])
    .await;

    assert_eq!(
        replied.response().finish_reason(),
        Some(rig_core::completion::FinishReason::ToolCalls)
    );
    let calls = replied.calls();
    assert_eq!(calls.len(), 2, "both parallel tool calls must be emitted");
    let first = calls.first().expect("first call");
    assert_eq!(first.id.to_string(), "call_a");
    assert_eq!(first.function.name, "get_weather");
    assert_eq!(
        first.function.arguments,
        serde_json::json!({"location": "Paris"})
    );
    let second = calls.get(1).expect("second call");
    assert_eq!(second.id.to_string(), "call_b");
    assert_eq!(second.function.name, "get_time");
    assert_eq!(
        second.function.arguments,
        serde_json::json!({"zone": "UTC"})
    );
}

#[tokio::test]
async fn tool_call_flushes_at_content_block_stop() {
    // The call does not wait for MessageStop: closing the block ends it.
    let replied = reply(vec![
        tool_start_event(0, "call_a", "get_weather"),
        tool_delta_event(0, "{}"),
        block_stop_event(0),
    ])
    .await;

    assert!(
        replied.events.iter().any(|event| matches!(
            event,
            StreamEvent::End {
                content: AssistantContent::ToolCall(_),
                ..
            }
        )),
        "the call ends at its block stop"
    );
    assert!(
        matches!(replied.outcome, Err(ProviderError::Truncated)),
        "a reply with no Metadata event is truncated: {:?}",
        replied.outcome
    );
}

#[tokio::test]
async fn message_stop_flushes_stragglers_missing_a_block_stop() {
    // A stream that omits ContentBlockStop still delivers every call at
    // MessageStop, in block order.
    let replied = reply(vec![
        tool_start_event(0, "call_a", "get_weather"),
        tool_delta_event(0, "{\"location\":\"Paris\"}"),
        tool_start_event(1, "call_b", "get_time"),
        tool_delta_event(1, "{\"zone\":\"UTC\"}"),
        message_stop_event(aws_bedrock::StopReason::ToolUse),
        metadata_event_with_usage(3, 1),
    ])
    .await;

    let ids: Vec<String> = replied
        .calls()
        .iter()
        .map(|call| call.id.to_string())
        .collect();
    assert_eq!(ids, ["call_a", "call_b"]);
}

#[tokio::test]
async fn text_after_closed_tool_block_is_delivered() {
    // A text block following a closed tool-use block is kept.
    let replied = reply(vec![
        tool_start_event(0, "call_a", "get_weather"),
        tool_delta_event(0, "{}"),
        block_stop_event(0),
        text_delta_event(1, "Checking the weather now."),
        block_stop_event(1),
        message_stop_event(aws_bedrock::StopReason::EndTurn),
        metadata_event_with_usage(3, 1),
    ])
    .await;

    let texts: Vec<&str> = replied
        .events
        .iter()
        .filter_map(|event| match event {
            StreamEvent::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(texts, vec!["Checking the weather now."]);
    assert_eq!(replied.calls().len(), 1);
}

#[tokio::test]
async fn malformed_tool_json_fails_the_reply() {
    // Malformed input is not silently dropped while the finish still claims
    // tool use: the reply fails naming the tool.
    let replied = reply(vec![
        tool_start_event(0, "call_a", "get_weather"),
        tool_delta_event(0, "{\"location\": not-json"),
        block_stop_event(0),
        message_stop_event(aws_bedrock::StopReason::ToolUse),
        metadata_event_with_usage(3, 1),
    ])
    .await;

    let error = replied.outcome.expect_err("malformed tool JSON fails");
    assert_eq!(error.kind(), rig_core::error::ErrorKind::Response);
    assert!(error.to_string().contains("get_weather"), "{error}");
}

#[tokio::test]
async fn max_tokens_stop_drops_in_flight_tool_block_without_deltas() {
    // A tool-use block cut off by MaxTokens before any input arrived gives
    // neither a fabricated `{}`-args call nor an error; the length finish
    // reports the cut.
    let replied = reply(vec![
        tool_start_event(0, "call_a", "get_weather"),
        message_stop_event(aws_bedrock::StopReason::MaxTokens),
        metadata_event_with_usage(3, 1),
    ])
    .await;

    assert!(replied.calls().is_empty());
    assert_eq!(
        replied.response().finish_reason(),
        Some(rig_core::completion::FinishReason::Length)
    );
}

#[tokio::test]
async fn max_tokens_stop_drops_in_flight_tool_block_with_partial_json() {
    // Same, with partial JSON buffered: it is not parsed into an error.
    let replied = reply(vec![
        tool_start_event(0, "call_a", "get_weather"),
        tool_delta_event(0, "{\"location\":\"Par"),
        message_stop_event(aws_bedrock::StopReason::MaxTokens),
        metadata_event_with_usage(3, 1),
    ])
    .await;

    assert!(replied.calls().is_empty());
    assert_eq!(
        replied.response().finish_reason(),
        Some(rig_core::completion::FinishReason::Length)
    );
}

#[tokio::test]
async fn empty_tool_input_becomes_empty_object() {
    // A tool with no parameters streams no input deltas at all.
    let replied = reply(vec![
        tool_start_event(0, "call_a", "ping"),
        block_stop_event(0),
        message_stop_event(aws_bedrock::StopReason::ToolUse),
        metadata_event_with_usage(3, 1),
    ])
    .await;

    let calls = replied.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls.first().expect("call").function.arguments,
        serde_json::json!({})
    );
}

/// Bedrock's terminal `Metadata` event carrying usage, so the stream ends
/// with a fully populated `BedrockStreamingResponse`.
fn metadata_event_with_usage(input: i32, output: i32) -> aws_bedrock::ConverseStreamOutput {
    aws_bedrock::ConverseStreamOutput::Metadata(
        aws_bedrock::ConverseStreamMetadataEvent::builder()
            .usage(
                aws_bedrock::TokenUsage::builder()
                    .input_tokens(input)
                    .output_tokens(output)
                    .total_tokens(input + output)
                    .build()
                    .expect("token usage should build"),
            )
            .build(),
    )
}

/// A stream's `raw` is Bedrock's own `BedrockStreamingResponse`: it
/// deserializes back into that type and re-serializes identically, and the
/// Bedrock `stopReason` spelling is only readable off it.
#[tokio::test]
async fn a_streams_raw_round_trips_into_the_terminal_type() {
    let replied = reply(vec![
        text_delta_event(0, "hi"),
        block_stop(0),
        message_stop_event(aws_bedrock::StopReason::EndTurn),
        metadata_event_with_usage(3, 1),
    ])
    .await;
    let response = replied.response();

    let typed: BedrockStreamingResponse =
        serde_json::from_value(response.raw.clone()).expect("raw must deserialize");
    assert_eq!(
        serde_json::to_value(&typed).expect("re-serialize"),
        response.raw,
        "the capture must be exactly what the terminal type serializes to"
    );
    assert_eq!(typed.stop_reason, Some(StopReason::EndTurn));
    assert_eq!(response.usage, rig_core::completion::Usage::from(&typed));
    assert_eq!(response.usage.total_tokens, Some(4));
    assert_eq!(
        response.finish_reason(),
        Some(rig_core::completion::FinishReason::Stop)
    );
}

/// A Claude stream names `anthropic` as its reasoning issuer, and the folded
/// turn's reasoning records it.
#[tokio::test]
async fn a_claude_stream_records_anthropic_as_its_reasoning_issuer() {
    let mut stream = stream_of(
        crate::completion::ANTHROPIC_CLAUDE_SONNET_4_6,
        vec![
            reasoning_text_delta(0, "thinking"),
            reasoning_signature_delta(0, "sig"),
            block_stop(0),
            text_delta_event(1, "done"),
            block_stop(1),
            message_stop_event(aws_bedrock::StopReason::EndTurn),
            metadata_event_with_usage(3, 1),
        ],
    );
    // The reply names the model it answers for before its first event.
    stream.next().await.expect("an event").expect("stream item");
    assert_eq!(stream.reasoning_issuer().as_str(), "anthropic");
    while let Some(item) = stream.next().await {
        item.expect("stream item");
    }
    let response = stream.finish().await.expect("a finished reply");
    let issuers: Vec<_> = response
        .choice
        .iter()
        .filter_map(|part| match part {
            AssistantContent::Reasoning(reasoning) => Some(reasoning.issuer().as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(issuers, ["anthropic"]);
}

/// Opening a Converse stream sends nothing until the first poll, as an HTTP
/// stream does, so a send that fails is the stream's first item rather than
/// an error from `stream`.
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
    let request = rig_core::completion::CompletionRequest::new("hi");
    let mut stream = Model::new(Converse::new("amazon.nova-lite-v1:0"), runtime)
        .stream(request)
        .expect("opening a stream sends nothing");
    let first = stream.next().await.expect("the stream yields the failure");
    assert!(
        first.is_err(),
        "the failed send is the first item: {first:?}"
    );
}

/// A unary reply goes through the same fold as a stream, so an empty text
/// block is no content on both paths.
#[tokio::test]
async fn an_empty_text_block_is_no_content_unary_or_streamed() {
    use crate::types::converse_output::InternalConverseOutput;

    let message = aws_bedrock::Message::builder()
        .role(aws_bedrock::ConversationRole::Assistant)
        .content(aws_bedrock::ContentBlock::Text(String::new()))
        .build()
        .expect("message builds");
    let output: InternalConverseOutput =
        aws_sdk_bedrockruntime::operation::converse::ConverseOutput::builder()
            .output(aws_bedrock::ConverseOutput::Message(message))
            .stop_reason(aws_bedrock::StopReason::EndTurn)
            .build()
            .expect("output builds")
            .try_into()
            .expect("output mirrors");
    let unary = crate::types::assistant_content::tests::complete(output).expect("unary reply");

    let mut stream = stream_of(
        "amazon.nova-lite-v1:0",
        vec![
            text_delta_event(0, ""),
            block_stop(0),
            message_stop_event(aws_bedrock::StopReason::EndTurn),
            metadata_event_with_usage(1, 0),
        ],
    );
    while let Some(item) = stream.next().await {
        item.expect("stream item");
    }
    let streamed = stream.finish().await.expect("streamed reply");

    assert!(unary.choice.is_empty(), "{:?}", unary.choice);
    assert!(streamed.choice.is_empty(), "{:?}", streamed.choice);
}
