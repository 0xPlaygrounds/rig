use super::*;
use futures::StreamExt;
use rig_core::message::Reasoning;
use rig_core::streaming::StreamedAssistantContent;

// ---- Event-seam helpers: no AWS transport, `stream_from_events` only ----

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

struct Drained {
    reasoning: Vec<Reasoning>,
    errors: Vec<String>,
    reached_terminal: bool,
}

async fn drain(events: Vec<aws_bedrock::ConverseStreamOutput>) -> Drained {
    let mut stream = stream_from_events(futures::stream::iter(events.into_iter().map(Ok)));
    let mut drained = Drained {
        reasoning: Vec::new(),
        errors: Vec::new(),
        reached_terminal: false,
    };

    while let Some(item) = stream.next().await {
        match item {
            Ok(StreamedAssistantContent::Reasoning { reasoning, .. }) => {
                drained.reasoning.push(reasoning);
            }
            Ok(StreamedAssistantContent::Final(_)) => drained.reached_terminal = true,
            Ok(_) => {}
            Err(error) => drained.errors.push(error.to_string()),
        }
    }

    drained
}

/// Ordinary extended-thinking shape through the SHARED driver: thinking
/// deltas, the block's whole-block close at `contentBlockStop`, then
/// visible text. The driver's boundary law must treat the same-key whole
/// block as a close — this exact stream used to abort every debug build
/// (sequence-law O1).
#[tokio::test]
async fn thinking_then_text_streams_through_the_driver_without_violation() {
    let drained = drain(vec![
        reasoning_text_delta(0, "let me think"),
        block_stop(0),
        text_delta_event(1, "the answer"),
        block_stop_event(1),
        message_stop_event(aws_bedrock::StopReason::EndTurn),
    ])
    .await;
    assert!(drained.errors.is_empty(), "{:?}", drained.errors);
    assert_eq!(drained.reasoning.len(), 1);
    assert_eq!(
        drained
            .reasoning
            .iter()
            .flat_map(|reasoning| reasoning.content.iter())
            .cloned()
            .collect::<Vec<_>>(),
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

    let drained = drain(events).await;

    assert!(drained.errors.is_empty(), "errors: {:?}", drained.errors);
    assert_eq!(
        drained
            .reasoning
            .iter()
            .flat_map(|reasoning| reasoning.content.iter())
            .cloned()
            .collect::<Vec<_>>(),
        vec![ReasoningContent::Redacted {
            data: BASE64_STANDARD.encode(REDACTED_BLOB),
        }]
    );
    assert!(drained.reached_terminal);
}

/// The redacted block must land BESIDE an open thinking block, not replace
/// it: both share `block-{index}`, so without draining the open state
/// first the accumulator would supersede the delta-built thinking part.
#[tokio::test]
async fn redacted_reasoning_is_a_sibling_of_the_open_thinking_block() {
    let mut events = vec![
        reasoning_text_delta(0, "visible thinking"),
        reasoning_signature_delta(0, "sig_1"),
        reasoning_redacted_delta(0, REDACTED_BLOB),
        block_stop(0),
    ];
    events.extend(terminal());

    let drained = drain(events).await;

    assert!(drained.errors.is_empty(), "errors: {:?}", drained.errors);
    let content: Vec<ReasoningContent> = drained
        .reasoning
        .iter()
        .flat_map(|reasoning| reasoning.content.iter())
        .cloned()
        .collect();
    assert_eq!(
        content,
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
    assert!(drained.reached_terminal);
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

    let drained = drain(events).await;

    assert!(
        drained.errors.is_empty(),
        "an unmodeled ContentBlockStart must not fail the stream: {:?}",
        drained.errors
    );
    assert!(
        drained.reached_terminal,
        "the stream must still reach its terminal record"
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
            input_tokens: 200,
            output_tokens: 75,
            total_tokens: 275,
            cached_input_tokens: 40,
            cache_creation_input_tokens: 10,
            tool_use_prompt_tokens: 0,
            reasoning_tokens: 0,
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

    // Zero-valued usage is rig's documented sentinel for "the provider
    // reported no usage metrics".
    assert_eq!(
        rig_core::completion::Usage::from(&response),
        rig_core::completion::Usage::new()
    );
    assert!(!rig_core::completion::Usage::from(&response).has_values());
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
            input_tokens: 448,
            output_tokens: 68,
            total_tokens: 516,
            cached_input_tokens: 80,
            cache_creation_input_tokens: 20,
            tool_use_prompt_tokens: 0,
            reasoning_tokens: 0,
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
/// text the shared accumulator assembled from the deltas — the exact
/// shape the next turn must replay to Bedrock.
#[tokio::test]
async fn signed_thinking_block_closes_with_its_signature() {
    let mut events = vec![
        reasoning_text_delta(0, "I am "),
        reasoning_text_delta(0, "thinking"),
        reasoning_signature_delta(0, "sig-abc"),
        block_stop(0),
    ];
    events.extend(terminal());

    let drained = drain(events).await;

    assert!(drained.errors.is_empty(), "errors: {:?}", drained.errors);
    assert_eq!(
        drained
            .reasoning
            .iter()
            .flat_map(|reasoning| reasoning.content.iter())
            .cloned()
            .collect::<Vec<_>>(),
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

    let drained = drain(events).await;

    assert!(drained.errors.is_empty(), "errors: {:?}", drained.errors);
    assert_eq!(
        drained
            .reasoning
            .iter()
            .flat_map(|reasoning| reasoning.content.iter())
            .cloned()
            .collect::<Vec<_>>(),
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

    let drained = drain(events).await;

    assert!(drained.errors.is_empty(), "errors: {:?}", drained.errors);
    assert!(drained.reasoning.is_empty());
    assert!(drained.reached_terminal);
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

/// Run a sequence of events through [`process_event`] with fresh state,
/// returning every item the stream would yield, plus the final state.
fn run_events(
    events: Vec<aws_bedrock::ConverseStreamOutput>,
) -> (
    Vec<Result<RawStreamingChoice<BedrockStreamingResponse>, CompletionError>>,
    StreamState,
) {
    let mut state = StreamState::default();
    let mut items = Vec::new();
    for event in events {
        items.extend(process_event(&mut state, event));
    }
    (items, state)
}

/// Drive the raw items through the same normalized pipeline the public
/// stream uses (terminal mapping plus the shared accumulator), returning
/// the completed tool calls and the in-band errors a consumer would see.
/// Tool-call finalization happens in the accumulator, so assertions about
/// completed calls and malformed-input errors belong at this level.
async fn assembled(
    items: Vec<Result<RawStreamingChoice<BedrockStreamingResponse>, CompletionError>>,
) -> (Vec<rig_core::message::ToolCall>, Vec<CompletionError>) {
    use futures::StreamExt;
    let raw: rig_core::streaming::RawStreamingResult<BedrockStreamingResponse> =
        Box::pin(futures::stream::iter(items));
    let mut stream =
        StreamingCompletionResponse::stream(PROVIDER_NAME, normalize_bedrock_stream(raw));
    let mut calls = Vec::new();
    let mut errors = Vec::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(rig_core::streaming::StreamedAssistantContent::ToolCall { tool_call, .. }) => {
                calls.push(tool_call)
            }
            Err(err) => errors.push(err),
            Ok(_) => {}
        }
    }
    (calls, errors)
}

#[tokio::test]
async fn parallel_tool_calls_all_emitted_with_tool_use_terminal() {
    // Two tool-use blocks in one message: both must survive, and the
    // latched stop reason must map to a tool-use terminal.
    let (items, state) = run_events(vec![
        tool_start_event(0, "call_a", "get_weather"),
        tool_delta_event(0, "{\"location\":"),
        tool_delta_event(0, "\"Paris\"}"),
        block_stop_event(0),
        tool_start_event(1, "call_b", "get_time"),
        tool_delta_event(1, "{\"zone\":\"UTC\"}"),
        block_stop_event(1),
        message_stop_event(aws_bedrock::StopReason::ToolUse),
    ]);

    assert!(items.iter().all(std::result::Result::is_ok));
    // The terminal reports tool use with the calls actually delivered.
    assert_eq!(state.final_stop_reason, Some(StopReason::ToolUse));
    assert_eq!(
        map_stop_reason(&StopReason::ToolUse),
        rig_core::completion::FinishReason::ToolCalls
    );

    let (calls, errors) = assembled(items).await;
    assert!(errors.is_empty());
    assert_eq!(calls.len(), 2, "both parallel tool calls must be emitted");
    let first = calls.first().expect("first call");
    assert_eq!(first.id, "call_a");
    assert_eq!(first.function.name, "get_weather");
    assert_eq!(
        first.function.arguments,
        serde_json::json!({"location": "Paris"})
    );
    let second = calls.get(1).expect("second call");
    assert_eq!(second.id, "call_b");
    assert_eq!(second.function.name, "get_time");
    assert_eq!(
        second.function.arguments,
        serde_json::json!({"zone": "UTC"})
    );
}

#[tokio::test]
async fn tool_call_flushes_at_content_block_stop() {
    // The call must not wait for MessageStop: closing the block emits it.
    let (items, state) = run_events(vec![
        tool_start_event(0, "call_a", "get_weather"),
        tool_delta_event(0, "{}"),
        block_stop_event(0),
    ]);

    assert!(state.tool_calls.is_empty(), "state must be cleared at stop");
    let (calls, errors) = assembled(items).await;
    assert!(errors.is_empty());
    assert_eq!(calls.len(), 1);
}

#[tokio::test]
async fn message_stop_flushes_stragglers_missing_a_block_stop() {
    // Defensive path: a stream that omits ContentBlockStop still delivers
    // every accumulated call at MessageStop, in block order.
    let (items, _state) = run_events(vec![
        tool_start_event(0, "call_a", "get_weather"),
        tool_delta_event(0, "{\"location\":\"Paris\"}"),
        tool_start_event(1, "call_b", "get_time"),
        tool_delta_event(1, "{\"zone\":\"UTC\"}"),
        message_stop_event(aws_bedrock::StopReason::ToolUse),
    ]);

    let (calls, errors) = assembled(items).await;
    assert!(errors.is_empty());
    assert_eq!(calls.len(), 2);
    assert_eq!(calls.first().expect("first call").id, "call_a");
    assert_eq!(calls.get(1).expect("second call").id, "call_b");
}

#[tokio::test]
async fn text_after_closed_tool_block_is_delivered() {
    // A text block following a closed tool-use block used to be discarded
    // because the single tool slot was never cleared.
    let (items, _state) = run_events(vec![
        tool_start_event(0, "call_a", "get_weather"),
        tool_delta_event(0, "{}"),
        block_stop_event(0),
        text_delta_event(1, "Checking the weather now."),
        block_stop_event(1),
        message_stop_event(aws_bedrock::StopReason::EndTurn),
    ]);

    let texts: Vec<&str> = items
        .iter()
        .filter_map(|item| match item {
            Ok(RawStreamingChoice::Message(text)) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(texts, vec!["Checking the weather now."]);
    let (calls, errors) = assembled(items).await;
    assert!(errors.is_empty());
    assert_eq!(calls.len(), 1);
}

#[tokio::test]
async fn malformed_tool_json_surfaces_an_error_item() {
    // Malformed accumulated input must not be silently dropped while the
    // terminal still claims tool use: the consumer gets an error item.
    let (items, _state) = run_events(vec![
        tool_start_event(0, "call_a", "get_weather"),
        tool_delta_event(0, "{\"location\": not-json"),
        block_stop_event(0),
        message_stop_event(aws_bedrock::StopReason::ToolUse),
    ]);

    let (calls, errors) = assembled(items).await;
    assert!(calls.is_empty());
    assert!(
        errors.iter().any(|err| matches!(
            err,
            CompletionError::ResponseError(msg) if msg.contains("get_weather")
        )),
        "malformed tool JSON must yield an error item"
    );
}

#[tokio::test]
async fn max_tokens_stop_drops_in_flight_tool_block_without_deltas() {
    // A tool-use block cut off by MaxTokens before any input arrived must
    // produce neither a fabricated `{}`-args call nor an error item; the
    // truncation is signaled by the Length-mapping stop reason on the
    // terminal record.
    let (items, state) = run_events(vec![
        tool_start_event(0, "call_a", "get_weather"),
        message_stop_event(aws_bedrock::StopReason::MaxTokens),
    ]);

    assert!(
        items.iter().all(std::result::Result::is_ok),
        "truncation must not surface as an error item"
    );
    assert_eq!(state.final_stop_reason, Some(StopReason::MaxTokens));
    assert_eq!(
        map_stop_reason(&StopReason::MaxTokens),
        rig_core::completion::FinishReason::Length
    );
    assert!(state.tool_calls.is_empty(), "state must be cleared at stop");
    let (calls, errors) = assembled(items).await;
    assert!(calls.is_empty());
    assert!(errors.is_empty(), "truncation must not surface as an error");
}

#[tokio::test]
async fn max_tokens_stop_drops_in_flight_tool_block_with_partial_json() {
    // Same, but with partial JSON accumulated: the malformed input must
    // not be parsed into a spurious Err at MessageStop.
    let (items, state) = run_events(vec![
        tool_start_event(0, "call_a", "get_weather"),
        tool_delta_event(0, "{\"location\":\"Par"),
        message_stop_event(aws_bedrock::StopReason::MaxTokens),
    ]);

    assert!(
        items.iter().all(std::result::Result::is_ok),
        "a truncated partial-JSON block must not yield an error item"
    );
    assert_eq!(state.final_stop_reason, Some(StopReason::MaxTokens));
    assert!(state.tool_calls.is_empty(), "state must be cleared at stop");
    let (calls, errors) = assembled(items).await;
    assert!(calls.is_empty());
    assert!(errors.is_empty(), "no spurious Err from the partial block");
}

#[tokio::test]
async fn empty_tool_input_becomes_empty_object() {
    // A tool with no parameters streams no input deltas at all.
    let (items, _state) = run_events(vec![
        tool_start_event(0, "call_a", "ping"),
        block_stop_event(0),
        message_stop_event(aws_bedrock::StopReason::ToolUse),
    ]);

    let (calls, errors) = assembled(items).await;
    assert!(errors.is_empty());
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

/// Drive `items` through the normalized pipeline exactly as the
/// `CompletionModel` seam does, returning the terminal.
async fn normalized_terminal(
    items: Vec<Result<RawStreamingChoice<BedrockStreamingResponse>, CompletionError>>,
) -> rig_core::streaming::StreamFinal {
    let raw: rig_core::streaming::RawStreamingResult<BedrockStreamingResponse> =
        Box::pin(futures::stream::iter(items));
    let mut stream =
        StreamingCompletionResponse::stream(PROVIDER_NAME, normalize_bedrock_stream(raw));
    while let Some(item) = stream.next().await {
        item.expect("stream item");
    }
    stream
        .response
        .expect("the stream must end with a terminal record")
}

/// The events-first seam captures like the request-driven one: its
/// terminal `raw` is the same `BedrockStreamingResponse` the model's
/// `stream()` would attach, because both funnel through
/// `normalize_bedrock_stream`.
#[tokio::test]
async fn stream_from_events_terminal_carries_raw() {
    let mut stream = stream_from_events(futures::stream::iter(
        vec![
            text_delta_event(0, "hi"),
            block_stop(0),
            message_stop_event(aws_bedrock::StopReason::EndTurn),
            metadata_event_with_usage(3, 1),
        ]
        .into_iter()
        .map(Ok),
    ));
    while let Some(item) = stream.next().await {
        item.expect("stream item");
    }
    let terminal = stream.response.expect("terminal record");

    let raw = &terminal.raw;
    let typed: BedrockStreamingResponse =
        serde_json::from_value(raw.clone()).expect("raw must deserialize");
    assert_eq!(typed.stop_reason, Some(StopReason::EndTurn));
    assert_eq!(terminal.usage.total_tokens, 4);
}

/// The load-bearing streaming capture property at the seam
/// `CompletionModel::stream` routes through: the terminal's `raw` is
/// Bedrock's own `BedrockStreamingResponse` — it deserializes back into
/// that type and re-serializes identically — and re-normalizing that
/// capture reproduces every normalized field. The Bedrock `stopReason`
/// spelling is only readable off the capture.
#[tokio::test]
async fn terminal_raw_round_trips_into_the_terminal_type() {
    let (items, _) = run_events(vec![
        text_delta_event(0, "hi"),
        block_stop(0),
        message_stop_event(aws_bedrock::StopReason::EndTurn),
        metadata_event_with_usage(3, 1),
    ]);
    let terminal = normalized_terminal(items).await;

    let raw = &terminal.raw;
    let typed: BedrockStreamingResponse =
        serde_json::from_value(raw.clone()).expect("raw must deserialize");
    assert_eq!(
        serde_json::to_value(&typed).expect("re-serialize"),
        *raw,
        "the capture must be exactly what the terminal type serializes to"
    );
    assert_eq!(typed.stop_reason, Some(StopReason::EndTurn));
    assert_eq!(
        typed.usage.as_ref().map(|usage| usage.total_tokens),
        Some(4)
    );

    // Feeding the capture back through the same pipeline tells the same
    // story as the terminal the stream produced.
    let renormalized =
        normalized_terminal(vec![Ok(RawStreamingChoice::FinalResponse(typed))]).await;
    assert_eq!(terminal.identity(), renormalized.identity());
    assert_eq!(terminal.finish_reason, renormalized.finish_reason);
    assert_eq!(terminal.model, renormalized.model);
    assert_eq!(terminal.usage, renormalized.usage);
    assert_eq!(
        terminal.finish_reason,
        Some(rig_core::completion::FinishReason::Stop)
    );
}
