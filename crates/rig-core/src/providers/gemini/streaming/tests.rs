use super::*;
use crate::completion::FinishReason;
use crate::message::AssistantContent;
use serde_json::json;

/// The request every stream test below sends. The decoder is what they
/// exercise, so the request only has to be well-formed.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
fn streaming_request() -> crate::completion::CompletionRequest {
    crate::completion::CompletionRequest::new("hello")
}

/// The GenerateContent wire for `model`, bound to a transport that answers
/// with `frames` as one SSE body.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
fn streamed(
    model: &str,
    frames: &[&str],
) -> crate::driver::Model<
    crate::providers::gemini::completion::GenerateContent,
    crate::test_utils::MockStreamingClient,
> {
    let sse_bytes = bytes::Bytes::from(
        frames
            .iter()
            .map(|frame| format!("data: {frame}\n\n"))
            .collect::<String>(),
    );
    crate::driver::Model::new(
        crate::providers::gemini::GeminiConfig::new("test-key").completion(model),
        crate::test_utils::MockStreamingClient { sse_bytes },
    )
}

/// `reply`, one whole `generateContent` body, decoded by the REST wire.
fn decoded(
    reply: serde_json::Value,
) -> Result<crate::completion::CompletionResponse, ProviderError> {
    let wire =
        crate::providers::gemini::GeminiConfig::new("test-key").completion("gemini-2.5-flash");
    crate::test_utils::history::decode(
        &wire,
        crate::wire::Mode::Unary,
        [WireFrame::Text(reply.to_string())],
    )
}

#[test]
fn test_deserialize_stream_response_with_single_text_part() {
    let json_data = json!({
        "candidates": [{
            "content": {
                "parts": [
                    {"text": "Hello, world!"}
                ],
                "role": "model"
            },
            "finishReason": "STOP",
            "index": 0
        }],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15
        }
    });

    let response = decoded(json_data).expect("the reply decodes");
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    let [AssistantContent::Text(text)] = response.choice.as_slice() else {
        panic!("Expected one text block: {:?}", response.choice);
    };
    assert_eq!(text.text, "Hello, world!");
}

#[test]
fn a_tool_protocol_finish_is_a_failed_turn_not_a_failed_reply() {
    use crate::test_utils::history_conformance::decode;
    let wire =
        crate::providers::gemini::GeminiConfig::new("test-key").completion("gemini-2.5-flash");
    for finish_reason in [
        "MALFORMED_FUNCTION_CALL",
        "UNEXPECTED_TOOL_CALL",
        "MISSING_THOUGHT_SIGNATURE",
        "TOO_MANY_TOOL_CALLS",
        "MALFORMED_RESPONSE",
    ] {
        let frame = json!({
            "candidates": [{
                "finishReason": finish_reason,
                "finishMessage": "the call was malformed",
                "index": 0
            }]
        });
        let response = decode(
            &wire,
            &crate::completion::CompletionRequest::new("hi"),
            crate::wire::Mode::Unary,
            [crate::wire::WireFrame::Text(frame.to_string())],
        )
        .unwrap_or_else(|error| panic!("{finish_reason} is a turn: {error}"));
        assert!(
            response.stop().is_failure(),
            "{finish_reason} ends the turn as a failure: {:?}",
            response.stop()
        );
    }
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn a_failure_finish_ends_the_stream_without_draining_later_frames() {
    use crate::streaming::{Item, StreamEvent};
    use futures::StreamExt;

    // A tool-protocol failure, then more frames: a text chunk, an unknown
    // frame, and a `STOP` chunk. The failure ends the reply, so nothing
    // after it is read and a later `STOP` cannot make the turn look clean.
    let frames = [
        r#"{"candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"},"index":0}]}"#,
        r#"{"candidates":[{"finishReason":"MALFORMED_FUNCTION_CALL","finishMessage":"malformed function call","index":0}]}"#,
        r#"{"candidates":[{"content":{"parts":[{"text":"dead"}],"role":"model"},"index":0}]}"#,
        r#"{"someFutureField":{"x":1}}"#,
        r#"{"candidates":[{"content":{"parts":[],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}"#,
    ];

    let model = streamed("gemini-2.5-flash", &frames);
    let mut stream = model
        .stream(streaming_request())
        .expect("stream should open");

    let mut texts = Vec::new();
    while let Some(item) = stream.next().await {
        match item {
            Ok(Item::Event(StreamEvent::Text { text, .. })) => texts.push(text),
            Ok(_) => {}
            Err(error) => panic!("a failure finish is not a stream error: {error}"),
        }
    }

    assert_eq!(texts, ["hi"]);
    let response = stream.finish().await.expect("the reply ended");
    assert!(
        response.stop().is_failure(),
        "the turn failed: {:?}",
        response.stop()
    );
}

#[test]
fn test_deserialize_stream_response_with_usage_only_chunk() {
    let json_data = json!({
        "responseId": "response-123",
        "modelVersion": "gemini-2.0-flash-001",
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15
        }
    });
    // A usage-only chunk carries no candidate: it neither opens a block nor
    // ends the reply, and the finish chunk after it reports its identity.
    let finish = json!({
        "candidates": [{ "content": { "parts": [{ "text": "ok" }], "role": "model" }, "finishReason": "STOP" }]
    });
    let wire =
        crate::providers::gemini::GeminiConfig::new("test-key").completion("gemini-2.5-flash");
    let response = crate::test_utils::history::decode(
        &wire,
        crate::wire::Mode::Streaming,
        [
            WireFrame::Text(json_data.to_string()),
            WireFrame::Text(finish.to_string()),
        ],
    )
    .expect("the reply decodes");

    assert_eq!(response.response_id(), Some("response-123"));
    assert_eq!(response.model(), Some("gemini-2.0-flash-001"));
    assert_eq!(response.text(), "ok");
    assert_eq!(response.usage.input_tokens, Some(10));
    assert_eq!(response.usage.output_tokens, Some(5));
    assert_eq!(response.usage.total_tokens, Some(15));
}

#[test]
fn test_deserialize_stream_response_with_multiple_text_parts() {
    let json_data = json!({
        "candidates": [{
            "content": {
                "parts": [
                    {"text": "Hello, "},
                    {"text": "world!"},
                    {"text": " How are you?"}
                ],
                "role": "model"
            },
            "finishReason": "STOP",
            "index": 0
        }],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 8,
            "totalTokenCount": 18
        }
    });

    let response = decoded(json_data).expect("the reply decodes");
    // Consecutive text parts continue one text block, and its provider item
    // holds the merged text.
    let [AssistantContent::Text(text)] = response.choice.as_slice() else {
        panic!("Expected one text block: {:?}", response.choice);
    };
    assert_eq!(text.text, "Hello, world! How are you?");
    assert_eq!(
        text.native.as_ref().map(|native| &native.item),
        Some(&json!({ "text": "Hello, world! How are you?" }))
    );
}

#[test]
fn test_deserialize_stream_response_with_multiple_tool_calls() {
    let json_data = json!({
        "candidates": [{
            "content": {
                "parts": [
                    {
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"city": "San Francisco"},
                            "id": "call-weather"
                        }
                    },
                    {
                        "functionCall": {
                            "name": "get_temperature",
                            "args": {"location": "New York"},
                            "id": "call-temperature"
                        }
                    }
                ],
                "role": "model"
            },
            "finishReason": "STOP",
            "index": 0
        }],
        "usageMetadata": {
            "promptTokenCount": 50,
            "candidatesTokenCount": 20,
            "totalTokenCount": 70
        }
    });

    let response = decoded(json_data).expect("the reply decodes");
    let [
        AssistantContent::ToolCall(weather),
        AssistantContent::ToolCall(temperature),
    ] = response.choice.as_slice()
    else {
        panic!("Expected two function calls: {:?}", response.choice);
    };
    assert_eq!(weather.function.name, "get_weather");
    assert_eq!(weather.id.wire(), "call-weather");
    assert_eq!(
        weather.function.arguments_value(),
        json!({"city": "San Francisco"})
    );
    assert_eq!(temperature.function.name, "get_temperature");
    assert_eq!(temperature.id.wire(), "call-temperature");
    assert_eq!(
        temperature.function.arguments_value(),
        json!({"location": "New York"})
    );
}

#[test]
fn test_deserialize_stream_response_with_mixed_parts() {
    let json_data = json!({
        "candidates": [{
            "content": {
                "parts": [
                    {
                        "text": "Let me think about this...",
                        "thought": true
                    },
                    {
                        "text": "Here's my response: "
                    },
                    {
                        "functionCall": {
                            "name": "search",
                            "args": {"query": "rust async"}
                        }
                    },
                    {
                        "text": "I found the answer!"
                    }
                ],
                "role": "model"
            },
            "finishReason": "STOP",
            "index": 0
        }],
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 50,
            "thoughtsTokenCount": 15,
            "totalTokenCount": 165
        }
    });

    let response = decoded(json_data).expect("the reply decodes");
    let [
        AssistantContent::Reasoning(thought),
        AssistantContent::Text(text),
        AssistantContent::ToolCall(call),
        AssistantContent::Text(last),
    ] = response.choice.as_slice()
    else {
        panic!(
            "Expected a thought, text, a call and text: {:?}",
            response.choice
        );
    };
    assert_eq!(thought.text, "Let me think about this...");
    assert_eq!(text.text, "Here's my response: ");
    assert_eq!(call.function.name, "search");
    assert_eq!(last.text, "I found the answer!");
    assert_eq!(response.usage.reasoning_tokens, Some(15));
}

#[test]
fn test_deserialize_stream_response_with_empty_parts() {
    let json_data = json!({
        "candidates": [{
            "content": {
                "parts": [],
                "role": "model"
            },
            "finishReason": "STOP",
            "index": 0
        }],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 0,
            "totalTokenCount": 10
        }
    });

    // An empty successful reply is a success with no blocks.
    let response = decoded(json_data).expect("an empty reply decodes");
    assert!(response.choice.is_empty(), "{:?}", response.choice);
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
}

#[test]
fn test_partial_usage_token_calculation() {
    let usage = json!({
        "totalTokenCount": 92,
        "cachedContentTokenCount": 20,
        "candidatesTokenCount": 30,
        "thoughtsTokenCount": 10,
        "promptTokenCount": 40,
        "toolUsePromptTokenCount": 12,
    });

    let token_usage = usage_of(&usage);
    // Input is the prompt plus the hosted-tool prompt, output the candidates
    // plus the thoughts, and the total their sum, as `totalTokenCount` is.
    assert_eq!(token_usage.input_tokens, Some(52));
    assert_eq!(token_usage.cached_input_tokens, Some(20));
    assert_eq!(token_usage.output_tokens, Some(40));
    assert_eq!(token_usage.reasoning_tokens, Some(10));
    assert_eq!(token_usage.tool_use_prompt_tokens, Some(12));
    assert_eq!(token_usage.total_tokens, Some(92));
}

#[test]
fn test_partial_usage_with_missing_counts() {
    let usage = json!({
        "totalTokenCount": 50,
        "candidatesTokenCount": 30,
        "promptTokenCount": 20,
    });

    let token_usage = usage_of(&usage);
    assert_eq!(token_usage.input_tokens, Some(20));
    assert_eq!(token_usage.cached_input_tokens, None);
    assert_eq!(token_usage.output_tokens, Some(30));
    assert_eq!(token_usage.reasoning_tokens, None);
    assert_eq!(token_usage.total_tokens, Some(50));
}

#[test]
fn test_partial_usage_reads_without_total_token_count() {
    // Gemini's proto3-JSON encoding omits fields whose value is the default (0),
    // so `totalTokenCount` is absent on short/empty/blocked generations.
    let usage = usage_of(&json!({"promptTokenCount": 12}));
    assert_eq!(usage.input_tokens, Some(12));
    assert_eq!(usage.output_tokens, Some(0));
    assert_eq!(usage.total_tokens, Some(12));
}

#[test]
fn test_streaming_completion_response_has_finish_reason_and_model_version() {
    let response = decoded(json!({
        "candidates": [{
            "content": { "parts": [{ "text": "hi" }], "role": "model" },
            "finishReason": "STOP"
        }],
        "modelVersion": "gemini-2.5-pro-preview-05-06"
    }))
    .expect("the reply decodes");

    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(response.model(), Some("gemini-2.5-pro-preview-05-06"));
    assert_eq!(response.raw["finish_reason"], "STOP");
    assert_eq!(
        response.raw["model_version"],
        "gemini-2.5-pro-preview-05-06"
    );
}

#[test]
fn test_streaming_completion_response_token_usage() {
    let response = decoded(json!({
        "candidates": [{
            "content": { "parts": [{ "text": "hi" }], "role": "model" },
            "finishReason": "STOP"
        }],
        "usageMetadata": {
            "totalTokenCount": 150,
            "candidatesTokenCount": 75,
            "promptTokenCount": 75
        },
        "modelVersion": "gemini-2.0-flash-001"
    }))
    .expect("the reply decodes");

    let token_usage = response.usage;
    assert_eq!(token_usage.input_tokens, Some(75));
    assert_eq!(token_usage.output_tokens, Some(75));
    assert_eq!(token_usage.reasoning_tokens, None);
    assert_eq!(token_usage.cached_input_tokens, None);
    assert_eq!(token_usage.total_tokens, Some(150));
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(response.model(), Some("gemini-2.0-flash-001"));
}

#[test]
fn test_partial_usage_reads_every_count_and_ignores_the_details() {
    let json_data = serde_json::json!({
        "promptTokenCount": 100,
        "cachedContentTokenCount": 25,
        "candidatesTokenCount": 50,
        "thoughtsTokenCount": 15,
        "totalTokenCount": 177,
        "promptTokensDetails": [
            { "modality": "TEXT", "tokenCount": 80 },
            { "modality": "IMAGE", "tokenCount": 20 }
        ],
        "cacheTokensDetails": [
            { "modality": "TEXT", "tokenCount": 25 }
        ],
        "candidatesTokensDetails": [
            { "modality": "TEXT", "tokenCount": 50 }
        ],
        "toolUsePromptTokenCount": 12,
        "toolUsePromptTokensDetails": [
            { "modality": "TEXT", "tokenCount": 12 }
        ],
        "trafficType": "PROVISIONED_THROUGHPUT"
    });

    let token_usage = usage_of(&json_data);
    // Input is the prompt plus the hosted-tool prompt, output the candidates
    // plus the thoughts, and the total their sum, as `totalTokenCount` is.
    assert_eq!(token_usage.input_tokens, Some(112));
    assert_eq!(token_usage.cached_input_tokens, Some(25));
    assert_eq!(token_usage.output_tokens, Some(65));
    assert_eq!(token_usage.reasoning_tokens, Some(15));
    assert_eq!(token_usage.tool_use_prompt_tokens, Some(12));
    assert_eq!(token_usage.total_tokens, Some(177));
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
mod terminal_emission {

    use crate::providers::gemini::GeminiConfig;
    use crate::providers::gemini::completion::{GEMINI_2_5_PRO_PREVIEW_06_05, GenerateContent};
    use crate::streaming::{Item, StreamEvent};
    use crate::test_utils::MockStreamingClient;
    use futures::StreamExt;

    /// The wire under test, unbound.
    fn wire() -> GenerateContent {
        GeminiConfig::new("test-key").completion(GEMINI_2_5_PRO_PREVIEW_06_05)
    }

    const CONTENT_CHUNK: &str = r#"{"candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"}}],"responseId":"resp-1","modelVersion":"gemini-2.5-pro"}"#;
    const TERMINAL_CHUNK: &str = r#"{"candidates":[{"content":{"parts":[{"text":"!"}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2,"totalTokenCount":7},"responseId":"resp-1","modelVersion":"gemini-2.5-pro"}"#;

    fn sse(frames: &[&str]) -> bytes::Bytes {
        bytes::Bytes::from(
            frames
                .iter()
                .map(|frame| format!("data: {frame}\n\n"))
                .collect::<String>(),
        )
    }

    /// The texts the stream yielded, whether an error item came, and what
    /// it finished with.
    async fn collect(
        sse_bytes: bytes::Bytes,
    ) -> (
        Vec<String>,
        bool,
        Result<crate::completion::CompletionResponse, crate::error::ProviderError>,
    ) {
        let model = crate::driver::Model::new(wire(), MockStreamingClient { sse_bytes });
        let stream = model
            .stream(super::streaming_request())
            .expect("stream should open");
        drain(stream).await
    }

    async fn drain(
        mut stream: crate::streaming::CompletionStream,
    ) -> (
        Vec<String>,
        bool,
        Result<crate::completion::CompletionResponse, crate::error::ProviderError>,
    ) {
        let mut texts = Vec::new();
        let mut saw_error = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(Item::Event(StreamEvent::Text { text, .. })) => texts.push(text),
                Ok(_) => {}
                Err(_) => saw_error = true,
            }
        }
        (texts, saw_error, stream.finish().await)
    }

    #[tokio::test]
    async fn a_signature_with_no_thought_text_still_emits_a_signed_block() {
        // gRPC and Interactions emit signature-only blocks; the REST
        // wire must not diverge — the signature is replay-required
        // provider state even when no thought text accumulated.
        const SIGNATURE_ONLY_CHUNK: &str = r#"{"candidates":[{"content":{"parts":[{"text":"","thought":true,"thoughtSignature":"sig-only"}],"role":"model"},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":1,"totalTokenCount":4}}"#;

        let model = crate::driver::Model::new(
            wire(),
            MockStreamingClient {
                sse_bytes: sse(&[SIGNATURE_ONLY_CHUNK]),
            },
        );
        let mut stream = model
            .stream(super::streaming_request())
            .expect("stream should open");

        let mut signed = None;
        while let Some(item) = stream.next().await {
            if let Item::Event(StreamEvent::End {
                content: crate::message::AssistantContent::Reasoning(reasoning),
                ..
            }) = item.expect("stream item should be Ok")
            {
                signed = Some(reasoning);
            }
        }
        let signed = signed.expect("signature-only block must be emitted");
        assert!(signed.text.is_empty());
        assert_eq!(
            signed
                .native
                .map(|native| native.item["thoughtSignature"].clone()),
            Some(serde_json::json!("sig-only"))
        );
    }

    #[tokio::test]
    async fn truncated_stream_yields_content_then_truncation() {
        let (texts, saw_error, finished) = collect(sse(&[CONTENT_CHUNK])).await;

        assert_eq!(texts, ["hi"]);
        assert!(saw_error, "the truncation is the last item");
        assert!(
            matches!(finished, Err(crate::error::ProviderError::Truncated)),
            "EOF without a finishReason chunk is truncation: {finished:?}"
        );
    }

    #[tokio::test]
    async fn errored_stream_forwards_the_error_and_no_end() {
        use crate::test_utils::SequencedStreamingHttpClient;

        // A transport failure after some content must reach the consumer
        // and must not be papered over with a synthesized end.
        let model = crate::driver::Model::new(
            wire(),
            SequencedStreamingHttpClient::new(vec![
                Ok(sse(&[CONTENT_CHUNK])),
                Err(crate::http_client::Error::non_success_with_details(
                    http::StatusCode::BAD_GATEWAY,
                    http::HeaderMap::new(),
                    "connection reset".to_string(),
                )),
            ]),
        );
        let stream = model
            .stream(super::streaming_request())
            .expect("stream should open");
        let (texts, saw_error, finished) = drain(stream).await;

        assert_eq!(texts, ["hi"]);
        assert!(saw_error, "the transport failure must reach the consumer");
        assert!(finished.is_err(), "a failed stream has no response");
    }

    #[tokio::test]
    async fn malformed_frame_then_eof_yields_error_and_no_end() {
        let (texts, saw_error, finished) = collect(sse(&[CONTENT_CHUNK, "{not json"])).await;

        assert_eq!(texts, ["hi"]);
        assert!(saw_error, "the malformed frame must reach the consumer");
        assert!(finished.is_err(), "a parse error is not a completed turn");
    }

    #[tokio::test]
    async fn id_only_frame_updates_terminal_metadata_without_content() {
        let (texts, saw_error, finished) = collect(sse(&[
            CONTENT_CHUNK,
            TERMINAL_CHUNK,
            r#"{"responseId":"last-response"}"#,
        ]))
        .await;
        assert_eq!(texts, ["hi", "!"]);
        assert!(!saw_error);
        assert_eq!(
            finished.expect("the reply ended").response_id(),
            Some("last-response")
        );
    }

    /// A corrupt frame ends the reply: a genuine finishReason chunk after
    /// it is never read.
    #[tokio::test]
    async fn a_malformed_frame_ends_the_reply_before_a_later_end() {
        let (texts, saw_error, finished) =
            collect(sse(&[CONTENT_CHUNK, "{not json", TERMINAL_CHUNK])).await;

        assert_eq!(texts, ["hi"]);
        assert!(saw_error, "the malformed frame must reach the consumer");
        assert!(finished.is_err(), "the corrupt frame ended the reply");
    }

    /// An undelivered reply without a finish reason is truncated however it
    /// arrived. With one, it is an empty turn carrying that reason and the
    /// usage the reply reported: the cap and the filter cut it short, and a
    /// turn that ran to completion (`STOP`) and delivered nothing is an
    /// empty success.
    #[tokio::test]
    async fn an_undelivered_reply_is_truncated_unless_it_names_a_finish_reason() {
        use crate::completion::FinishReason;
        use crate::test_utils::RecordingHttpClient;

        const SILENT: &str = r#"{"candidates":[],"usageMetadata":{"promptTokenCount":9}}"#;

        for (reason, normalized) in [
            ("MAX_TOKENS", FinishReason::Length),
            ("SAFETY", FinishReason::ContentFilter),
        ] {
            let body = format!(
                r#"{{"candidates":[{{"finishReason":"{reason}","index":0}}],"usageMetadata":{{"promptTokenCount":9,"candidatesTokenCount":32,"totalTokenCount":41}}}}"#
            );
            let model = crate::driver::Model::new(wire(), RecordingHttpClient::new(body));
            let response = model
                .call(super::streaming_request())
                .await
                .expect("a terminal that cut the turn short legalizes an empty choice");
            assert!(response.choice.is_empty());
            assert_eq!(response.finish_reason(), Some(normalized));
            assert_eq!(response.usage.output_tokens, Some(32));
        }

        let completed = crate::driver::Model::new(
            wire(),
            RecordingHttpClient::new(
                r#"{"candidates":[{"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":9}}"#,
            ),
        );
        let response = completed
            .call(super::streaming_request())
            .await
            .expect("an empty successful reply is a success");
        assert!(response.choice.is_empty());
        assert_eq!(response.finish_reason(), Some(FinishReason::Stop));

        let silent = crate::driver::Model::new(wire(), RecordingHttpClient::new(SILENT));
        let error = silent
            .call(super::streaming_request())
            .await
            .expect_err("a whole reply that named no finish reason");
        assert!(
            matches!(error, crate::error::ProviderError::Truncated),
            "{error}"
        );

        // The same undelivered bytes on the streamed path end the same way.
        let (texts, _, finished) = collect(sse(&[SILENT])).await;
        assert!(texts.is_empty());
        assert!(matches!(
            finished,
            Err(crate::error::ProviderError::Truncated)
        ));
    }
}

/// Open a `streamGenerateContent` stream over the given SSE frames and
/// collect every item the consumer sees, and whether the reply ended.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
async fn collect_stream(
    frames: &[&str],
) -> (
    Vec<Result<crate::streaming::Item<crate::streaming::StreamEvent>, crate::error::ErrorReport>>,
    bool,
) {
    use futures::StreamExt;

    let model = streamed("gemini-2.5-flash", frames);
    let mut stream = model
        .stream(streaming_request())
        .expect("stream should open");
    let mut items = Vec::new();
    while let Some(item) = stream.next().await {
        items.push(item.map_err(|error| crate::error::ErrorReport::from(&error)));
    }
    let finished = stream.finish().await.is_ok();
    (items, finished)
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn blocked_prompt_is_a_provider_error_naming_the_block_reason() {
    // Gemini answers a blocked prompt on the streaming wire with one chunk
    // that carries `promptFeedback.blockReason` and no candidates, then
    // closes the stream. That is the provider's definitive verdict, not a
    // truncation: the consumer must get an error naming the reason (and the
    // safety ratings that explain it), never a stream that simply ends
    // before its terminal record.
    let frames = [
        r#"{"promptFeedback":{"blockReason":"PROHIBITED_CONTENT","safetyRatings":[{"category":"HARM_CATEGORY_DANGEROUS_CONTENT","probability":"HIGH"}]},"usageMetadata":{"promptTokenCount":12,"totalTokenCount":12},"modelVersion":"gemini-2.5-flash","responseId":"abc123"}"#,
    ];
    let (items, finished) = collect_stream(&frames).await;
    assert_eq!(items.len(), 1, "exactly the refusal: {items:?}");
    let Err(report) = &items[0] else {
        panic!("expected a provider error, got {:?}", items[0]);
    };
    assert_eq!(
        report.kind,
        crate::error::ErrorKind::ProviderResponse,
        "{report:?}"
    );
    assert!(
        report.refusal,
        "the block is a refusal on the report: {report:?}"
    );
    assert_eq!(
        report.code.as_deref(),
        Some("PROHIBITED_CONTENT"),
        "{report:?}"
    );
    assert!(!report.retryable, "a refusal is not retryable: {report:?}");
    let message = &report.message;
    assert!(message.contains("blocked the prompt"), "{message}");
    assert!(message.contains("PROHIBITED_CONTENT"), "{message}");
    assert!(
        message.contains("HARM_CATEGORY_DANGEROUS_CONTENT"),
        "{message}"
    );
    assert!(message.contains("HIGH"), "{message}");
    assert!(!finished, "a blocked prompt has no terminal record");
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn blocked_prompt_without_usage_is_still_recognised_and_ends_the_stream() {
    // The block chunk may carry no `usageMetadata` at all (proto3 JSON omits
    // default-valued fields). It must still decode as the wire's chunk shape
    // rather than pass through as an unknown frame, and it ends the turn:
    // nothing after it is interpreted.
    let frames = [
        r#"{"promptFeedback":{"blockReason":"SAFETY"}}"#,
        r#"{"candidates":[{"content":{"parts":[{"text":"dead"}],"role":"model"},"finishReason":"STOP","index":0}]}"#,
    ];
    let (items, finished) = collect_stream(&frames).await;
    assert_eq!(items.len(), 1, "exactly the refusal: {items:?}");
    assert!(
        matches!(&items[0], Err(report) if report.kind == crate::error::ErrorKind::ProviderResponse && report.refusal && report.message.contains("SAFETY")),
        "{:?}",
        items[0]
    );
    assert!(!finished);
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn prompt_feedback_without_a_block_reason_is_not_an_error() {
    // Gemini also attaches `promptFeedback` (safety ratings only, no
    // `blockReason`) to ordinary answers. Only a set `blockReason` is a
    // refusal; the ratings alone must not fail a completed turn.
    let frames = [
        r#"{"promptFeedback":{"safetyRatings":[{"category":"HARM_CATEGORY_HARASSMENT","probability":"NEGLIGIBLE"}]},"candidates":[{"content":{"parts":[{"text":"hi"}],"role":"model"},"index":0}]}"#,
        r#"{"candidates":[{"content":{"parts":[],"role":"model"},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1,"totalTokenCount":2}}"#,
    ];
    let (items, finished) = collect_stream(&frames).await;
    assert!(items.iter().all(Result::is_ok), "{items:?}");
    assert!(finished, "the turn completed normally");
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn blocked_prompt_with_an_unrecognised_safety_category_still_names_the_block() {
    // Google adds harm categories without notice. The block chunk carries
    // the ratings, so an unknown category must not turn the provider's
    // verdict into a corrupt-frame decode error.
    let frames = [
        r#"{"promptFeedback":{"blockReason":"OTHER","safetyRatings":[{"category":"HARM_CATEGORY_JAILBREAK","probability":"SOMEDAY"}]}}"#,
    ];
    let (items, finished) = collect_stream(&frames).await;
    assert_eq!(items.len(), 1, "{items:?}");
    let Err(report) = &items[0] else {
        panic!("{:?}", items[0]);
    };
    // `OTHER` is not a verdict on the content: it is retryable, and it
    // reaches the consumer as a provider response, not a refusal.
    assert_eq!(
        report.kind,
        crate::error::ErrorKind::ProviderResponse,
        "{report:?}"
    );
    assert!(report.retryable, "an OTHER block is transient: {report:?}");
    assert_eq!(report.code.as_deref(), Some("OTHER"));
    assert!(
        report.message.contains("block_reason=OTHER"),
        "{}",
        report.message
    );
    assert!(
        report.message.contains("HARM_CATEGORY_JAILBREAK=SOMEDAY"),
        "{}",
        report.message
    );
    assert!(!finished);
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn in_band_http_errors_match_unary_classification_and_preserve_the_envelope() {
    for code in [400, 401, 404, 429, 500, 503] {
        let body = json!({"error": {"code": code, "message": "Keep CASE", "status": "VERDICT"}})
            .to_string();
        let (items, finished) = collect_stream(&[&body]).await;
        let errors: Vec<_> = items
            .iter()
            .filter_map(|item| item.as_ref().err())
            .collect();
        assert_eq!(errors.len(), 1, "{items:?}");
        assert!(!finished);
        let unary = crate::error::ErrorReport::from(ProviderError::from_http_response(
            http::StatusCode::from_u16(code).expect("HTTP error status"),
            body.clone(),
        ));
        assert_eq!(errors[0].kind, unary.kind);
        assert_eq!(errors[0].http_status, unary.http_status);
        assert_eq!(errors[0].is_retryable(), unary.is_retryable());
        assert_eq!(errors[0].provider_response_body(), Some(body.as_str()));
    }
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn in_band_opaque_or_invalid_codes_do_not_invent_http_status() {
    for code in [
        json!(null),
        json!("503"),
        json!(14),
        json!(-1),
        json!(200),
        json!(700),
        json!(65536),
    ] {
        let body = json!({"error": {"code": code, "message": "unknown"}}).to_string();
        let (items, finished) = collect_stream(&[&body]).await;
        let errors: Vec<_> = items
            .iter()
            .filter_map(|item| item.as_ref().err())
            .collect();
        assert_eq!(errors.len(), 1, "{items:?}");
        assert!(!finished);
        assert_eq!(errors[0].kind, crate::error::ErrorKind::ProviderResponse);
        assert_eq!(errors[0].http_status, None);
        assert!(!errors[0].is_retryable());
        assert_eq!(errors[0].provider_response_body(), Some(body.as_str()));
    }
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn each_block_reason_classifies_the_same_on_the_stream_as_unary() {
    // The refusal reasons are final; `OTHER` and a reason this crate has
    // never seen are transient. Both wires agree.
    for (reason, retryable) in [
        ("SAFETY", false),
        ("BLOCKLIST", false),
        ("PROHIBITED_CONTENT", false),
        ("OTHER", true),
        ("SOMETHING_NEW", true),
    ] {
        let frame = format!(r#"{{"promptFeedback":{{"blockReason":"{reason}"}}}}"#);
        let (items, finished) = collect_stream(&[frame.as_str()]).await;
        assert_eq!(items.len(), 1, "{reason}: {items:?}");
        let Err(report) = &items[0] else {
            panic!("{reason}: {:?}", items[0]);
        };
        assert_eq!(report.retryable, retryable, "{reason}: {report:?}");
        assert_eq!(
            report.kind,
            crate::error::ErrorKind::ProviderResponse,
            "{reason}: {report:?}"
        );
        assert_eq!(
            report.refusal, !retryable,
            "a block on the content is a refusal, an unknown one is not: {reason}: {report:?}"
        );
        assert!(
            report.message.contains(&format!("block_reason={reason}")),
            "{reason}: {}",
            report.message
        );
        assert!(
            !finished,
            "{reason}: a blocked prompt has no terminal record"
        );
    }
}

/// A part becomes a provider item only at its end: the next part, or the
/// reply's finish. A text part the reply was cut off in has no item, so it
/// can never replay as a provider part; the part before it, which a later
/// part ended, keeps its own.
#[test]
fn a_part_cut_off_before_its_end_keeps_no_provider_item() {
    use crate::test_utils::history_conformance::partial;
    let wire = crate::providers::gemini::GeminiConfig::new("test-key")
        .completion("gemini-3-flash-preview");
    let frames = [
        r#"{"candidates":[{"content":{"parts":[{"text":"plan","thought":true,"thoughtSignature":"c2ln"}],"role":"model"}}]}"#,
        r#"{"candidates":[{"content":{"parts":[{"text":"the answer is"}],"role":"model"}}]}"#,
    ]
    .map(|frame| crate::wire::WireFrame::Text(frame.to_owned()));
    let (response, ended) = partial(&wire, crate::wire::Mode::Streaming, frames);
    assert!(!ended, "no finish arrived");
    let [
        AssistantContent::Reasoning(plan),
        AssistantContent::Text(answer),
    ] = response.choice.as_slice()
    else {
        panic!("the ended thought and the open text: {:?}", response.choice);
    };
    assert_eq!(
        plan.native.as_ref().map(|native| &native.item),
        Some(&json!({"text": "plan", "thought": true, "thoughtSignature": "c2ln"}))
    );
    assert_eq!(answer.text, "the answer is");
    assert!(answer.native.is_none(), "{answer:?}");
    assert!(response.stop().is_failure());
}

/// Round 4 NEW-3: a part that carries only a thought signature, as a
/// stream's last chunk can, joins the text before it, so the signature
/// replays to the same model.
#[test]
fn a_signature_only_part_joins_the_text_before_it() {
    use crate::wire::{Operation, Wire};
    let wire =
        crate::providers::gemini::GeminiConfig::new("k").completion("gemini-3-flash-preview");
    let chunk = |parts: serde_json::Value, finish: Option<&str>| {
        let mut candidate = json!({"content": {"role": "model", "parts": parts}});
        if let Some(finish) = finish {
            candidate["finishReason"] = json!(finish);
        }
        WireFrame::Text(json!({"candidates": [candidate], "modelVersion": "m"}).to_string())
    };
    let frames = [
        chunk(json!([{"text": "Answer"}]), None),
        chunk(json!([{"thoughtSignature": "c2ln"}]), Some("STOP")),
    ];
    let response = crate::test_utils::history::decode(&wire, crate::wire::Mode::Streaming, frames)
        .expect("the reply decodes");
    assert_eq!(response.choice.len(), 1, "{:?}", response.choice);
    let history = vec![
        crate::message::Message::user("q"),
        response.message().expect("a turn"),
        crate::message::Message::user("n"),
    ];
    let request = crate::operation::Completion::prepare(
        crate::completion::CompletionRequest::from(history),
        &wire.describe(),
    )
    .expect("the history prepares");
    let encoded = wire
        .encode(request, crate::wire::Mode::Unary)
        .expect("the history encodes");
    let body = crate::test_utils::json_body(&encoded.request);
    assert_eq!(
        body["contents"][1]["parts"],
        json!([{"text": "Answer", "thoughtSignature": "c2ln"}]),
        "{body}"
    );
}

/// The request body the REST wire sends `model` to continue `response`.
fn replayed(model: &str, response: &crate::completion::CompletionResponse) -> serde_json::Value {
    use crate::wire::{Operation, Wire};
    let wire = crate::providers::gemini::GeminiConfig::new("k").completion(model);
    let mut history = vec![
        crate::message::Message::user("q"),
        response.message().expect("a turn"),
    ];
    let calls: Vec<_> = response.tool_calls().cloned().collect();
    let mut request = if calls.is_empty() {
        history.push(crate::message::Message::user("n"));
        crate::completion::CompletionRequest::from(history)
    } else {
        history.push(crate::message::Message::User {
            content: calls
                .iter()
                .map(|call| {
                    crate::message::UserContent::ToolResult(
                        call.result(vec![crate::message::ToolResultContent::text("ok")]),
                    )
                })
                .collect(),
        });
        crate::completion::CompletionRequest::from(history)
    };
    request.tools = calls
        .iter()
        .map(|call| crate::completion::ToolDefinition {
            name: call.function.name.clone(),
            description: "a tool".to_owned(),
            parameters: json!({"type": "object"}),
        })
        .collect();
    let request = crate::operation::Completion::prepare(request, &wire.describe())
        .expect("the history prepares");
    let encoded = wire
        .encode(request, crate::wire::Mode::Unary)
        .expect("the history encodes");
    crate::test_utils::json_body(&encoded.request)
}

/// A signature sent alone signs the block beside it whatever its kind: the
/// call before it, or the text after it when it comes first.
#[test]
fn a_signature_only_part_signs_the_block_beside_it() {
    let wire =
        crate::providers::gemini::GeminiConfig::new("k").completion("gemini-3-flash-preview");
    let chunk = |parts: serde_json::Value, finish: Option<&str>| {
        let mut candidate = json!({"content": {"role": "model", "parts": parts}});
        if let Some(finish) = finish {
            candidate["finishReason"] = json!(finish);
        }
        WireFrame::Text(json!({"candidates": [candidate], "modelVersion": "m"}).to_string())
    };
    let after_call = crate::test_utils::history::decode(
        &wire,
        crate::wire::Mode::Streaming,
        [
            chunk(
                json!([{"functionCall": {"name": "lookup", "args": {"q": 1}, "id": "a"}}]),
                None,
            ),
            chunk(json!([{"thoughtSignature": "c2lnLWNhbGw="}]), Some("STOP")),
        ],
    )
    .expect("the reply decodes");
    let body = replayed("gemini-3-flash-preview", &after_call);
    assert_eq!(
        body["contents"][1]["parts"][0]["thoughtSignature"], "c2lnLWNhbGw=",
        "{body}"
    );
    let leading = crate::test_utils::history::decode(
        &wire,
        crate::wire::Mode::Unary,
        [chunk(
            json!([{"thoughtSignature": "c2lnLWxlYWQ="}, {"text": "Answer"}]),
            Some("STOP"),
        )],
    )
    .expect("the reply decodes");
    let body = replayed("gemini-3-flash-preview", &leading);
    assert_eq!(
        body["contents"][1]["parts"],
        json!([{"text": "Answer", "thoughtSignature": "c2lnLWxlYWQ="}]),
        "{body}"
    );
}

/// A reply part that is not an object is no part Gemini takes back, so it
/// never replays.
#[test]
fn a_part_that_is_not_an_object_never_replays() {
    for odd in [json!(null), json!("stray")] {
        let response = decoded(json!({
            "candidates": [{
                "content": {"role": "model", "parts": [odd.clone(), {"text": "hi"}]},
                "finishReason": "STOP"
            }]
        }))
        .expect("the reply decodes");
        let body = replayed("gemini-2.5-flash", &response);
        assert_eq!(
            body["contents"][1]["parts"],
            json!([{"text": "hi"}]),
            "{odd}: {body}"
        );
    }
}
