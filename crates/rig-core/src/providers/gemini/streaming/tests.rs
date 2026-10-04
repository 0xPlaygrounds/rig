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

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
mod terminal_emission {}

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

/// Two thought parts with their own signatures stay two blocks, each with
/// its signature: merged, the first signature would be lost.
#[test]
fn two_signed_thought_parts_keep_both_signatures() {
    let response = decoded(json!({
        "candidates": [{"content": {"role": "model", "parts": [
            {"thought": true, "text": "one", "thoughtSignature": "b25l"},
            {"thought": true, "text": "two", "thoughtSignature": "dHdv"},
            {"text": "answer"}
        ]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2}
    }))
    .expect("the reply decodes");
    let signatures: Vec<_> = response
        .choice
        .iter()
        .filter_map(|block| block.native_item())
        .filter_map(|item| item["thoughtSignature"].as_str())
        .collect();
    assert_eq!(signatures, ["b25l", "dHdv"], "{:?}", response.choice);
}
