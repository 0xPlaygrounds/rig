//! The chat wire, driven from recorded bytes.
//!
//! The property the whole model exists for is that a unary reply and a
//! streamed reply of the *same turn* fold to the same response. These tests
//! assert it against real recorded bodies: the pairs under
//! `tests/cassettes/openai/raw_capture_matrix/**` and
//! `tests/cassettes/openai/raw_stream_capture_matrix/**` are the same prompt
//! answered both ways, so there is nothing hand-written for the two paths to
//! agree about by accident.

use bytes::Bytes;
use futures::StreamExt;

use super::*;
use crate::completion::{CompletionModel as _, FinishReason};
use crate::driver::Bound;
use crate::message::AssistantContent;
use crate::providers::openai::wire::{GROQ, OPENAI, OpenAI};
use crate::test_utils::{MockStreamingClient, RecordingHttpClient};

use super::super::tests::{recorded, recorded_json};

fn wire() -> Chat {
    OpenAI::new("sk-test")
        .with_dialect(&OPENAI)
        .chat("gpt-4.1-nano")
}

fn prompt(text: &str) -> CompletionRequest {
    CompletionRequest {
        model: None,
        chat_history: vec![crate::message::Message::user(text)],
        documents: Vec::new(),
        tools: Vec::new(),
        temperature: Some(0.0),
        max_tokens: Some(16),
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    }
}

/// Fold both recorded shapes of one turn and hand back the two responses.
async fn fold_both(
    unary_cassette: &str,
    stream_cassette: &str,
    request: CompletionRequest,
) -> (
    crate::completion::CompletionResponse,
    crate::completion::CompletionResponse,
) {
    let buffered = Bound::new(
        wire(),
        RecordingHttpClient::new(recorded("then", unary_cassette)),
    )
    .completion(request.clone())
    .await
    .expect("the recorded unary reply decodes");

    let streaming = Bound::new(
        wire(),
        MockStreamingClient {
            sse_bytes: Bytes::from(recorded("then", stream_cassette)),
        },
    );
    let mut response = streaming
        .stream(request)
        .await
        .expect("the recorded stream opens");
    while response.next().await.is_some() {}
    (buffered, response.finish())
}

#[tokio::test]
async fn a_recorded_text_turn_folds_alike_from_both_reply_shapes() {
    let (buffered, streamed) = fold_both(
        "raw_capture_matrix/chat_raw_round_trips_typed.yaml",
        "raw_stream_capture_matrix/chat_stream_raw_round_trips_typed.yaml",
        prompt("Reply with exactly the single word: pong"),
    )
    .await;

    assert_eq!(buffered.choice, streamed.choice);
    assert_eq!(buffered.usage, streamed.usage);
    assert_eq!(buffered.finish_reason(), streamed.finish_reason());
    assert_eq!(buffered.model, streamed.model);

    // The fixture, so a change in the recorded bytes cannot make the
    // agreement vacuous.
    assert_eq!(
        buffered.choice.first(),
        Some(&AssistantContent::text("pong"))
    );
    assert_eq!(buffered.usage.input_tokens, Some(15));
    assert_eq!(buffered.usage.output_tokens, Some(1));
    assert_eq!(buffered.usage.total_tokens, Some(16));
    assert_eq!(buffered.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(buffered.model.as_deref(), Some("gpt-4.1-nano-2025-04-14"));
}

#[tokio::test]
async fn a_recorded_tool_call_turn_folds_alike_from_both_reply_shapes() {
    let mut request = prompt("Call ping exactly once with no arguments.");
    request.max_tokens = Some(64);
    let (buffered, streamed) = fold_both(
        "raw_capture_matrix/chat_tool_call_raw_round_trips_typed.yaml",
        "raw_stream_capture_matrix/chat_tool_call_stream_raw_round_trips_typed.yaml",
        request,
    )
    .await;

    assert_eq!(buffered.choice, streamed.choice);
    assert_eq!(buffered.usage, streamed.usage);
    assert_eq!(buffered.finish_reason(), streamed.finish_reason());

    // The unary body delivers the call whole and the stream delivers it in
    // two fragments, so equality here is the synthesis working rather than
    // two copies of one mapping agreeing.
    let Some(AssistantContent::ToolCall(call)) = buffered.choice.first() else {
        panic!("the recorded turn is a tool call: {:?}", buffered.choice);
    };
    assert_eq!(call.function.name, "ping");
    assert_eq!(call.function.arguments, serde_json::json!({}));
    assert_eq!(call.id.explicit(), Some("call_REDACTED_1"));
    assert_eq!(buffered.finish_reason(), Some(FinishReason::ToolCalls));
    assert_eq!(buffered.usage.output_tokens, Some(10));
}

/// A streamed request genuinely sends different bytes, which is why `encode`
/// takes the mode; the cassettes pin both.
#[test]
fn the_mode_decides_whether_the_body_asks_for_a_stream() {
    fn body(mode: Mode) -> serde_json::Value {
        let encoded = wire()
            .encode(prompt("Reply with exactly the single word: pong"), mode)
            .expect("the request encodes");
        let [request] = encoded.requests.as_slice() else {
            panic!("a chat turn is one request");
        };
        let Body::Bytes(bytes) = request.body() else {
            panic!("a chat request body is bytes");
        };
        serde_json::from_slice(bytes).expect("the body is JSON")
    }

    let unary = body(Mode::Unary);
    assert!(unary.get("stream").is_none());
    assert!(unary.get("stream_options").is_none());

    let streaming = body(Mode::Streaming);
    assert_eq!(streaming.get("stream"), Some(&serde_json::json!(true)));
    assert_eq!(
        streaming.get("stream_options"),
        Some(&serde_json::json!({"include_usage": true}))
    );

    // Both are the bytes the cassettes recorded for this turn.
    assert_eq!(
        unary,
        recorded_json("when", "raw_capture_matrix/chat_raw_round_trips_typed.yaml")
    );
    assert_eq!(
        streaming,
        recorded_json(
            "when",
            "raw_stream_capture_matrix/chat_stream_raw_round_trips_typed.yaml",
        )
    );
}

/// A streamed reply's framing is SSE and a unary reply's is the whole body,
/// which is what lets the driver reject a non-event-stream 200 on a stream
/// while reading the unary reply as one JSON document.
#[test]
fn the_mode_decides_the_reply_framing() {
    let request = prompt("hi");
    assert_eq!(
        wire()
            .encode(request.clone(), Mode::Unary)
            .expect("encodes")
            .framing,
        Framing::Whole
    );
    assert_eq!(
        wire().encode(request, Mode::Streaming).expect("encodes").framing,
        Framing::Sse
    );
}

/// `[DONE]` is a modeled event, not a transport filter: a stream whose only
/// terminal signal is the sentinel still produces the terminal record.
#[tokio::test]
async fn the_done_sentinel_emits_the_deferred_terminal() {
    const BODY: &str = concat!(
        "data: {\"object\":\"chat.completion.chunk\",\"id\":\"chatcmpl-1\",",
        "\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n",
        "data: [DONE]\n\n",
    );
    let bound = Bound::new(
        wire(),
        MockStreamingClient {
            sse_bytes: Bytes::from_static(BODY.as_bytes()),
        },
    );
    let mut response = bound.stream(prompt("hi")).await.expect("the stream opens");
    while response.next().await.is_some() {}
    let folded = response.finish();

    assert_eq!(folded.choice.first(), Some(&AssistantContent::text("hi")));
    // No chunk reported a reason, so the record carries none — the sentinel
    // is the completion signal, not a fabricated `stop`.
    assert_eq!(folded.finish_reason(), None);
    assert_eq!(folded.response_id.as_deref(), Some("chatcmpl-1"));
}

/// A stream that reaches EOF with neither `[DONE]` nor a finish reason is
/// truncation, and must not be dressed up as a default-usage success.
#[tokio::test]
async fn a_truncated_stream_yields_no_terminal_record() {
    const BODY: &str = concat!(
        "data: {\"object\":\"chat.completion.chunk\",\"id\":\"chatcmpl-1\",",
        "\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"par\"}}]}\n\n",
    );
    let bound = Bound::new(
        wire(),
        MockStreamingClient {
            sse_bytes: Bytes::from_static(BODY.as_bytes()),
        },
    );
    let mut response = bound.stream(prompt("hi")).await.expect("the stream opens");
    while response.next().await.is_some() {}
    let folded = response.finish();
    assert_eq!(folded.usage, crate::completion::Usage::default());
    assert_eq!(folded.finish_reason(), None);
}

/// The wire's in-band error envelope arrives with a 200 status, so only the
/// decoder can see it: it must fail the turn rather than read as a chunk.
#[tokio::test]
async fn an_in_band_error_envelope_fails_the_turn() {
    const BODY: &str =
        "data: {\"error\":{\"message\":\"rate limited\"},\"choices\":[]}\n\ndata: [DONE]\n\n";
    let bound = Bound::new(
        wire(),
        MockStreamingClient {
            sse_bytes: Bytes::from_static(BODY.as_bytes()),
        },
    );
    let mut response = bound.stream(prompt("hi")).await.expect("the stream opens");
    let mut errors = Vec::new();
    while let Some(item) = response.next().await {
        if let Err(error) = item {
            errors.push(error.to_string());
        }
    }
    assert_eq!(errors.len(), 1, "one terminal failure: {errors:?}");
    assert!(
        errors[0].contains("rate limited"),
        "the provider's message survives: {errors:?}"
    );
    // A failed turn commits no terminal record, so nothing reports usage.
    assert_eq!(response.finish().finish_reason(), None);
}

/// A dialect that spells the cap `max_completion_tokens` does so only for
/// the reasoning families, because this same wire reaches compatible servers
/// that know only the legacy field.
#[test]
fn the_output_cap_spelling_follows_the_model_family() {
    fn cap_key(model: &str) -> &'static str {
        let encoded = OpenAI::new("sk-test")
            .chat(model)
            .encode(prompt("hi"), Mode::Unary)
            .expect("encodes");
        let [request] = encoded.requests.as_slice() else {
            panic!("a chat turn is one request");
        };
        let Body::Bytes(bytes) = request.body() else {
            panic!("a chat request body is bytes");
        };
        let body: serde_json::Value = serde_json::from_slice(bytes).expect("JSON");
        if body.get("max_completion_tokens").is_some() {
            "max_completion_tokens"
        } else if body.get("max_tokens").is_some() {
            "max_tokens"
        } else {
            "none"
        }
    }

    assert_eq!(cap_key("gpt-5.2"), "max_completion_tokens");
    assert_eq!(cap_key("o3-mini"), "max_completion_tokens");
    assert_eq!(cap_key("gpt-4.1-nano"), "max_tokens");

    // A dialect whose endpoint was never observed to reject the legacy field
    // keeps sending it, whatever the model is called.
    let groq = OpenAI::new("gsk-test")
        .with_dialect(&GROQ)
        .chat("gpt-5.2")
        .encode(prompt("hi"), Mode::Unary)
        .expect("encodes");
    let [groq_request] = groq.requests.as_slice() else {
        panic!("a chat turn is one request");
    };
    let Body::Bytes(bytes) = groq_request.body() else {
        panic!("a chat request body is bytes");
    };
    let body: serde_json::Value = serde_json::from_slice(bytes).expect("JSON");
    assert!(body.get("max_tokens").is_some());
    assert!(body.get("max_completion_tokens").is_none());
}
