//! The chat wire, driven from recorded bytes.
//!
//! The property the whole model exists for is that a unary reply and a
//! streamed reply of the *same turn* fold to the same response. These tests
//! assert it against real recorded bodies: the pairs under
//! `crates/rig-cassette/fixtures/cassettes/openai/raw_capture_matrix/**` and
//! `crates/rig-cassette/fixtures/cassettes/openai/raw_stream_capture_matrix/**` are the same prompt
//! answered both ways, so there is nothing hand-written for the two paths to
//! agree about by accident.

use bytes::Bytes;
use futures::StreamExt;

use super::*;
use crate::completion::{CompletionModel as _, FinishReason};
use crate::driver::Bound;
use crate::message::AssistantContent;
use crate::providers::openai::wire::{Dialect, GROQ, OPENAI, OpenAI};
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
    (
        buffered,
        response
            .finish()
            .expect("the stream produced a terminal record"),
    )
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

    assert_eq!(buffered.usage, streamed.usage);
    assert_eq!(buffered.finish_reason(), streamed.finish_reason());

    // The unary body delivers the call whole and the stream delivers it in
    // two fragments, so equality here is the synthesis working rather than
    // two copies of one mapping agreeing. Each recording mints its own call
    // id, so the ids are checked against their own bytes instead.
    let ([AssistantContent::ToolCall(call)], [AssistantContent::ToolCall(streamed_call)]) =
        (buffered.choice.as_slice(), streamed.choice.as_slice())
    else {
        panic!(
            "each recorded turn is one tool call: {:?} / {:?}",
            buffered.choice, streamed.choice
        );
    };
    assert_eq!(call.function, streamed_call.function);
    assert_eq!(call.signature, streamed_call.signature);
    assert_eq!(
        call.provider.as_ref().map(|provider| &provider.item_id),
        streamed_call
            .provider
            .as_ref()
            .map(|provider| &provider.item_id)
    );
    assert_eq!(call.additional_params, streamed_call.additional_params);
    for (call, cassette) in [
        (
            call,
            "raw_capture_matrix/chat_tool_call_raw_round_trips_typed.yaml",
        ),
        (
            streamed_call,
            "raw_stream_capture_matrix/chat_tool_call_stream_raw_round_trips_typed.yaml",
        ),
    ] {
        let id = call
            .id
            .explicit()
            .expect("the call keeps the provider's id");
        assert!(
            recorded("then", cassette).contains(&format!("\"id\":\"{id}\"")),
            "{cassette}: the call id {id} is the recorded one"
        );
        assert_eq!(
            call.provider
                .as_ref()
                .map(|provider| provider.call_id.as_str()),
            Some(id)
        );
    }
    assert_eq!(call.function.name, "ping");
    assert_eq!(call.function.arguments, serde_json::json!({}));
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
        wire()
            .encode(request, Mode::Streaming)
            .expect("encodes")
            .framing,
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
    let folded = response
        .finish()
        .expect("the stream produced a terminal record");

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
    assert!(
        response.response.is_none(),
        "no terminal record is synthesized"
    );
    let error = response
        .finish()
        .expect_err("a stream without a terminal record has no response to fold");
    assert!(error.to_string().contains("truncated"), "{error}");
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
    // A failed turn commits no terminal record, so there is nothing to fold.
    assert!(response.response.is_none());
    assert!(response.finish().is_err());
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

/// OpenRouter's message conversion refused a document carrying only a
/// provider file id. A refusal is behaviour: dropping it would turn a legible
/// local error into an opaque gateway 400.
#[test]
fn openrouter_refuses_a_document_that_is_only_a_file_id() {
    use crate::message::{Document, DocumentSourceKind, Message, UserContent};
    use crate::providers::openai::wire::OPENROUTER;

    let with_file_id = || {
        let mut request = prompt("read this");
        request.chat_history = vec![Message::User {
            content: vec![UserContent::Document(Document {
                data: DocumentSourceKind::FileId("file-abc".to_owned()),
                media_type: None,
                additional_params: None,
            })],
        }];
        request
    };

    let error = OpenAI::new("k")
        .with_dialect(&OPENROUTER)
        .chat("openai/gpt-4o")
        .encode(with_file_id(), Mode::Unary)
        .expect_err("OpenRouter refuses a bare file id");
    assert!(
        error
            .to_string()
            .contains("Provider file IDs are not supported for OpenRouter document inputs"),
        "the message is the one the conversion returned: {error}"
    );

    // Every other dialect on this wire accepted them, so the refusal is
    // OpenRouter's and not the wire's.
    assert!(
        OpenAI::new("k")
            .chat("gpt-4.1-nano")
            .encode(with_file_id(), Mode::Unary)
            .is_ok(),
        "only OpenRouter refuses a file id"
    );
}

/// The terminal record is readable back out of the stream's `raw`, which is
/// the escape hatch for every provider field this wire does not normalize.
#[tokio::test]
async fn the_streamed_terminal_reads_back_as_the_provider_record() {
    use crate::providers::openai::wire::{ChatUsage, StreamingCompletionResponse};

    let mut response = Bound::new(
        wire(),
        MockStreamingClient {
            sse_bytes: Bytes::from(recorded(
                "then",
                "raw_stream_capture_matrix/chat_stream_raw_round_trips_typed.yaml",
            )),
        },
    )
    .stream(prompt("Reply with exactly the single word: pong"))
    .await
    .expect("the stream opens");
    while response.next().await.is_some() {}
    let folded = response
        .finish()
        .expect("the stream produced a terminal record");

    // `serde_json::from_value`, not `Type::deserialize` — the latter needs
    // `serde::Deserialize` in scope at the call site, which is the trait-bound
    // error this spelling avoids.
    let record: StreamingCompletionResponse<ChatUsage> =
        serde_json::from_value(folded.raw.clone()).expect("the terminal record reads back");

    let usage = record.usage.expect("the stream carried usage");
    assert_eq!(usage.openai.prompt_tokens, 15);
    assert_eq!(usage.openai.completion_tokens, Some(1));
    assert_eq!(usage.openai.total_tokens, 16);
    assert_eq!(record.model.as_deref(), Some("gpt-4.1-nano-2025-04-14"));
    assert_eq!(
        record.response_id.as_deref(),
        Some(recorded_chunk_field("id").as_str())
    );

    // The provider-native fields the wire does not normalize reach the caller
    // here, which is why `raw` is the record and not the parse.
    let additional = record
        .additional_params
        .expect("the chunks carried provider metadata");
    let additional = serde_json::Value::from(additional);
    assert_eq!(additional["service_tier"], "default");
    assert_eq!(
        additional["system_fingerprint"],
        recorded_chunk_field("system_fingerprint")
    );

    // And the normalized view agrees with it.
    assert_eq!(folded.usage.input_tokens, Some(15));
    assert_eq!(folded.model.as_deref(), Some("gpt-4.1-nano-2025-04-14"));
}

/// A unary reply carrying `reasoning_content` beside text must fold through
/// the same reasoning lifecycle the streamed path uses.
///
/// Open-coding the emission in the unary branch minted a reasoning part and
/// then interleaved text into it with no derived boundary end, which trips
/// the sequence law in debug builds — and it was the unary/stream drift this
/// model exists to delete, reappearing in the one branch that bypassed the
/// shared emitter.
#[tokio::test]
async fn a_unary_reply_with_reasoning_folds_like_the_stream_of_the_same_turn() {
    use crate::message::Reasoning;

    const UNARY: &str = concat!(
        r#"{"object":"chat.completion","id":"chatcmpl-1","model":"m","#,
        r#""choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","#,
        r#""reasoning_content":"let me think","content":"the answer"}}],"#,
        r#""usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12}}"#,
    );
    const STREAM: &str = concat!(
        "data: {\"object\":\"chat.completion.chunk\",\"id\":\"chatcmpl-1\",\"model\":\"m\",",
        "\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"let me think\"}}]}\n\n",
        "data: {\"object\":\"chat.completion.chunk\",\"id\":\"chatcmpl-1\",\"model\":\"m\",",
        "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"the answer\"}}]}\n\n",
        "data: {\"object\":\"chat.completion.chunk\",\"id\":\"chatcmpl-1\",\"model\":\"m\",",
        "\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],",
        "\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":7,\"total_tokens\":12}}\n\n",
        "data: [DONE]\n\n",
    );

    let buffered = Bound::new(wire(), RecordingHttpClient::new(UNARY))
        .completion(prompt("think then answer"))
        .await
        .expect("the unary reply decodes without tripping the sequence law");

    let streaming = Bound::new(
        wire(),
        MockStreamingClient {
            sse_bytes: Bytes::from_static(STREAM.as_bytes()),
        },
    );
    let mut response = streaming
        .stream(prompt("think then answer"))
        .await
        .expect("the stream opens");
    while response.next().await.is_some() {}
    let streamed = response
        .finish()
        .expect("the stream produced a terminal record");

    // Reasoning first, then the visible text — one emitter, one order.
    let reasoning_text = |choice: &[AssistantContent]| match choice.first() {
        Some(AssistantContent::Reasoning(Reasoning { content, .. })) => {
            Some(format!("{content:?}"))
        }
        _ => None,
    };
    assert!(
        reasoning_text(&buffered.choice).is_some(),
        "the unary reply's reasoning is a reasoning block: {:?}",
        buffered.choice
    );
    assert_eq!(
        reasoning_text(&buffered.choice),
        reasoning_text(&streamed.choice),
        "the two shapes of one turn produce the same reasoning block"
    );
    assert_eq!(buffered.choice, streamed.choice);
    assert_eq!(buffered.usage, streamed.usage);
    assert_eq!(buffered.finish_reason(), streamed.finish_reason());
    assert_eq!(
        buffered.choice.last(),
        Some(&AssistantContent::text("the answer"))
    );
}

/// The terminal record accumulates every top-level chunk field, `object`
/// included — a named field would have consumed the key and dropped it while
/// its neighbours survived.
#[tokio::test]
async fn the_streamed_terminal_keeps_every_envelope_field() {
    use crate::providers::openai::wire::{ChatUsage, StreamingCompletionResponse};

    let mut response = Bound::new(
        wire(),
        MockStreamingClient {
            sse_bytes: Bytes::from(recorded(
                "then",
                "raw_stream_capture_matrix/chat_stream_raw_round_trips_typed.yaml",
            )),
        },
    )
    .stream(prompt("Reply with exactly the single word: pong"))
    .await
    .expect("the stream opens");
    while response.next().await.is_some() {}
    let folded = response
        .finish()
        .expect("the stream produced a terminal record");

    let record: StreamingCompletionResponse<ChatUsage> =
        serde_json::from_value(folded.raw.clone()).expect("the terminal record reads back");
    let additional = serde_json::Value::from(
        record
            .additional_params
            .expect("the chunks carried provider metadata"),
    );
    assert_eq!(
        additional["object"], "chat.completion.chunk",
        "`object` is on every chunk of the fixture and must survive: {additional}"
    );
    assert_eq!(additional["service_tier"], "default");
    assert_eq!(
        additional["system_fingerprint"],
        recorded_chunk_field("system_fingerprint")
    );
}

/// A dialect that streams a full `message` on every chunk beside its `delta`
/// is still streaming, and its terminator's envelope fields are the ones the
/// terminal record carries.
///
/// Perplexity does exactly that and tags its terminator
/// `chat.completion.done`. Keying the whole-vs-chunk decision on "a choice
/// carries a message" read frame one as the whole reply, emitted a terminal
/// there, and the driver stopped — so the turn was the first frame and the
/// accumulated envelope was frame one's.
#[tokio::test]
async fn a_dialect_that_streams_a_message_per_chunk_is_still_streaming() {
    use crate::providers::openai::wire::{ChatUsage, PERPLEXITY, StreamingCompletionResponse};

    const BODY: &str = concat!(
        "data: {\"object\":\"chat.completion.chunk\",\"id\":\"78385058-uuid\",\"model\":\"sonar\",",
        "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},",
        "\"message\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
        "data: {\"object\":\"chat.completion.chunk\",\"id\":\"78385058-uuid\",\"model\":\"sonar\",",
        "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"pong\"},",
        "\"message\":{\"role\":\"assistant\",\"content\":\"pong\"}}]}\n\n",
        "data: {\"object\":\"chat.completion.done\",\"id\":\"78385058-uuid\",\"model\":\"sonar\",",
        "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},",
        "\"finish_reason\":\"stop\",\"message\":{\"role\":\"assistant\",\"content\":\"pong\"}}],",
        "\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":1,\"total_tokens\":8}}\n\n",
    );

    let wire = OpenAI::new("pplx").with_dialect(&PERPLEXITY).chat("sonar");
    let mut response = Bound::new(
        wire,
        MockStreamingClient {
            sse_bytes: Bytes::from_static(BODY.as_bytes()),
        },
    )
    .stream(prompt("ping"))
    .await
    .expect("the stream opens");
    while response.next().await.is_some() {}
    let folded = response
        .finish()
        .expect("the stream produced a terminal record");

    // The whole turn, not just its first frame.
    assert_eq!(folded.choice.first(), Some(&AssistantContent::text("pong")));
    assert_eq!(folded.usage.input_tokens, Some(7));
    assert_eq!(folded.usage.total_tokens, Some(8));
    assert_eq!(folded.finish_reason(), Some(FinishReason::Stop));
    // The body's own id, never the transport request id.
    assert_eq!(folded.response_id.as_deref(), Some("78385058-uuid"));

    // The terminator's envelope is the one the record carries: last frame
    // wins, which is what `AdditionalParams::merge` does for a scalar.
    let record: StreamingCompletionResponse<ChatUsage> =
        serde_json::from_value(folded.raw.clone()).expect("the terminal record reads back");
    let additional = serde_json::Value::from(
        record
            .additional_params
            .expect("the chunks carried provider metadata"),
    );
    assert_eq!(
        additional["object"], "chat.completion.done",
        "the terminator's value, not the first chunk's: {additional}"
    );
    assert_eq!(record.response_id.as_deref(), Some("78385058-uuid"));
}

/// A tool call the output-token budget cut mid-arguments must not take the
/// turn down with it — the defect [#2359](https://github.com/0xPlaygrounds/rig/pull/2359)
/// fixed, and the one the `Chat` wire's strict argument decode reintroduced.
///
/// The body carries two calls under `finish_reason: "length"`: one complete,
/// one whose argument string stops inside a JSON string. Only the cut one is
/// unusable, and everything else — the good call, the text beside it, the
/// usage the caller is billed for, the id, the finish reason — is output the
/// provider genuinely delivered.
#[tokio::test]
async fn a_tool_call_cut_mid_arguments_drops_only_itself() {
    const BODY: &str = concat!(
        r#"{"id":"chatcmpl-cut","object":"chat.completion","model":"gpt-4.1-nano","#,
        r#""choices":[{"index":0,"finish_reason":"length","message":{"role":"assistant","#,
        r#""content":"noting both","tool_calls":["#,
        r#"{"id":"call_whole","type":"function","function":{"name":"record","#,
        r#""arguments":"{\"note\": \"first\"}"}},"#,
        r#"{"id":"call_cut","type":"function","function":{"name":"record","#,
        r#""arguments":"{\"note\": \"The"}}]}}],"#,
        r#""usage":{"prompt_tokens":165,"completion_tokens":20,"total_tokens":185}}"#,
    );

    let folded = Bound::new(wire(), RecordingHttpClient::new(BODY))
        .completion(prompt("Record two notes."))
        .await
        .expect(
            "a body cut mid-arguments must still decode: erroring discards the turn's \
             usage, id, finish reason and every other part over one unusable fragment",
        );

    let calls: Vec<_> = folded
        .choice
        .iter()
        .filter_map(|item| match item {
            AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect();
    let [call] = calls.as_slice() else {
        panic!(
            "exactly the complete call survives — a half-parsed one must never reach \
             the caller: {:?}",
            folded.choice
        );
    };
    assert_eq!(call.id.explicit(), Some("call_whole"));
    assert_eq!(call.function.name, "record");
    assert_eq!(
        call.function.arguments,
        serde_json::json!({"note": "first"})
    );

    // The rest of the turn, which erroring would have thrown away.
    assert!(
        folded
            .choice
            .iter()
            .any(|item| item == &AssistantContent::text("noting both")),
        "the text delivered beside the calls survives: {:?}",
        folded.choice
    );
    assert_eq!(folded.usage.output_tokens, Some(20));
    assert_eq!(folded.usage.input_tokens, Some(165));
    assert_eq!(folded.finish_reason(), Some(FinishReason::Length));
}

/// The tolerance is scoped to the budget, not to bad JSON in general.
///
/// A `tool_calls` finish reason is the provider claiming it finished the
/// call; malformed arguments there are its own defect, and silently rewriting
/// the turn as though the call had never been returned would hide it.
#[tokio::test]
async fn malformed_arguments_on_a_completed_tool_turn_stay_a_decode_error() {
    const BODY: &str = concat!(
        r#"{"id":"chatcmpl-bad","object":"chat.completion","model":"gpt-4.1-nano","#,
        r#""choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","#,
        r#""content":null,"tool_calls":["#,
        r#"{"id":"call_bad","type":"function","function":{"name":"record","#,
        r#""arguments":"{\"note\": \"The"}}]}}],"#,
        r#""usage":{"prompt_tokens":165,"completion_tokens":20,"total_tokens":185}}"#,
    );

    let error = Bound::new(wire(), RecordingHttpClient::new(BODY))
        .completion(prompt("Record a note."))
        .await
        .expect_err("a completed tool-call turn with malformed arguments is a defect");
    assert!(
        matches!(error, ProviderError::Json(_)),
        "the malformed payload must stay loud rather than be dropped: {error:?}"
    );
}

/// Arguments that parse are never dropped, whatever they contain.
///
/// Under the same `length` finish reason a call whose arguments are valid
/// JSON is usable tool input; whether its *content* matches the tool's schema
/// is the tool's business, not the decoder's. Dropping it would delete
/// provider output on a guess.
#[tokio::test]
async fn valid_arguments_survive_a_length_truncated_turn() {
    const BODY: &str = concat!(
        r#"{"id":"chatcmpl-odd","object":"chat.completion","model":"gpt-4.1-nano","#,
        r#""choices":[{"index":0,"finish_reason":"length","message":{"role":"assistant","#,
        r#""content":null,"tool_calls":["#,
        r#"{"id":"call_odd","type":"function","function":{"name":"record","#,
        r#""arguments":"{\"unexpected\": 1}"}}]}}],"#,
        r#""usage":{"prompt_tokens":165,"completion_tokens":20,"total_tokens":185}}"#,
    );

    let folded = Bound::new(wire(), RecordingHttpClient::new(BODY))
        .completion(prompt("Record a note."))
        .await
        .expect("valid arguments decode");

    let Some(AssistantContent::ToolCall(call)) = folded.choice.first() else {
        panic!(
            "a parseable call is kept even under a truncated turn: {:?}",
            folded.choice
        );
    };
    assert_eq!(call.id.explicit(), Some("call_odd"));
    assert_eq!(
        call.function.arguments,
        serde_json::json!({"unexpected": 1})
    );
    assert_eq!(folded.finish_reason(), Some(FinishReason::Length));
}

/// Mira's gateway can answer a chat request with a bare JSON string instead
/// of a completion envelope. The shared classifier reads a non-object frame
/// as `Unknown`, so without the modeled event the turn produces no answer at
/// all — a working call becomes a parse error.
#[tokio::test]
async fn a_gateway_may_answer_with_a_bare_string() {
    use crate::providers::openai::wire::MIRA;

    let response = Bound::new(
        OpenAI::new("k").with_dialect(&MIRA).chat("gpt-4o"),
        RecordingHttpClient::new(r#""the whole answer""#),
    )
    .completion(prompt("ask"))
    .await
    .expect("a bare string is the whole reply");

    assert_eq!(
        response.choice.first(),
        Some(&AssistantContent::text("the whole answer"))
    );
    // No metadata and no terminal reason: that is what the gateway sent.
    assert_eq!(response.finish_reason(), None);
    assert_eq!(response.usage, crate::completion::Usage::default());
    assert_eq!(response.response_id, None);

    // Only the dialects measured to do it are tolerant; elsewhere a bare
    // string is still an unmodeled frame, which the classifier warn-skips
    // — so the reply delivered nothing and reported nothing, and that is
    // the shared empty-response rejection rather than a silent, empty
    // success.
    let strict = Bound::new(wire(), RecordingHttpClient::new(r#""the whole answer""#))
        .completion(prompt("ask"))
        .await;
    let Err(ProviderError::Response(message)) = &strict else {
        panic!("openai does not answer with a bare string: {strict:?}");
    };
    assert_eq!(message, crate::message::EMPTY_RESPONSE_ERROR);
}

/// One unary `chat.completion` body whose single choice is empty, ending
/// for `finish_reason` (absent when `None`).
fn empty_turn_body(finish_reason: Option<&str>) -> String {
    let reason = finish_reason.map_or(serde_json::Value::Null, |reason| {
        serde_json::Value::String(reason.to_owned())
    });
    serde_json::json!({
        "id": "chatcmpl-empty",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4.1-nano",
        "choices": [{
            "index": 0,
            "finish_reason": reason,
            "message": {"role": "assistant", "content": ""},
        }],
        "usage": {"prompt_tokens": 900, "completion_tokens": 32, "total_tokens": 932},
    })
    .to_string()
}

/// An empty turn the provider CUT SHORT is kept, because the reason is the
/// only diagnostic the caller has and the usage is what it is billed for.
///
/// The accepting half of [`FinishReason::truncated_output`], which is the
/// one statement of the rule; this wire routes through it rather than
/// restating which reasons are legally empty. Live traffic cannot enumerate
/// the vocabulary — no prompt reliably produces an empty `content_filter`
/// turn — so the boundary is asserted here and the recorded matrices
/// (`crates/rig-cassette/tests/providers/openai/cassette/truncated_turn_matrix.rs`) confirm the
/// `length` half against real bytes.
#[tokio::test]
async fn an_empty_turn_the_provider_cut_short_keeps_its_reason_and_usage() {
    for (reason, expected) in [
        ("length", FinishReason::Length),
        ("content_filter", FinishReason::ContentFilter),
    ] {
        let response = Bound::new(
            wire(),
            RecordingHttpClient::new(empty_turn_body(Some(reason))),
        )
        .completion(prompt("ask"))
        .await
        .unwrap_or_else(|error| panic!("`{reason}` is a cut-short turn, not a defect: {error}"));

        assert!(
            response.choice.is_empty(),
            "{reason}: {:?}",
            response.choice
        );
        assert_eq!(response.finish_reason(), Some(expected), "{reason}");
        // Raising would have thrown these away, which is the whole reason
        // the cut-short turn is kept.
        assert_eq!(response.usage.input_tokens, Some(900), "{reason}");
        assert_eq!(response.usage.output_tokens, Some(32), "{reason}");
        assert_eq!(response.usage.total_tokens, Some(932), "{reason}");
    }
}

/// An empty turn that RAN TO COMPLETION is a provider defect, and so is one
/// that named no terminal at all.
///
/// The rejecting half of the same predicate. `stop` is the case worth
/// naming twice: a provider that says the model finished and hands back
/// nothing has misbehaved, so `stop` must stay out of the legal set even
/// though a stop sequence can consume a whole answer — which is exactly
/// what `crates/rig-cassette/tests/providers/llamacpp/cassette/content_matrix.rs`'s
/// stop-sequence cell records.
#[tokio::test]
async fn an_empty_turn_that_ran_to_completion_is_a_provider_defect() {
    for reason in [
        Some("stop"),
        Some("tool_calls"),
        Some("bespoke_reason"),
        None,
    ] {
        let folded = Bound::new(wire(), RecordingHttpClient::new(empty_turn_body(reason)))
            .completion(prompt("ask"))
            .await;

        let Err(ProviderError::Response(message)) = &folded else {
            panic!("`{reason:?}` does not license an empty turn: {folded:?}");
        };
        assert_eq!(message, crate::message::EMPTY_RESPONSE_ERROR, "{reason:?}");
    }
}

/// Mistral validates message content as a tagged union of its own chunks, so
/// an OpenAI content part has to be rebuilt rather than forwarded — and a
/// part it has no chunk for must fail loudly rather than be dropped.
///
/// The bug this guards is rig#2290: a text-only flattening kept only parts
/// with a `text`/`refusal` key, so an attached image, document or audio clip
/// vanished from the request and the caller got an ordinary completion
/// answering a prompt it never sent.
#[test]
fn the_mistral_body_rebuilds_content_as_its_own_chunks() {
    use crate::message::{Document, DocumentMediaType, Image, Message, UserContent};
    use crate::providers::openai::wire::MISTRAL;

    let encode = |content: Vec<UserContent>| {
        let mut request = prompt("look at this");
        request.chat_history = vec![Message::User { content }];
        OpenAI::new("k")
            .with_dialect(&MISTRAL)
            .chat("mistral-small-latest")
            .encode(request, Mode::Unary)
            .map(|encoded| {
                let [http_request] = encoded.requests.as_slice() else {
                    panic!("one request")
                };
                let Body::Bytes(bytes) = http_request.body() else {
                    panic!("bytes")
                };
                serde_json::from_slice::<serde_json::Value>(bytes).expect("JSON")
            })
    };

    // Text-only content keeps the plain-string form it always took.
    let body = encode(vec![UserContent::text("just words")]).expect("encodes");
    assert_eq!(body["messages"][0]["content"], "just words");

    // An image rides as Mistral's own `image_url` chunk, in an array \u2014 not
    // flattened away.
    let body = encode(vec![
        UserContent::text("describe"),
        UserContent::Image(Image {
            data: crate::message::DocumentSourceKind::Url("https://x.invalid/a.png".to_owned()),
            media_type: None,
            detail: None,
            additional_params: None,
        }),
    ])
    .expect("encodes");
    let parts = body["messages"][0]["content"]
        .as_array()
        .unwrap_or_else(|| panic!("content is a chunk array: {}", body["messages"][0]));
    assert_eq!(parts.len(), 2, "the image survives: {parts:?}");
    assert_eq!(parts[0]["type"], "text");
    assert_eq!(parts[1]["type"], "image_url");

    // Content Mistral has no chunk for fails here rather than being removed.
    let refused = encode(vec![
        UserContent::text("watch"),
        UserContent::Video(crate::message::Video {
            data: crate::message::DocumentSourceKind::Url("https://x.invalid/a.mp4".to_owned()),
            media_type: None,
            additional_params: None,
        }),
    ]);
    let error = refused.expect_err("Mistral carries no video chunk");
    assert!(
        error.to_string().contains("Mistral cannot carry"),
        "the error names the constraint: {error}"
    );

    // A document with inline bytes becomes `document_url`, carrying its name
    // in Mistral's own optional field.
    let body = encode(vec![UserContent::Document(Document {
        data: crate::message::DocumentSourceKind::Base64("ZGF0YQ==".to_owned()),
        media_type: Some(DocumentMediaType::PDF),
        additional_params: None,
    })])
    .expect("encodes");
    let parts = body["messages"][0]["content"]
        .as_array()
        .unwrap_or_else(|| panic!("content is a chunk array: {}", body["messages"][0]));
    assert_eq!(parts[0]["type"], "document_url");
    assert!(
        parts[0].get("document_url").is_some(),
        "the payload is under Mistral's own key: {parts:?}"
    );

    // And no other dialect rebuilds content this way.
    let mut request = prompt("describe");
    request.chat_history = vec![Message::User {
        content: vec![UserContent::Image(Image {
            data: crate::message::DocumentSourceKind::Url("https://x.invalid/a.png".to_owned()),
            media_type: None,
            detail: None,
            additional_params: None,
        })],
    }];
    let openai = wire().encode(request, Mode::Unary).expect("encodes");
    let [http_request] = openai.requests.as_slice() else {
        panic!("one request")
    };
    let Body::Bytes(bytes) = http_request.body() else {
        panic!("bytes")
    };
    let body: serde_json::Value = serde_json::from_slice(bytes).expect("JSON");
    assert_eq!(body["messages"][0]["content"][0]["type"], "image_url");
    assert!(
        body["messages"][0]["content"][0]["image_url"]["url"].is_string(),
        "OpenAI keeps its own nesting: {body}"
    );
}

/// Groq answers with `reasoning` and rejects `reasoning_content` on an
/// assistant message (HTTP 400, "property 'reasoning_content' is
/// unsupported"), so a reasoning turn replays without it. DeepSeek keeps
/// echoing it: its thinking tool loops require the field.
#[test]
fn groq_replays_reasoning_turns_without_reasoning_content() {
    use crate::message::{Message, Reasoning};
    use crate::providers::openai::wire::DEEPSEEK;

    let history = || {
        let mut request = prompt("and then?");
        request.chat_history = vec![
            Message::user("think first"),
            Message::Assistant {
                id: None,
                content: vec![
                    AssistantContent::Reasoning(Reasoning::new("private chain")),
                    AssistantContent::text("visible answer"),
                ],
            },
            Message::user("and then?"),
        ];
        request
    };
    let assistant = |dialect: &Dialect| {
        let encoded = OpenAI::new("k")
            .with_dialect(dialect)
            .chat("m")
            .encode(history(), Mode::Unary)
            .expect("encodes");
        let [request] = encoded.requests.as_slice() else {
            panic!("a chat turn is one request");
        };
        let Body::Bytes(bytes) = request.body() else {
            panic!("a chat request body is bytes");
        };
        let body: serde_json::Value = serde_json::from_slice(bytes).expect("JSON");
        body["messages"][1].clone()
    };

    let groq = assistant(&GROQ);
    assert_eq!(groq["role"], "assistant");
    assert_eq!(groq["content"][0]["text"], "visible answer");
    assert!(groq.get("reasoning_content").is_none(), "{groq}");

    let deepseek = assistant(&DEEPSEEK);
    assert_eq!(deepseek["reasoning_content"], "private chain");
}

/// The last value of a top-level `field` across the recorded text stream's
/// chunks, so an assertion follows the fixture's own bytes.
fn recorded_chunk_field(field: &str) -> String {
    recorded(
        "then",
        "raw_stream_capture_matrix/chat_stream_raw_round_trips_typed.yaml",
    )
    .lines()
    .filter_map(|line| line.trim().strip_prefix("data:"))
    .filter_map(|data| serde_json::from_str::<serde_json::Value>(data.trim()).ok())
    .filter_map(|chunk| {
        chunk
            .get(field)
            .and_then(|value| value.as_str().map(str::to_owned))
    })
    .next_back()
    .expect("the recorded chunks carry the field")
}
