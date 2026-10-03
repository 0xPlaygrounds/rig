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
use crate::completion::FinishReason;
use crate::message::AssistantContent;
use crate::providers::openai::wire::{Dialect, GROQ, OPENAI, OpenAIConfig};
use crate::test_utils::json_body;
use crate::test_utils::{MockStreamingClient, RecordingHttpClient};

use super::super::tests::{recorded, recorded_json};

fn wire() -> Chat {
    OpenAIConfig::new("sk-test")
        .with_dialect(&OPENAI)
        .chat("gpt-4.1-nano")
}

fn prompt(text: &str) -> CompletionRequest {
    CompletionRequest::new(text).temperature(0.0).max_tokens(16)
}

/// The blocks without their provider items.
fn canonical(choice: &[AssistantContent]) -> Vec<AssistantContent> {
    choice.iter().map(AssistantContent::canonical).collect()
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
    let buffered = crate::driver::Model::new(
        wire(),
        RecordingHttpClient::new(recorded("then", unary_cassette)),
    )
    .call(request.clone())
    .await
    .expect("the recorded unary reply decodes");

    let streaming = crate::driver::Model::new(
        wire(),
        MockStreamingClient {
            sse_bytes: Bytes::from(recorded("then", stream_cassette)),
        },
    );
    let mut response = streaming
        .stream(request)
        .expect("the recorded stream opens");
    while response.next().await.is_some() {}
    (
        buffered,
        response
            .finish()
            .await
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

    // Two recordings of one turn: the same blocks, each holding its own
    // reply's provider item.
    assert_eq!(canonical(&buffered.choice), canonical(&streamed.choice));
    assert_eq!(buffered.usage, streamed.usage);
    assert_eq!(buffered.finish_reason(), streamed.finish_reason());
    assert_eq!(buffered.model(), streamed.model());

    // The fixture, so a change in the recorded bytes cannot make the
    // agreement vacuous.
    assert_eq!(
        buffered.choice.first().map(AssistantContent::canonical),
        Some(AssistantContent::text("pong"))
    );
    assert_eq!(buffered.usage.input_tokens, Some(15));
    assert_eq!(buffered.usage.output_tokens, Some(1));
    assert_eq!(buffered.usage.total_tokens, Some(16));
    assert_eq!(buffered.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(buffered.model(), Some("gpt-4.1-nano-2025-04-14"));
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
            .provider()
            .map(|provider| provider.as_str())
            .expect("the call keeps the provider's id");
        assert!(
            recorded("then", cassette).contains(&format!("\"id\":\"{id}\"")),
            "{cassette}: the call id {id} is the recorded one"
        );
        assert_eq!(
            call.id.provider().map(|provider| provider.as_str()),
            Some(id)
        );
    }
    assert_eq!(call.function.name, "ping");
    assert_eq!(call.function.arguments_value(), serde_json::json!({}));
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
        json_body(&encoded.request)
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

/// `[DONE]` before any finish reason fails the turn, as pi's
/// `openai-completions` throws: the provider never said the turn ended.
#[tokio::test]
async fn the_done_sentinel_without_a_finish_fails() {
    const BODY: &str = concat!(
        "data: {\"object\":\"chat.completion.chunk\",\"id\":\"chatcmpl-1\",",
        "\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n",
        "data: [DONE]\n\n",
    );
    let bound = crate::driver::Model::new(
        wire(),
        MockStreamingClient {
            sse_bytes: Bytes::from_static(BODY.as_bytes()),
        },
    );
    let mut response = bound.stream(prompt("hi")).expect("the stream opens");
    while response.next().await.is_some() {}
    let error = response
        .finish()
        .await
        .expect_err("a stream the provider never finished has no response");
    assert!(matches!(error, ProviderError::Truncated), "{error:?}");
}

/// A stream that reaches EOF with neither `[DONE]` nor a finish reason is
/// truncation, and must not be dressed up as a default-usage success.
#[tokio::test]
async fn a_truncated_stream_yields_no_terminal_record() {
    const BODY: &str = concat!(
        "data: {\"object\":\"chat.completion.chunk\",\"id\":\"chatcmpl-1\",",
        "\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"par\"}}]}\n\n",
    );
    let bound = crate::driver::Model::new(
        wire(),
        MockStreamingClient {
            sse_bytes: Bytes::from_static(BODY.as_bytes()),
        },
    );
    let mut response = bound.stream(prompt("hi")).expect("the stream opens");
    while response.next().await.is_some() {}
    let error = response
        .finish()
        .await
        .expect_err("a stream without a terminal record has no response to fold");
    assert!(matches!(error, ProviderError::Truncated), "{error:?}");
}

/// The wire's in-band error envelope arrives with a 200 status, so only the
/// decoder can see it: it must fail the turn rather than read as a chunk.
#[tokio::test]
async fn an_in_band_error_envelope_fails_the_turn() {
    const BODY: &str =
        "data: {\"error\":{\"message\":\"rate limited\"},\"choices\":[]}\n\ndata: [DONE]\n\n";
    let bound = crate::driver::Model::new(
        wire(),
        MockStreamingClient {
            sse_bytes: Bytes::from_static(BODY.as_bytes()),
        },
    );
    let mut response = bound.stream(prompt("hi")).expect("the stream opens");
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
    // A failed turn has nothing to fold.
    assert!(response.finish().await.is_err());
}

/// A dialect that spells the cap `max_completion_tokens` does so only for
/// the reasoning families, because this same wire reaches compatible servers
/// that know only the legacy field.
#[test]
fn the_output_cap_spelling_follows_the_model_family() {
    fn cap_key(model: &str) -> &'static str {
        let encoded = OpenAIConfig::new("sk-test")
            .chat(model)
            .encode(prompt("hi"), Mode::Unary)
            .expect("encodes");
        let body = json_body(&encoded.request);
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
    let groq = OpenAIConfig::new("gsk-test")
        .with_dialect(&GROQ)
        .chat("gpt-5.2")
        .encode(prompt("hi"), Mode::Unary)
        .expect("encodes");
    let body = json_body(&groq.request);
    assert!(body.get("max_tokens").is_some());
    assert!(body.get("max_completion_tokens").is_none());
}

/// GPT-6 on Chat Completions answers a request with function tools with a
/// 400 unless it runs at `reasoning_effort: "none"` (recorded 2026-09-29:
/// "Function tools with reasoning_effort are not supported for gpt-6-sol in
/// /v1/chat/completions"), and Astra and 6.1 Sol reject `"none"` itself. Rig
/// refuses before sending, so this is an encode error with no request to
/// record: it cannot be a cassette test.
#[test]
fn gpt_6_chat_tools_need_effort_none_or_the_responses_wire() {
    use crate::completion::ToolDefinition;
    use crate::providers::openai::completion::{
        GPT_5_4, GPT_6_1_SOL, GPT_6_ASTRA, GPT_6_LUNA, GPT_6_SOL,
    };

    let with_tool = |params: Option<serde_json::Value>| {
        let mut request = prompt("look it up");
        request.tools = vec![ToolDefinition {
            name: crate::message::ToolName::new("lookup").expect("tool name"),
            description: "Look something up.".to_owned(),
            parameters: serde_json::json!({"type": "object", "properties": {}}),
        }];
        request.temperature = None;
        request.additional_params = params;
        request
    };
    let encode = |model: &str, params: Option<serde_json::Value>| {
        OpenAIConfig::new("sk-test")
            .chat(model)
            .encode(with_tool(params), Mode::Unary)
    };
    let none = || Some(serde_json::json!({"reasoning_effort": "none"}));

    for model in [GPT_6_SOL, GPT_6_LUNA] {
        let error = encode(model, None).expect_err("tools at the default effort are refused");
        assert!(
            error
                .to_string()
                .contains("only at reasoning_effort \"none\""),
            "{model}: {error}"
        );
        assert!(
            encode(model, none()).is_ok(),
            "{model} takes tools at effort none"
        );
    }
    for model in [GPT_6_ASTRA, GPT_6_1_SOL] {
        for params in [None, none()] {
            let error = encode(model, params).expect_err("no effort lets these call tools here");
            assert!(
                error.to_string().contains("Use the Responses wire"),
                "{model}: {error}"
            );
        }
    }
    // Without tools, and on models that take tools here, nothing changes.
    assert!(
        OpenAIConfig::new("sk-test")
            .chat(GPT_6_ASTRA)
            .encode(prompt("hi"), Mode::Unary)
            .is_ok()
    );
    assert!(encode(GPT_5_4, None).is_ok());
    // Compatible gateways on this wire are not OpenAI's endpoint.
    assert!(
        OpenAIConfig::new("gsk-test")
            .with_dialect(&GROQ)
            .chat(GPT_6_SOL)
            .encode(with_tool(None), Mode::Unary)
            .is_ok()
    );
}

/// OpenRouter takes no provider file id, so the adapter leaves a placeholder
/// for a document carrying only one; every other dialect sends the id.
#[test]
fn openrouter_gets_a_placeholder_for_a_file_id_document() {
    use crate::message::{Document, DocumentSourceKind, Message, UserContent};
    use crate::providers::openai::wire::OPENROUTER;

    let body = |chat: Chat| {
        let mut request = prompt("read this");
        request.chat_history = vec![Message::User {
            content: vec![UserContent::Document(Document {
                data: DocumentSourceKind::FileId("file-abc".to_owned()),
                media_type: None,
                additional_params: None,
            })],
        }];
        let request = <crate::operation::Completion as crate::wire::Operation>::prepare(
            request,
            &chat.describe(),
        )
        .expect("the request prepares");
        json_body(&chat.encode(request, Mode::Unary).expect("encodes").request).to_string()
    };
    let openrouter = body(
        OpenAIConfig::new("k")
            .with_dialect(&OPENROUTER)
            .chat("openai/gpt-4o"),
    );
    assert!(
        openrouter.contains(crate::completion::history::DOCUMENT_UNSENDABLE)
            && !openrouter.contains("file-abc"),
        "{openrouter}"
    );
    let openai = body(wire());
    assert!(openai.contains("\"file_id\":\"file-abc\""), "{openai}");
}

/// The terminal record is readable back out of the stream's `raw`, which is
/// the escape hatch for every provider field this wire does not normalize.
#[tokio::test]
async fn the_streamed_terminal_reads_back_as_the_provider_record() {
    let mut response = crate::driver::Model::new(
        wire(),
        MockStreamingClient {
            sse_bytes: Bytes::from(recorded(
                "then",
                "raw_stream_capture_matrix/chat_stream_raw_round_trips_typed.yaml",
            )),
        },
    )
    .stream(prompt("Reply with exactly the single word: pong"))
    .expect("the stream opens");
    while response.next().await.is_some() {}
    let folded = response
        .finish()
        .await
        .expect("the stream produced a terminal record");

    let record = &folded.raw;
    let usage = &record["usage"];
    assert_eq!(usage["prompt_tokens"], 15);
    assert_eq!(usage["completion_tokens"], 1);
    assert_eq!(usage["total_tokens"], 16);
    assert_eq!(record["model"], "gpt-4.1-nano-2025-04-14");
    assert_eq!(record["response_id"], recorded_chunk_field("id"));

    // The provider-native fields the wire does not normalize reach the caller
    // here, which is why `raw` is the record and not the parse.
    let additional = record["additional_params"].clone();
    assert_eq!(additional["service_tier"], "default");
    assert_eq!(
        additional["system_fingerprint"],
        recorded_chunk_field("system_fingerprint")
    );

    // And the normalized view agrees with it.
    assert_eq!(folded.usage.input_tokens, Some(15));
    assert_eq!(folded.model(), Some("gpt-4.1-nano-2025-04-14"));
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

    let buffered = crate::driver::Model::new(wire(), RecordingHttpClient::new(UNARY))
        .call(prompt("think then answer"))
        .await
        .expect("the unary reply decodes without tripping the sequence law");

    let streaming = crate::driver::Model::new(
        wire(),
        MockStreamingClient {
            sse_bytes: Bytes::from_static(STREAM.as_bytes()),
        },
    );
    let mut response = streaming
        .stream(prompt("think then answer"))
        .expect("the stream opens");
    while response.next().await.is_some() {}
    let streamed = response
        .finish()
        .await
        .expect("the stream produced a terminal record");

    // Reasoning first, then the visible text — one emitter, one order.
    let reasoning_text = |choice: &[AssistantContent]| match choice.first() {
        Some(AssistantContent::Reasoning(reasoning)) => Some(reasoning.text.clone()),
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
    assert_eq!(canonical(&buffered.choice), canonical(&streamed.choice));
    assert_eq!(buffered.usage, streamed.usage);
    assert_eq!(buffered.finish_reason(), streamed.finish_reason());
    assert_eq!(
        buffered.choice.last().map(AssistantContent::canonical),
        Some(AssistantContent::text("the answer"))
    );
}

/// The terminal record accumulates every top-level chunk field, `object`
/// included — a named field would have consumed the key and dropped it while
/// its neighbours survived.
#[tokio::test]
async fn the_streamed_terminal_keeps_every_envelope_field() {
    let mut response = crate::driver::Model::new(
        wire(),
        MockStreamingClient {
            sse_bytes: Bytes::from(recorded(
                "then",
                "raw_stream_capture_matrix/chat_stream_raw_round_trips_typed.yaml",
            )),
        },
    )
    .stream(prompt("Reply with exactly the single word: pong"))
    .expect("the stream opens");
    while response.next().await.is_some() {}
    let folded = response
        .finish()
        .await
        .expect("the stream produced a terminal record");

    let additional = folded.raw["additional_params"].clone();
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
    use crate::providers::openai::wire::PERPLEXITY;

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

    let wire = OpenAIConfig::new("pplx")
        .with_dialect(&PERPLEXITY)
        .chat("sonar");
    let mut response = crate::driver::Model::new(
        wire,
        MockStreamingClient {
            sse_bytes: Bytes::from_static(BODY.as_bytes()),
        },
    )
    .stream(prompt("ping"))
    .expect("the stream opens");
    while response.next().await.is_some() {}
    let folded = response
        .finish()
        .await
        .expect("the stream produced a terminal record");

    // The whole turn, not just its first frame.
    assert_eq!(
        folded.choice.first().map(AssistantContent::canonical),
        Some(AssistantContent::text("pong"))
    );
    assert_eq!(folded.usage.input_tokens, Some(7));
    assert_eq!(folded.usage.total_tokens, Some(8));
    assert_eq!(folded.finish_reason(), Some(FinishReason::Stop));
    // The body's own id, never the transport request id.
    assert_eq!(folded.response_id(), Some("78385058-uuid"));

    // The terminator's envelope is the one the record carries: last frame
    // wins, which is what `AdditionalParams::merge` does for a scalar.
    let additional = folded.raw["additional_params"].clone();
    assert_eq!(
        additional["object"], "chat.completion.done",
        "the terminator's value, not the first chunk's: {additional}"
    );
    assert_eq!(folded.raw["response_id"], "78385058-uuid");
}

/// A tool call the output-token budget cut mid-arguments must not take the
/// turn down with it (#2359). The body carries two calls under
/// `finish_reason: "length"`: one complete, one whose argument string stops
/// inside a JSON string. The cut one keeps what its arguments state, with
/// their text, and everything else the provider delivered survives.
#[tokio::test]
async fn a_tool_call_cut_mid_arguments_keeps_what_it_states() {
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

    let folded = crate::driver::Model::new(wire(), RecordingHttpClient::new(BODY))
        .call(prompt("Record two notes."))
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
    let [call, cut] = calls.as_slice() else {
        panic!("both calls survive: {:?}", folded.choice);
    };
    assert_eq!(
        cut.function.arguments_value(),
        serde_json::json!({"note": "The"})
    );
    assert_eq!(
        cut.function.invalid_arguments.as_deref(),
        Some(r#"{"note": "The"#)
    );
    assert_eq!(
        call.id.provider().map(|provider| provider.as_str()),
        Some("call_whole")
    );
    assert_eq!(call.function.name, "record");
    assert_eq!(
        call.function.arguments_value(),
        serde_json::json!({"note": "first"})
    );

    // The rest of the turn, which erroring would have thrown away.
    assert!(
        folded
            .choice
            .iter()
            .any(|item| item.canonical() == AssistantContent::text("noting both")),
        "the text delivered beside the calls survives: {:?}",
        folded.choice
    );
    assert_eq!(folded.usage.output_tokens, Some(20));
    assert_eq!(folded.usage.input_tokens, Some(165));
    assert_eq!(folded.finish_reason(), Some(FinishReason::Length));
}

/// Malformed arguments on a completed tool turn never fail the reply: the
/// call is kept with its raw text, and the agent answers it with an error.
#[tokio::test]
async fn malformed_arguments_on_a_completed_tool_turn_keep_their_text() {
    const BODY: &str = concat!(
        r#"{"id":"chatcmpl-bad","object":"chat.completion","model":"gpt-4.1-nano","#,
        r#""choices":[{"index":0,"finish_reason":"tool_calls","message":{"role":"assistant","#,
        r#""content":null,"tool_calls":["#,
        r#"{"id":"call_bad","type":"function","function":{"name":"record","#,
        r#""arguments":"{\"note\": \"The"}}]}}],"#,
        r#""usage":{"prompt_tokens":165,"completion_tokens":20,"total_tokens":185}}"#,
    );

    let response = crate::driver::Model::new(wire(), RecordingHttpClient::new(BODY))
        .call(prompt("Record a note."))
        .await
        .expect("a malformed call does not fail the reply");
    let call = response.tool_calls().next().expect("the call is kept");
    assert_eq!(
        call.function.arguments_value(),
        serde_json::json!({"note": "The"})
    );
    assert_eq!(
        call.function.invalid_arguments.as_deref(),
        Some(r#"{"note": "The"#)
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

    let folded = crate::driver::Model::new(wire(), RecordingHttpClient::new(BODY))
        .call(prompt("Record a note."))
        .await
        .expect("valid arguments decode");

    let Some(AssistantContent::ToolCall(call)) = folded.choice.first() else {
        panic!(
            "a parseable call is kept even under a truncated turn: {:?}",
            folded.choice
        );
    };
    assert_eq!(
        call.id.provider().map(|provider| provider.as_str()),
        Some("call_odd")
    );
    assert_eq!(
        call.function.arguments_value(),
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

    let response = crate::driver::Model::new(
        OpenAIConfig::new("k").with_dialect(&MIRA).chat("gpt-4o"),
        RecordingHttpClient::new(r#""the whole answer""#),
    )
    .call(prompt("ask"))
    .await
    .expect("a bare string is the whole reply");

    assert_eq!(
        response.choice.first().map(AssistantContent::canonical),
        Some(AssistantContent::text("the whole answer"))
    );
    // No metadata and no terminal reason: the whole answer is a stop, as
    // pi ends a reply on a wire that states none.
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(response.usage, crate::completion::Usage::default());
    assert_eq!(response.response_id(), None);

    // Only the dialects measured to do it are tolerant; elsewhere a bare
    // string is still an unmodeled frame, which the classifier warn-skips
    // — so the provider never ended the reply, and that is truncation
    // rather than a silent, empty success.
    let strict =
        crate::driver::Model::new(wire(), RecordingHttpClient::new(r#""the whole answer""#))
            .call(prompt("ask"))
            .await;
    assert!(
        matches!(strict, Err(ProviderError::Truncated)),
        "openai does not answer with a bare string: {strict:?}"
    );
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
        let response = crate::driver::Model::new(
            wire(),
            RecordingHttpClient::new(empty_turn_body(Some(reason))),
        )
        .call(prompt("ask"))
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

/// An empty turn that ran to completion is an empty turn, in both modes,
/// as pi keeps it: emptiness is the core's to judge, not the decoder's. A
/// reason outside the vocabulary, or none at all, fails it.
#[tokio::test]
async fn an_empty_turn_that_ran_to_completion_is_an_empty_turn() {
    for (reason, failed) in [
        (Some("stop"), false),
        (Some("tool_calls"), false),
        (Some("bespoke_reason"), true),
        (None, true),
    ] {
        let folded =
            crate::driver::Model::new(wire(), RecordingHttpClient::new(empty_turn_body(reason)))
                .call(prompt("ask"))
                .await
                .unwrap_or_else(|error| panic!("{reason:?}: {error}"));
        assert!(folded.choice.is_empty(), "{reason:?}: {:?}", folded.choice);
        assert_eq!(
            folded.stop().is_failure(),
            failed,
            "{reason:?}: {:?}",
            folded.stop()
        );
    }
}

/// Mistral validates message content as a tagged union of its own chunks, so
/// an OpenAI content part has to be rebuilt rather than forwarded, and a
/// part it has no chunk for is a placeholder rather than dropped.
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
        OpenAIConfig::new("k")
            .with_dialect(&MISTRAL)
            .chat("mistral-small-latest")
            .encode(request, Mode::Unary)
            .map(|encoded| json_body(&encoded.request))
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
            native: None,
        }),
    ])
    .expect("encodes");
    let parts = body["messages"][0]["content"]
        .as_array()
        .unwrap_or_else(|| panic!("content is a chunk array: {}", body["messages"][0]));
    assert_eq!(parts.len(), 2, "the image survives: {parts:?}");
    assert_eq!(parts[0]["type"], "text");
    assert_eq!(parts[1]["type"], "image_url");

    // Content Mistral has no chunk for is a placeholder the adapter leaves,
    // never removed.
    let mut request = prompt("watch");
    request.chat_history = vec![Message::User {
        content: vec![
            UserContent::text("watch"),
            UserContent::Video(crate::message::Video {
                data: crate::message::DocumentSourceKind::Url("https://x.invalid/a.mp4".to_owned()),
                media_type: None,
                additional_params: None,
            }),
        ],
    }];
    let mistral = OpenAIConfig::new("k")
        .with_dialect(&MISTRAL)
        .chat("mistral-small-latest");
    let request = <crate::operation::Completion as crate::wire::Operation>::prepare(
        request,
        &mistral.describe(),
    )
    .expect("the request prepares");
    let body = json_body(
        &mistral
            .encode(request, Mode::Unary)
            .expect("encodes")
            .request,
    );
    assert!(
        body.to_string()
            .contains(crate::completion::history::VIDEO_UNSENDABLE),
        "the video is a placeholder: {body}"
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
            native: None,
        })],
    }];
    let openai = wire().encode(request, Mode::Unary).expect("encodes");
    let body = json_body(&openai.request);
    assert_eq!(body["messages"][0]["content"][0]["type"], "image_url");
    assert!(
        body["messages"][0]["content"][0]["image_url"]["url"].is_string(),
        "OpenAI keeps its own nesting: {body}"
    );
}

/// Groq answers with `reasoning` and rejects `reasoning_content` on an
/// assistant message (HTTP 400, "property 'reasoning_content' is
/// unsupported"), while it accepts its own `reasoning` back (probed live
/// 2026-10-01). A turn replays under the field its provider sent, so no
/// dialect rewrites it: each provider's message goes back as it came.
#[test]
fn a_reasoning_turn_replays_under_the_field_its_provider_sent() {
    use crate::message::{Message, Reasoning};
    use crate::providers::openai::wire::DEEPSEEK;

    let assistant = |dialect: &Dialect, field: &str| {
        let turn = crate::message::AssistantMessage::new(vec![
            AssistantContent::Reasoning(Reasoning::new("private chain"))
                .with_native(serde_json::json!({ field: "private chain" })),
            AssistantContent::text("visible answer"),
        ]);
        let mut request = prompt("and then?");
        request.chat_history = vec![
            Message::user("think first"),
            Message::Assistant(turn),
            Message::user("and then?"),
        ];
        let encoded = OpenAIConfig::new("k")
            .with_dialect(dialect)
            .chat("m")
            .encode(request, Mode::Unary)
            .expect("encodes");
        json_body(&encoded.request)["messages"][1].clone()
    };

    let groq = assistant(&GROQ, "reasoning");
    assert_eq!(groq["content"], "visible answer");
    assert_eq!(groq["reasoning"], "private chain");
    assert!(groq.get("reasoning_content").is_none(), "{groq}");

    let deepseek = assistant(&DEEPSEEK, "reasoning_content");
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

/// One frame of a chunked reply, as the decoder reads it.
fn frame(chunk: &serde_json::Value) -> crate::wire::WireFrame {
    crate::wire::WireFrame::Text(chunk.to_string())
}

/// A `chat.completion.chunk` whose primary choice carries `delta`.
fn delta_chunk(delta: serde_json::Value, finish: Option<&str>) -> crate::wire::WireFrame {
    frame(&serde_json::json!({
        "id": "chatcmpl-items",
        "model": "gpt-4.1-nano",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }))
}

/// Every classified event of this wire has a sample, so a new one fails to
/// compile until it is numbered here and fails until a frame classifies to
/// it.
#[test]
fn every_chat_event_has_a_sample() {
    let index = |event: &ChatEvent| match event {
        ChatEvent::Chunk(_) => 0,
        ChatEvent::Whole(_) => 1,
        ChatEvent::Done => 2,
        ChatEvent::Failure(_) => 3,
        ChatEvent::BareText(_) => 4,
    };
    let classify = |dialect: &'static Dialect, data: &str| {
        let decoder = OpenAIConfig::new("k")
            .with_dialect(dialect)
            .chat("m")
            .decoder();
        match Decoder::<'_, Completion>::classify(
            &decoder,
            crate::wire::WireFrame::Text(data.to_owned()),
        ) {
            WireEvent::Known(event) => event,
            _ => panic!("{data} classifies as a known event"),
        }
    };
    let samples = [
        classify(
            &OPENAI,
            r#"{"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"hi"}}]}"#,
        ),
        classify(
            &OPENAI,
            r#"{"object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}"#,
        ),
        classify(&OPENAI, "[DONE]"),
        classify(&OPENAI, r#"{"error":{"message":"overloaded"}}"#),
        classify(&super::super::MIRA, r#""the whole answer""#),
    ];
    crate::test_utils::history::assert_every_variant(&samples, index, 5);
}

/// The whole reply below, and the same turn streamed.
fn invented_turn() -> (Vec<crate::wire::WireFrame>, Vec<crate::wire::WireFrame>) {
    let call = serde_json::json!({
        "id": "call_probe",
        "type": "function",
        "function": {"name": "add", "arguments": "{\"x\":1}"},
        "x_call_probe": {"kept": true},
    });
    let whole = serde_json::json!({
        "id": "chatcmpl-items",
        "model": "gpt-4.1-nano",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": "adding",
                "x_message_probe": [{"kind": "invented"}],
                "tool_calls": [call],
            },
        }],
    });
    let streamed = vec![
        delta_chunk(
            serde_json::json!({"role": "assistant", "content": "add"}),
            None,
        ),
        delta_chunk(
            serde_json::json!({"content": "ing", "x_message_probe": [{"kind": "invented"}]}),
            None,
        ),
        delta_chunk(
            serde_json::json!({"tool_calls": [{
                "index": 0,
                "id": "call_probe",
                "type": "function",
                "function": {"name": "add", "arguments": "{\"x\""},
                "x_call_probe": {"kept": true},
            }]}),
            None,
        ),
        delta_chunk(
            serde_json::json!({"tool_calls": [{"index": 0, "function": {"arguments": ":1}"}}]}),
            None,
        ),
        delta_chunk(serde_json::json!({}), Some("tool_calls")),
        crate::wire::WireFrame::Text("[DONE]".to_owned()),
    ];
    (vec![frame(&whole)], streamed)
}

/// A message field rig has never seen stays in the turn's message, for
/// display, but never goes back: the message is rebuilt from its blocks. A
/// field rig has never seen on a tool call is the call's own item, and goes
/// back to the model that sent it.
#[test]
fn an_invented_call_field_replays_and_an_invented_message_field_does_not() {
    use crate::message::{AssistantMessage, Message};
    use crate::wire::Mode;

    let wire = wire();
    let (whole, streamed) = invented_turn();
    crate::test_utils::history::assert_restated_agrees(&wire, whole.clone(), streamed.clone());
    for (mode, frames) in [(Mode::Unary, whole), (Mode::Streaming, streamed)] {
        let response =
            crate::test_utils::history::decode(&wire, mode, frames).expect("the reply decodes");
        let turn = AssistantMessage {
            content: response.choice.clone(),
            ..response.head()
        };
        let call = turn
            .content
            .iter()
            .find_map(AssistantContent::native_item)
            .filter(|item| item.get("x_call_probe").is_some())
            .expect("the call keeps its item");
        assert_eq!(call["x_call_probe"]["kept"], true, "{mode:?}");

        let mut request = prompt("and then?");
        request.chat_history = crate::completion::history::adapt(
            &[
                Message::user("add one"),
                Message::Assistant(turn.clone()),
                Message::tool_result(
                    crate::message::CallId::from_wire("call_probe"),
                    crate::message::ToolName::new("add").expect("tool name"),
                    "1",
                ),
            ],
            &wire,
        );
        let body = json_body(&wire.encode(request, Mode::Unary).expect("encodes").request);
        let replayed = &body["messages"][1];
        assert!(
            replayed.get("x_message_probe").is_none(),
            "{mode:?}: {replayed}"
        );
        assert_eq!(
            replayed["tool_calls"][0]["x_call_probe"]["kept"], true,
            "{mode:?}"
        );
        assert_eq!(replayed["content"], "adding", "{mode:?}");
    }
}

/// OpenRouter streams its reasoning details as fragments; the reasoning
/// block keeps each detail's id, format and kind as the provider sent them,
/// the encrypted entry whole beside the merged summary.
#[test]
fn a_streamed_reasoning_detail_keeps_its_id_format_and_kind() {
    use crate::wire::Mode;

    let summary = |text: &str| {
        serde_json::json!({"reasoning": text, "reasoning_details": [{
            "type": "reasoning.summary", "format": "openai-responses-v1", "index": 0, "summary": text,
        }]})
    };
    let frames = vec![
        delta_chunk(summary("Weighing "), None),
        delta_chunk(summary("it up"), None),
        delta_chunk(
            serde_json::json!({"reasoning_details": [{
                "type": "reasoning.encrypted", "id": "rs_1", "format": "openai-responses-v1",
                "index": 0, "data": "gAAA",
            }]}),
            None,
        ),
        delta_chunk(serde_json::json!({"content": "done"}), Some("stop")),
        crate::wire::WireFrame::Text("[DONE]".to_owned()),
    ];
    let wire = OpenAIConfig::new("k")
        .with_dialect(&super::super::OPENROUTER)
        .chat("m");
    let response = crate::test_utils::history::decode(&wire, Mode::Streaming, frames)
        .expect("the reply decodes");
    let Some(AssistantContent::Reasoning(reasoning)) = response.choice.first() else {
        panic!("the reasoning leads: {:?}", response.choice);
    };
    assert_eq!(reasoning.text, "Weighing it up");
    let item = &reasoning
        .native
        .as_ref()
        .expect("the block keeps its fields")
        .item;
    assert_eq!(
        item["reasoning_details"],
        serde_json::json!([
            {"type": "reasoning.summary", "format": "openai-responses-v1", "index": 0, "summary": "Weighing it up"},
            {"type": "reasoning.encrypted", "id": "rs_1", "format": "openai-responses-v1", "index": 0, "data": "gAAA"},
        ])
    );
    assert_eq!(item["reasoning"], "Weighing it up");
}

/// A stream's annotations and audio reach the turn: its text block holds
/// the audio's id, and the next request sends the text and the id back.
#[test]
fn streamed_annotations_and_audio_survive() {
    use crate::wire::Mode;

    let frames = vec![
        delta_chunk(
            serde_json::json!({"role": "assistant", "content": "See ", "audio": {"id": "audio_1", "transcript": "See "}}),
            None,
        ),
        delta_chunk(
            serde_json::json!({"content": "rig.rs", "audio": {"transcript": "rig.rs", "data": "UklG"}}),
            None,
        ),
        delta_chunk(
            serde_json::json!({"annotations": [{"type": "url_citation", "url_citation": {"url": "https://rig.rs", "start_index": 4, "end_index": 10}}]}),
            Some("stop"),
        ),
        crate::wire::WireFrame::Text("[DONE]".to_owned()),
    ];
    let response = crate::test_utils::history::decode(&wire(), Mode::Streaming, frames)
        .expect("the reply decodes");
    let [AssistantContent::Text(text)] = response.choice.as_slice() else {
        panic!("a text block holding the audio: {:?}", response.choice);
    };
    assert_eq!(text.text, "See rig.rs");
    assert_eq!(
        AssistantContent::Text(text.clone()).native_item(),
        Some(&serde_json::json!({"audio": {"id": "audio_1"}}))
    );
    // Replay sends what OpenAI takes back: the text and the audio's id.
    let turn = crate::message::AssistantMessage {
        content: response.choice.clone(),
        ..response.head()
    };
    let replayed = replayed(&wire(), turn);
    assert_eq!(
        replayed,
        serde_json::json!({"role": "assistant", "content": "See rig.rs", "audio": {"id": "audio_1"}})
    );
}

/// pi's id rule for this wire (`openai-completions.js`, `normalizeToolCallId`)
/// and Mistral's nine alphanumerics. The hashes are pi's `shortHash`.
#[test]
fn foreign_call_ids_are_normalized_the_way_pi_does() {
    use crate::completion::ReplayTarget;

    let openai = wire();
    assert_eq!(
        openai.normalize_tool_call_id("call_abc123", openai.model(), None),
        "call_abc123"
    );
    // OpenAI's own ids are cut to 40; nothing else about them changes.
    assert_eq!(
        openai.normalize_tool_call_id(
            "toolu_01ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij",
            openai.model(),
            None
        ),
        "toolu_01ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef"
    );
    assert_eq!(
        openai.normalize_tool_call_id("a.b:c", openai.model(), None),
        "a.b:c"
    );
    // A Responses `call|item` id joins its sanitized halves, hashed past 40.
    assert_eq!(
        openai.normalize_tool_call_id("call_1|fc+ab/c", openai.model(), None),
        "call_1_fc_ab_c"
    );
    assert_eq!(
        openai.normalize_tool_call_id(
            "call_abcdefghijklmnopqrstuvwxyz0123|fc_0123456789abcdefghijklmnopqrstuvwxyz",
            openai.model(),
            None
        ),
        "call_abcdefghijklmnopqrstuvwxyz_7gi8bx11"
    );
    // Any other dialect keeps the id.
    let deepseek = OpenAIConfig::new("k")
        .with_dialect(&super::super::DEEPSEEK)
        .chat("m");
    assert_eq!(
        deepseek.normalize_tool_call_id(
            "toolu_01ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij",
            deepseek.model(),
            None
        ),
        "toolu_01ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"
    );

    let mistral = OpenAIConfig::new("k")
        .with_dialect(&super::super::MISTRAL)
        .chat("m");
    assert_eq!(
        mistral.normalize_tool_call_id("abc123XYZ", mistral.model(), None),
        "abc123XYZ"
    );
    assert_eq!(
        mistral.normalize_tool_call_id("call_abc123", mistral.model(), None),
        "9k918q7jl"
    );
}

/// Groq streams gpt-oss messages with a `channel` and rejects it coming back:
/// "'messages.2' : for 'role:assistant' the following must be
/// satisfied[('messages.2' : property 'channel' is unsupported)]" (HTTP 400,
/// recorded 2026-10-01). The rest of the message replays as it came.
#[test]
fn groq_never_gets_its_streamed_channel_back() {
    use crate::message::Message;

    let wire = OpenAIConfig::new("k")
        .with_dialect(&GROQ)
        .chat("openai/gpt-oss-20b");
    let frames = vec![
        delta_chunk(
            serde_json::json!({"role": "assistant", "channel": "analysis", "reasoning": "plan"}),
            None,
        ),
        delta_chunk(
            serde_json::json!({"channel": "final", "content": "done"}),
            Some("stop"),
        ),
        crate::wire::WireFrame::Text("[DONE]".to_owned()),
    ];
    let response = crate::test_utils::history::decode(&wire, Mode::Streaming, frames)
        .expect("the reply decodes");
    let turn = crate::message::AssistantMessage {
        content: response.choice.clone(),
        ..response.head()
    };
    let mut request = prompt("and then?");
    request.chat_history = crate::completion::history::adapt(
        &[
            Message::user("go"),
            Message::Assistant(turn),
            Message::user("and then?"),
        ],
        &wire,
    );
    let encoded = wire.encode(request, Mode::Unary).expect("encodes");
    assert_eq!(
        json_body(&encoded.request)["messages"][1],
        serde_json::json!({"role": "assistant", "reasoning": "plan", "content": "done"})
    );
}

/// A streamed function call that never gets a name is dropped, since
/// nothing can answer it; a custom call is a call whose arguments are its
/// `{"input"}`, kept with its item; a call of a kind rig cannot answer is
/// kept but never sent back; and content that is not text fails the reply.
#[test]
fn malformed_and_unknown_calls_are_never_dropped_silently() {
    use crate::wire::Mode;

    let wire = wire();
    let nameless = vec![delta_chunk(
        serde_json::json!({"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                "function": {"arguments": "{}"}}]}),
        Some("tool_calls"),
    )];
    let response = crate::test_utils::history::decode(&wire, Mode::Streaming, nameless)
        .expect("a nameless call does not fail the reply");
    assert!(response.choice.is_empty(), "{:?}", response.choice);

    let custom = vec![delta_chunk(
        serde_json::json!({"tool_calls": [{"index": 0, "id": "call_1", "type": "custom",
                "custom": {"name": "grep", "input": "x"}}]}),
        Some("tool_calls"),
    )];
    let response = crate::test_utils::history::decode(&wire, Mode::Streaming, custom)
        .expect("a custom call decodes");
    let [AssistantContent::ToolCall(call)] = response.choice.as_slice() else {
        panic!("a custom call is a call: {:?}", response.choice);
    };
    assert_eq!(call.function.name.as_str(), "grep");
    assert_eq!(
        call.function.arguments_value(),
        serde_json::json!({"input": "x"})
    );
    assert_eq!(
        call.native
            .as_ref()
            .map(|native| native.item["type"].clone()),
        Some(serde_json::json!("custom"))
    );

    let invented = vec![delta_chunk(
        serde_json::json!({"tool_calls": [{"index": 0, "id": "call_1", "type": "x_rig_invented",
                "x_rig_invented": {"name": "grep"}}]}),
        Some("tool_calls"),
    )];
    let response = crate::test_utils::history::decode(&wire, Mode::Streaming, invented)
        .expect("an unknown call kind decodes");
    assert!(
        matches!(response.choice.as_slice(), [AssistantContent::Opaque(opaque)]
            if !opaque.replay && opaque.item["type"] == "x_rig_invented"),
        "{:?}",
        response.choice
    );

    // Content of no content type is read leniently, as nothing.
    let numeric = vec![
        delta_chunk(serde_json::json!({"content": 42}), Some("stop")),
        crate::wire::WireFrame::Text("[DONE]".to_owned()),
    ];
    let response = crate::test_utils::history::decode(&wire, Mode::Streaming, numeric)
        .expect("a reply with numeric content decodes");
    assert!(response.choice.is_empty(), "{:?}", response.choice);
}

/// A reply that is only audio is its transcript, holding the audio, so the
/// turn reaches history and replays it.
#[test]
fn an_audio_only_reply_keeps_its_audio() {
    use crate::wire::Mode;

    let frames = vec![
        delta_chunk(
            serde_json::json!({"role": "assistant",
                "audio": {"id": "audio_1", "data": "AAA", "transcript": "hi"}}),
            None,
        ),
        delta_chunk(serde_json::json!({}), Some("stop")),
    ];
    let whole = vec![crate::wire::WireFrame::Text(
        serde_json::json!({"id": "chatcmpl-items", "model": "gpt-4.1-nano",
            "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant",
                "content": null,
                "audio": {"id": "audio_1", "data": "AAA", "transcript": "hi"}}}]})
        .to_string(),
    )];
    for (mode, frames) in [(Mode::Streaming, frames), (Mode::Unary, whole)] {
        let response = crate::test_utils::history::decode(&wire(), mode, frames)
            .expect("an audio reply decodes");
        assert!(
            matches!(response.choice.as_slice(), [block @ AssistantContent::Text(text)]
                if text.text == "hi"
                    && block.native_item() == Some(&serde_json::json!({"audio": {"id": "audio_1"}}))),
            "{mode:?}: {:?}",
            response.choice
        );
    }
}

/// The assistant message `wire` sends back for `turn`: the second message
/// of a request that continues it, prepared as the driver prepares it.
pub(super) fn replayed(wire: &Chat, turn: crate::message::AssistantMessage) -> serde_json::Value {
    use crate::message::Message;
    use crate::wire::Operation;
    let request = CompletionRequest::from(vec![
        Message::user("q"),
        Message::Assistant(turn),
        Message::user("next"),
    ]);
    let request = crate::operation::Completion::prepare(request, &wire.describe())
        .expect("the request prepares");
    let body = json_body(&wire.encode(request, Mode::Unary).expect("encodes").request);
    body["messages"][1].clone()
}
