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
use crate::providers::openai::wire::{OPENAI, OpenAIConfig};
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

/// Drive `body` through a gateway whose quirks end a `[DONE]` stream that
/// names no finish reason.
async fn lenient_stream(
    body: &'static str,
) -> Result<crate::completion::CompletionResponse, ProviderError> {
    use crate::providers::openai::wire::{Dialect, Quirks};

    // Built at run time: the quirk is a `const fn` callers use in constants.
    let dialect = Dialect::gateway("acme", "https://api.acme.test/v1", "ACME_API_KEY")
        .with_quirks(Quirks::openai().done_without_finish_reason());
    let bound = crate::driver::Model::new(
        OpenAIConfig::new("k").with_dialect(&dialect).chat("m"),
        MockStreamingClient {
            sse_bytes: Bytes::from_static(body.as_bytes()),
        },
    );
    let mut response = bound.stream(prompt("hi")).expect("the stream opens");
    while response.next().await.is_some() {}
    response.finish().await
}

/// With the quirk, `[DONE]` ends a text turn that named no reason as a
/// stop, as pi's `compat.supportsFinishReason: false` does.
#[tokio::test]
async fn the_done_quirk_ends_a_text_turn_as_a_stop() {
    const BODY: &str = concat!(
        "data: {\"object\":\"chat.completion.chunk\",\"id\":\"chatcmpl-1\",",
        "\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"complete answer\"},",
        "\"finish_reason\":null}]}\n\n",
        "data: [DONE]\n\n",
    );
    let response = lenient_stream(BODY)
        .await
        .expect("the quirk ends the turn at the sentinel");
    assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
    assert_eq!(
        canonical(&response.choice),
        vec![AssistantContent::text("complete answer")]
    );
}

/// With the quirk, a turn holding a tool call ends as a tool call, and the
/// call is delivered whole.
#[tokio::test]
async fn the_done_quirk_ends_a_tool_call_turn_as_tool_calls() {
    const BODY: &str = concat!(
        "data: {\"object\":\"chat.completion.chunk\",\"id\":\"chatcmpl-1\",\"model\":\"m\",",
        "\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",",
        "\"type\":\"function\",\"function\":{\"name\":\"read\",\"arguments\":\"{\\\"path\\\"\"}}]}}]}\n\n",
        "data: {\"object\":\"chat.completion.chunk\",\"id\":\"chatcmpl-1\",\"model\":\"m\",",
        "\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,",
        "\"function\":{\"arguments\":\":\\\"README.md\\\"}\"}}]}}]}\n\n",
        "data: [DONE]\n\n",
    );
    let response = lenient_stream(BODY)
        .await
        .expect("the quirk ends the turn at the sentinel");
    assert_eq!(response.finish_reason(), Some(FinishReason::ToolCalls));
    let [AssistantContent::ToolCall(call)] = response.choice.as_slice() else {
        panic!("the turn is one tool call: {:?}", response.choice);
    };
    assert_eq!(call.function.name, "read");
    assert_eq!(
        call.function.arguments_value(),
        serde_json::json!({"path": "README.md"})
    );
}

/// The quirk never turns an empty stream into an answer: `[DONE]` with no
/// chunk before it, or only frames the wire does not model, is truncated.
#[tokio::test]
async fn the_done_quirk_still_fails_a_stream_with_no_chunk() {
    for body in ["data: [DONE]\n\n", "data: \"noise\"\n\ndata: [DONE]\n\n"] {
        let error = lenient_stream(body)
            .await
            .expect_err("a stream with no chunk has no response");
        assert!(
            matches!(error, ProviderError::Truncated),
            "{body}: {error:?}"
        );
    }
}

/// The quirk reads only the sentinel: a stream that stops without `[DONE]`
/// or a finish reason was still cut short.
#[tokio::test]
async fn the_done_quirk_does_not_end_a_stream_without_the_sentinel() {
    const BODY: &str = concat!(
        "data: {\"object\":\"chat.completion.chunk\",\"id\":\"chatcmpl-1\",",
        "\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"}}]}\n\n",
    );
    let error = lenient_stream(BODY)
        .await
        .expect_err("a stream cut before the sentinel has no response");
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

/// A stream's `raw` is the `chat.completion` its chunks add up to, read by
/// the paths a unary body is read by: the escape hatch for every provider
/// field this wire does not normalize.
#[tokio::test]
async fn the_streamed_raw_reads_back_as_the_unary_document() {
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
    let folded = response.finish().await.expect("the stream ends");

    let document = &folded.raw;
    let usage = &document["usage"];
    assert_eq!(usage["prompt_tokens"], 15);
    assert_eq!(usage["completion_tokens"], 1);
    assert_eq!(usage["total_tokens"], 16);
    assert_eq!(document["model"], "gpt-4.1-nano-2025-04-14");
    assert_eq!(document["id"], recorded_chunk_field("id"));
    assert_eq!(document["object"], "chat.completion");
    assert_eq!(document["choices"][0]["finish_reason"], "stop");
    assert_eq!(document["choices"][0]["message"]["content"], "pong");
    assert_eq!(document["choices"][0]["message"]["role"], "assistant");

    // The provider-native fields the wire does not normalize sit where the
    // unary body has them.
    assert_eq!(document["service_tier"], "default");
    assert_eq!(
        document["system_fingerprint"],
        recorded_chunk_field("system_fingerprint")
    );
    assert!(
        document.get("obfuscation").is_none(),
        "stream padding is not part of the document: {document}"
    );

    // And the normalized view agrees with it.
    assert_eq!(folded.usage.input_tokens, Some(15));
    assert_eq!(folded.model(), Some("gpt-4.1-nano-2025-04-14"));
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

/// pi's id rule for this wire (`openai-completions.js`, `normalizeToolCallId`).
/// The hashes are pi's `shortHash`. Mistral takes the same ids (checked live
/// on eight models), where pi derives nine alphanumerics.
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
        "call_abc123"
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

/// A gateway that reports usage on the choice rather than the top level
/// still has its usage observed: the decoder accepts `choices[0].usage` for
/// `Finish`, so the observation must see the same counts. The raw-bytes
/// projector reads only `/usage`, and reports nothing.
#[test]
fn per_choice_usage_is_observed_as_the_decoder_reads_it() {
    use crate::observe::{Action, AdapterContext, ObservationLog, Subject};
    let log = std::sync::Arc::new(ObservationLog::default());
    let context = AdapterContext::new(log.clone(), Subject::default(), "call");
    let mut attempt = context
        .attempt_for(&http::Request::new(()), "/chat/completions")
        .expect("an attempt starts");
    let payload = br#"{"id":"c","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop","usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}]}"#;
    attempt.project(|sink| ChatDecoder::project(payload, sink));
    drop(attempt);
    let observed: Vec<(Option<u64>, Option<u64>)> = log
        .trace()
        .observations
        .iter()
        .filter_map(|observation| match &observation.action {
            Action::Adapter { observation } => match &observation.event {
                AdapterEvent::Usage { usage } => Some((usage.input_tokens, usage.output_tokens)),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert_eq!(
        observed,
        [(Some(3), Some(4))],
        "per-choice usage is observed"
    );
}
