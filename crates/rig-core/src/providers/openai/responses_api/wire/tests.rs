//! The Responses wire, driven from recorded bytes.
//!
//! The bodies are read out of the cassettes rather than copied into this
//! file, so a recorded turn and the assertion about it cannot drift. The
//! cassettes are read-only here.

use super::*;
use crate::completion::{CompletionModel, CompletionRequest};
use crate::driver::Bound;
use crate::message::{self, Message};
use crate::providers::chatgpt::DIALECT as CHATGPT;
use crate::providers::xai::DIALECT as XAI;
use crate::test_utils::{MockStreamingClient, RecordingHttpClient};
use crate::wire::{Body, Mode};
use bytes::Bytes;
use futures::StreamExt;

// ── the cassettes, as bytes ─────────────────────────────────────────────

/// One recorded interaction's reply body, read out of a cassette.
///
/// The format is one or more `when:`/`then:` documents; the reply body is
/// either a single-quoted scalar (a JSON body, with `''` for a quote) or a
/// `|+` literal block (an SSE body). Parsed here rather than with a YAML
/// dependency, and never written.
fn cassette_body(path: &str) -> String {
    let file = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../rig-cassette/fixtures/cassettes")
        .join(path);
    let text = std::fs::read_to_string(&file)
        .unwrap_or_else(|error| panic!("{} should be readable: {error}", file.display()));
    let reply = text
        .split_once("\nthen:")
        .unwrap_or_else(|| panic!("{} should record a reply", file.display()))
        .1;
    let body = reply
        .split_once("  body: ")
        .unwrap_or_else(|| panic!("{} should record a reply body", file.display()))
        .1;
    match body.strip_prefix("|+\n") {
        // A literal block: four-space-indented lines up to the first line
        // that is neither blank nor indented.
        Some(block) => {
            let mut out = String::new();
            for line in block.lines() {
                match line.strip_prefix("    ") {
                    Some(line) => out.push_str(line),
                    None if line.trim().is_empty() => {}
                    None => break,
                }
                out.push('\n');
            }
            out
        }
        // A single-quoted scalar on one line.
        None => body
            .lines()
            .next()
            .unwrap_or_default()
            .trim()
            .trim_start_matches('\'')
            .trim_end_matches('\'')
            .replace("''", "'"),
    }
}

/// The unary body of the turn a recorded SSE body streams: the response
/// object the provider itself restates on `response.completed`, which is
/// byte-for-byte the shape its unary endpoint answers with.
fn terminal_response_body(sse: &str) -> String {
    let event = sse
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|data| serde_json::from_str::<serde_json::Value>(data).ok())
        .find(|event| {
            matches!(
                event.get("type").and_then(serde_json::Value::as_str),
                Some("response.completed") | Some("response.incomplete")
            )
        })
        .expect("the recorded stream ends with a terminal response event");
    event
        .get("response")
        .expect("a terminal event carries its response object")
        .to_string()
}

fn prompt() -> CompletionRequest {
    CompletionRequest {
        model: None,
        chat_history: vec![Message::user("say hi")],
        documents: Vec::new(),
        tools: Vec::new(),
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    }
}

/// Fold a recorded unary body through the wire, as [`crate::driver::call`]
/// does.
async fn folded_unary(wire: Responses, body: &str) -> completion::CompletionResponse {
    Bound::new(wire, RecordingHttpClient::new(Bytes::from(body.to_owned())))
        .completion(prompt())
        .await
        .expect("the recorded body folds")
}

/// Fold a recorded SSE body through the wire, as [`crate::driver::stream`]
/// does, draining every event first.
async fn folded_stream(wire: Responses, body: &str) -> completion::CompletionResponse {
    let bound = Bound::new(
        wire,
        MockStreamingClient {
            sse_bytes: Bytes::from(body.to_owned()),
        },
    );
    let mut response = bound.stream(prompt()).await.expect("the stream opens");
    while response.next().await.is_some() {}
    response
        .finish()
        .expect("the stream produced a terminal record")
}

fn openai() -> Responses {
    OpenAI::new("test-key").responses("gpt-4o")
}

// ── the property the model exists for ───────────────────────────────────

/// The recorded stream of one turn and that same turn's unary body — the
/// response object its `response.completed` event restates — fold to one
/// answer.
#[tokio::test]
async fn a_unary_body_and_the_stream_of_the_same_turn_fold_alike() {
    let sse = cassette_body("openai/response_identity/responses_streaming_carries_identity.yaml");
    let unary = terminal_response_body(&sse);

    let buffered = folded_unary(openai(), &unary).await;
    let streamed = folded_stream(openai(), &sse).await;

    assert_eq!(buffered.choice, streamed.choice);
    assert_eq!(buffered.usage, streamed.usage);
    assert_eq!(buffered.finish_reason(), streamed.finish_reason());
    assert_eq!(buffered.model, streamed.model);
    assert_eq!(buffered.message_id, streamed.message_id);
    assert_eq!(buffered.response_id, streamed.response_id);
    assert_eq!(
        text_of(&buffered),
        Some("stream identity probe".to_owned()),
        "the recorded turn's text must survive both paths"
    );
}

/// A recorded tool turn: the call the provider restates in its unary body is
/// the call its stream assembles from fragments.
#[tokio::test]
async fn a_unary_tool_turn_and_its_stream_fold_alike() {
    let sse = cassette_body("openai/streaming_grammar/tool_then_followup_text.yaml");
    let unary = terminal_response_body(&sse);

    let buffered = folded_unary(openai(), &unary).await;
    let streamed = folded_stream(openai(), &sse).await;

    assert_eq!(buffered.choice, streamed.choice);
    assert_eq!(buffered.usage, streamed.usage);
    assert_eq!(buffered.finish_reason(), streamed.finish_reason());
    assert!(
        buffered
            .choice
            .iter()
            .any(|content| matches!(content, message::AssistantContent::ToolCall(_))),
        "the recorded turn calls a tool: {:?}",
        buffered.choice
    );
}

/// ChatGPT answers a unary request with a replayed event stream, so the same
/// recorded bytes go through `call` and through `stream`; both fold to the
/// same turn.
#[tokio::test]
async fn a_chatgpt_replayed_body_folds_the_same_unary_and_streamed() {
    let sse = cassette_body("chatgpt/codex_tool_args/zero_argument_tool_call_nonstreaming.yaml");
    let wire = OpenAI::with_key(&CHATGPT, "test-token").responses("gpt-5.4");

    let buffered = folded_unary(wire.clone(), &sse).await;
    let streamed = folded_stream(wire, &sse).await;

    assert_eq!(buffered.choice, streamed.choice);
    assert_eq!(buffered.usage, streamed.usage);
    assert_eq!(buffered.finish_reason(), streamed.finish_reason());
    assert_eq!(buffered.provider, "chatgpt");
    assert!(
        buffered
            .choice
            .iter()
            .any(|content| matches!(content, message::AssistantContent::ToolCall(_))),
        "the recorded turn calls a tool: {:?}",
        buffered.choice
    );
}

fn text_of(response: &completion::CompletionResponse) -> Option<String> {
    response.choice.iter().find_map(|content| match content {
        message::AssistantContent::Text(text) => Some(text.text.clone()),
        _ => None,
    })
}

// ── what each dialect sends ─────────────────────────────────────────────

fn encoded_body(wire: &Responses, mode: Mode) -> serde_json::Value {
    encoded_body_of(wire, prompt(), mode)
}

/// One request's body, for a turn other than the bare [`prompt`].
fn encoded_body_of(wire: &Responses, request: CompletionRequest, mode: Mode) -> serde_json::Value {
    let encoded = wire.encode(request, mode).expect("the request encodes");
    let request = encoded
        .requests
        .first()
        .expect("a Responses request is one request");
    let Body::Bytes(body) = request.body() else {
        panic!("a Responses body is bytes");
    };
    serde_json::from_slice(body).expect("the body is JSON")
}

/// The bare [`prompt`] with a history of its own.
fn turn(chat_history: Vec<Message>) -> CompletionRequest {
    CompletionRequest {
        chat_history,
        ..prompt()
    }
}

fn chatgpt() -> Responses {
    OpenAI::with_key(&CHATGPT, "test-token").responses("gpt-5.4")
}

#[test]
fn a_streamed_request_asks_for_a_stream_and_a_unary_one_does_not() {
    let wire = openai();
    assert_eq!(
        encoded_body(&wire, Mode::Streaming).get("stream"),
        Some(&serde_json::Value::Bool(true))
    );
    assert_eq!(encoded_body(&wire, Mode::Unary).get("stream"), None);
    assert_eq!(
        wire.encode(prompt(), Mode::Unary)
            .expect("the request encodes")
            .framing,
        Framing::Whole
    );
}

/// ChatGPT's gateway answers with an event stream whatever was asked for, so
/// even a unary call asks for one and accepts a reply that names no content
/// type.
#[test]
fn the_chatgpt_dialect_always_streams_and_relaxes_the_content_type() {
    let wire = chatgpt();
    let encoded = wire
        .encode(prompt(), Mode::Unary)
        .expect("the request encodes");

    assert_eq!(encoded.framing, Framing::Sse);
    assert!(encoded.relaxed_content_type);
    assert_eq!(
        encoded_body(&wire, Mode::Unary).get("stream"),
        Some(&serde_json::Value::Bool(true))
    );
}

/// The codex gateway takes the turn and its tools; the sampling, storage and
/// structured-output parameters are not its to accept, and the reasoning
/// payload must be asked for because the gateway stores nothing.
#[test]
fn the_chatgpt_dialect_sends_only_the_codex_parameter_subset() {
    let wire = chatgpt();
    let body = encoded_body(&wire, Mode::Unary);

    assert_eq!(body.get("temperature"), None);
    assert_eq!(body.get("max_output_tokens"), None);
    assert_eq!(body.get("top_p"), None);
    assert_eq!(body.get("store"), Some(&serde_json::Value::Bool(false)));
    assert_eq!(
        body.get("include"),
        Some(&serde_json::json!(["reasoning.encrypted_content"]))
    );
    assert_eq!(
        body.get("instructions").and_then(serde_json::Value::as_str),
        Some("You are ChatGPT, a helpful AI assistant.")
    );
}

/// The gateway's own instructions lead, the caller's follow: a backend that
/// expects instructions of its own gets them ahead of the turn's preamble
/// rather than instead of it.
#[test]
fn the_chatgpt_dialect_merges_its_instructions_ahead_of_the_callers() {
    let body = encoded_body_of(
        &chatgpt(),
        turn(vec![
            Message::system("Respond tersely."),
            Message::user("say hi"),
        ]),
        Mode::Unary,
    );

    assert_eq!(
        body.get("instructions").and_then(serde_json::Value::as_str),
        Some("You are ChatGPT, a helpful AI assistant.\n\nRespond tersely.")
    );
}

/// ...and they are not stated twice when the caller's preamble already
/// carries them, which is what a replayed conversation's history looks like.
#[test]
fn the_chatgpt_dialect_does_not_repeat_instructions_the_caller_already_carries() {
    let carried = "You are ChatGPT, a helpful AI assistant.\n\nRespond tersely.";
    let body = encoded_body_of(
        &chatgpt(),
        turn(vec![Message::system(carried), Message::user("say hi")]),
        Mode::Unary,
    );

    assert_eq!(
        body.get("instructions").and_then(serde_json::Value::as_str),
        Some(carried)
    );
}

/// This gateway rejects the `system` role in `input` outright, so every
/// system message is lifted — the leading run *and* the mid-conversation
/// ones — leaving only the non-system turns as input items.
#[test]
fn the_chatgpt_dialect_lifts_every_system_message_into_instructions() {
    let body = encoded_body_of(
        &chatgpt(),
        turn(vec![
            Message::system("System one"),
            Message::user("hi"),
            Message::system("Mid-conversation instruction"),
            Message::user("again"),
        ]),
        Mode::Unary,
    );

    assert_eq!(
        body.get("instructions").and_then(serde_json::Value::as_str),
        Some(
            "You are ChatGPT, a helpful AI assistant.\n\nSystem one\n\nMid-conversation instruction"
        )
    );
    assert_eq!(
        body.get("input")
            .and_then(serde_json::Value::as_array)
            .map(Vec::len),
        Some(2),
        "only the two user turns remain as input: {body}"
    );
}

/// A turn whose terminal event restates the assembled output.
const CHATGPT_ASSEMBLED_OUTPUT: &str = r#"data: {"type":"response.output_text.delta","delta":"hi"}

data: {"type":"response.completed","response":{"id":"resp_chatgpt_raw","object":"response","created_at":1,"status":"completed","error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"model":"gpt-5.4","service_tier":"default","usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2},"output":[{"type":"message","id":"msg_chatgpt_raw","status":"completed","role":"assistant","content":[{"type":"output_text","annotations":[],"text":"hi"}]}],"tools":[]}}

data: [DONE]"#;

/// The same turn with an empty terminal `output`: the deltas are the only
/// place the content exists. A recorded shape, not a synthetic one.
const CHATGPT_EMPTY_OUTPUT: &str = r#"data: {"type":"response.output_text.delta","delta":"hi"}

data: {"type":"response.completed","response":{"id":"resp_chatgpt_raw","object":"response","created_at":1,"status":"completed","error":null,"incomplete_details":null,"instructions":null,"max_output_tokens":null,"model":"gpt-5.4","service_tier":"default","usage":{"input_tokens":1,"input_tokens_details":{"cached_tokens":0},"output_tokens":1,"output_tokens_details":{"reasoning_tokens":0},"total_tokens":2},"output":[],"tools":[]}}

data: [DONE]"#;

/// A gateway that answers every request with an event stream sends no reply
/// document of its own, so the terminal `response.completed` *is* the
/// document: `raw` must be that object — deserializable back into the wire
/// type and re-serializing value-equal, carrying the fields rig does not
/// normalize (`service_tier`) — whether or not its `output` restates the
/// turn, and the choice comes from the deltas either way.
#[tokio::test]
async fn a_chatgpt_reply_captures_the_terminal_response_object_as_raw() {
    for (body, case) in [
        (CHATGPT_ASSEMBLED_OUTPUT, "assembled output"),
        (CHATGPT_EMPTY_OUTPUT, "empty output"),
    ] {
        let response = folded_unary(chatgpt(), body).await;

        let typed: crate::providers::openai::responses_api::CompletionResponse =
            serde_json::from_value(response.raw.clone())
                .expect("raw must deserialize back into the wire type");
        assert_eq!(
            serde_json::to_value(&typed).expect("re-serialize"),
            response.raw,
            "{case}: the capture must be exactly what the wire type serializes to"
        );
        assert_eq!(response.raw["service_tier"], "default", "{case}");
        assert_eq!(typed.id, "resp_chatgpt_raw", "{case}");

        assert_eq!(
            response.choice,
            vec![message::AssistantContent::text("hi")],
            "{case}: the deltas are the content"
        );
        assert_eq!(response.usage.total_tokens, Some(2), "{case}");
        assert_eq!(
            response.identity().response_id.as_deref(),
            Some("resp_chatgpt_raw"),
            "{case}"
        );
    }
}

fn xai() -> Responses {
    OpenAI::with_key(&XAI, "test-key").responses("grok-4")
}

/// xAI's endpoint lives under `/v1` and rejects top-level `instructions`, so
/// every system message — the leading run and the mid-conversation ones —
/// stays in `input` where it was, and the turn's documents follow the
/// preamble as the shared history conversion places them.
#[test]
fn the_xai_dialect_keeps_every_system_message_in_input() {
    let wire = xai();
    let encoded = wire
        .encode(prompt(), Mode::Unary)
        .expect("the request encodes");
    assert_eq!(
        encoded.requests.first().expect("one request").uri(),
        "https://api.x.ai/v1/responses"
    );

    let body = encoded_body_of(
        &wire,
        CompletionRequest {
            documents: vec![crate::completion::Document {
                id: "doc_1".to_owned(),
                text: "Definition of glarb-glarb: an ancient tool.".to_owned(),
                additional_props: Default::default(),
            }],
            ..turn(vec![
                Message::system("System prompt"),
                Message::assistant("Earlier assistant turn"),
                Message::system("Mid-conversation instruction"),
                Message::user("What is glarb-glarb?"),
            ])
        },
        Mode::Unary,
    );

    assert_eq!(body.get("instructions"), None, "{body}");
    let input = body["input"].as_array().expect("input is an array");
    let roles: Vec<_> = input
        .iter()
        .map(|item| item["role"].as_str().unwrap_or_default())
        .collect();
    assert_eq!(
        roles,
        ["system", "user", "assistant", "system", "user"],
        "{body}"
    );
    assert_eq!(
        input
            .iter()
            .filter(|item| item.to_string().contains("<file id: doc_1>"))
            .count(),
        1,
        "the document rides one user item, after the preamble: {body}"
    );
    assert!(input[1].to_string().contains("<file id: doc_1>"), "{body}");
}

/// A user turn interleaving text and a tool result keeps its order on the
/// wire: text before the result is one message item, the result is a
/// `function_call_output` under its call id, text after is another message.
#[test]
fn the_xai_dialect_folds_tool_results_between_user_text_in_order() {
    let body = encoded_body_of(
        &xai(),
        turn(vec![Message::User {
            content: vec![
                message::UserContent::text("before"),
                message::UserContent::tool_result_with_call_id(
                    "result-id",
                    "call-id".to_owned(),
                    "tool",
                    vec![message::ToolResultContent::json(
                        serde_json::json!({ "ok": true }),
                    )],
                ),
                message::UserContent::text("after"),
            ],
        }]),
        Mode::Unary,
    );

    let input = body["input"].as_array().expect("input is an array");
    assert_eq!(input.len(), 3, "{body}");
    assert_eq!(input[0]["type"], "message");
    assert_eq!(input[0]["role"], "user");
    assert_eq!(input[0]["content"][0]["text"], "before");
    assert_eq!(input[1]["type"], "function_call_output");
    assert_eq!(input[1]["call_id"], "call-id");
    assert_eq!(input[1]["output"], r#"{"ok":true}"#);
    assert_eq!(input[2]["type"], "message");
    assert_eq!(input[2]["content"][0]["text"], "after");
}

/// A replayed reasoning turn goes back under the id the wire issued, its
/// summary as `summary` and its opaque block as the one `encrypted_content`
/// — never as summary text — ahead of the tool call it preceded.
#[test]
fn the_xai_dialect_replays_reasoning_by_wire_id_with_its_encrypted_payload() {
    let body = encoded_body_of(
        &xai(),
        turn(vec![
            Message::user("Use the tool."),
            Message::Assistant {
                id: Some("msg_1".to_owned()),
                content: vec![
                    message::AssistantContent::Reasoning(message::Reasoning {
                        provider: None,
                        id: Some("rs_1".to_owned()),
                        content: vec![
                            message::ReasoningContent::Summary("explain".to_owned()),
                            message::ReasoningContent::Redacted {
                                data: "opaque-redacted".to_owned(),
                            },
                        ],
                    }),
                    message::AssistantContent::tool_call(
                        "call_1",
                        "my_tool",
                        serde_json::json!({"arg": "value"}),
                    ),
                ],
            },
        ]),
        Mode::Unary,
    );

    let input = body["input"].as_array().expect("input is an array");
    assert_eq!(input.len(), 3, "{body}");
    let reasoning = &input[1];
    assert_eq!(reasoning["type"], "reasoning");
    assert_eq!(reasoning["id"], "rs_1");
    assert_eq!(
        reasoning["summary"],
        serde_json::json!([{"type": "summary_text", "text": "explain"}])
    );
    assert_eq!(reasoning["encrypted_content"], "opaque-redacted");
    assert_eq!(reasoning.get("content"), None, "{reasoning}");
    assert_eq!(input[2]["type"], "function_call");
    assert_eq!(input[2]["call_id"], "call_1");
    assert_eq!(input[2]["name"], "my_tool");
}

/// A success carrying the provider's error envelope instead of a response is
/// the provider's failure, not a decode defect — on the dialects that answer
/// that way.
#[tokio::test]
async fn an_error_envelope_on_a_success_fails_the_xai_call() {
    let error = Bound::new(
        xai(),
        RecordingHttpClient::new(Bytes::from_static(
            br#"{"error":{"message":"no capacity","code":"overloaded"}}"#,
        )),
    )
    .completion(prompt())
    .await
    .expect_err("an error envelope fails the call");

    assert!(
        error.to_string().contains("no capacity"),
        "the provider's own message must survive: {error}"
    );
}
