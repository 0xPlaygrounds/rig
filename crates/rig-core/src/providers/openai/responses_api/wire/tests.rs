//! The Responses wire, driven from recorded bytes.
//!
//! The bodies are read out of the cassettes rather than copied into this
//! file, so a recorded turn and the assertion about it cannot drift. The
//! cassettes are read-only here.

use super::*;
use crate::completion::CompletionRequest;
use crate::message::{self, Message};
use crate::providers::chatgpt::DIALECT as CHATGPT;
use crate::providers::xai::DIALECT as XAI;
use crate::test_utils::json_body;
use crate::test_utils::{MockStreamingClient, RecordingHttpClient};
use crate::wire::Mode;
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
    CompletionRequest::new("say hi")
}

/// Fold a recorded unary body through the wire, as `Model::call`
/// does.
async fn folded_unary(wire: Responses, body: &str) -> completion::CompletionResponse {
    crate::driver::Model::new(wire, RecordingHttpClient::new(Bytes::from(body.to_owned())))
        .call(prompt())
        .await
        .expect("the recorded body folds")
}

/// Fold a recorded SSE body through the wire, as `Model::stream`
/// does, draining every event first.
async fn folded_stream(wire: Responses, body: &str) -> completion::CompletionResponse {
    let bound = crate::driver::Model::new(
        wire,
        MockStreamingClient {
            sse_bytes: Bytes::from(body.to_owned()),
        },
    );
    let mut response = bound.stream(prompt()).expect("the stream opens");
    while response.next().await.is_some() {}
    response
        .finish()
        .await
        .expect("the stream produced a terminal record")
}

fn openai() -> Responses {
    OpenAIConfig::new("test-key").responses("gpt-4o")
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
    let wire = OpenAIConfig::with_key(&CHATGPT, "test-token").responses("gpt-5.4");

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
    json_body(&encoded.request)
}

/// The bare [`prompt`] with a history of its own.
fn turn(chat_history: Vec<Message>) -> CompletionRequest {
    CompletionRequest {
        chat_history,
        ..prompt()
    }
}

fn chatgpt() -> Responses {
    OpenAIConfig::with_key(&CHATGPT, "test-token").responses("gpt-5.4")
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
    OpenAIConfig::with_key(&XAI, "test-key").responses("grok-4")
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
    assert_eq!(encoded.request.uri(), "https://api.x.ai/v1/responses");

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
                message::UserContent::tool_result(
                    crate::message::CallId::from_dual_wire("result-id", "call-id".to_owned()),
                    crate::message::ToolName::new("tool").expect("tool name"),
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
                    message::AssistantContent::Reasoning(
                        message::Reasoning {
                            id: Some("rs_1".to_owned()),
                            content: vec![
                                message::ReasoningContent::Summary("explain".to_owned()),
                                message::ReasoningContent::Redacted {
                                    data: "opaque-redacted".to_owned(),
                                },
                            ],
                        }
                        .sealed("xai"),
                    ),
                    message::AssistantContent::tool_call(
                        "call_1",
                        crate::message::ToolName::new("my_tool").expect("tool name"),
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
    let error = crate::driver::Model::new(
        xai(),
        RecordingHttpClient::new(Bytes::from_static(
            br#"{"error":{"message":"no capacity","code":"overloaded"}}"#,
        )),
    )
    .call(prompt())
    .await
    .expect_err("an error envelope fails the call");

    assert!(
        error.to_string().contains("no capacity"),
        "the provider's own message must survive: {error}"
    );
}

// ── totality: every output item survives decode, history and replay ─────

/// The index of `item`'s variant. Exhaustive and wildcard-free, so a new
/// [`Output`] variant does not compile until it is numbered here, and the
/// coverage check below then fails until a sample decodes to it.
fn output_variant_index(item: &super::super::Output) -> usize {
    use super::super::Output;
    match item {
        Output::Message(_) => 0,
        Output::FunctionCall(_) => 1,
        Output::Reasoning { .. } => 2,
        Output::Unknown(_) => 3,
    }
}

const OUTPUT_VARIANTS: usize = 4;

/// One output item, and whether it lifts to a canonical part (which replays
/// re-encoded) or is native (which replays verbatim).
fn output_samples() -> Vec<(serde_json::Value, bool)> {
    use serde_json::json;
    vec![
        (
            json!({"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                   "content": [{"type": "output_text", "text": "hi", "annotations": []}]}),
            true,
        ),
        (
            json!({"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "lookup",
                   "arguments": "{\"q\":\"x\"}", "status": "completed"}),
            true,
        ),
        (
            json!({"type": "reasoning", "id": "rs_1", "encrypted_content": "enc",
                   "summary": [{"type": "summary_text", "text": "s"}]}),
            true,
        ),
        // Loss 2: must-replay compaction.
        (
            json!({"type": "compaction", "id": "cmp_1", "encrypted_content": "opaque", "status": "completed"}),
            false,
        ),
        // Loss 3: items with no canonical slot.
        (
            json!({"type": "computer_call", "id": "cu_1", "call_id": "call_c", "status": "completed",
                   "action": {"type": "click", "x": 1, "y": 2}, "pending_safety_checks": []}),
            false,
        ),
        (
            json!({"type": "local_shell_call", "id": "lsh_1", "call_id": "call_s", "status": "completed",
                   "action": {"type": "exec", "command": ["ls"], "env": {}}}),
            false,
        ),
        (
            json!({"type": "custom_tool_call", "id": "ctc_1", "call_id": "call_t", "name": "grammar", "input": "x+y"}),
            false,
        ),
        (
            json!({"type": "mcp_approval_request", "id": "mcpr_1", "server_label": "docs",
                   "name": "search", "arguments": "{}"}),
            false,
        ),
        (
            json!({"type": "web_search_call", "id": "ws_1", "status": "completed",
                   "action": {"type": "search", "query": "rig"}}),
            false,
        ),
        // An item type invented after this crate was written.
        (
            json!({"type": "novel_item_2027", "id": "nv_1", "data": {"a": [1, {"b": null}]}}),
            false,
        ),
    ]
}

/// `items` as one unary response body.
fn response_body(items: &[serde_json::Value]) -> serde_json::Value {
    serde_json::json!({
        "id": "resp_1", "object": "response", "created_at": 0, "status": "completed",
        "model": "gpt-5.2", "output": items, "tools": [],
        "usage": {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8}
    })
}

/// `items` as the SSE body a stream states them with: each item added,
/// its text or arguments as deltas, then done, then the terminal response.
fn stream_body(items: &[serde_json::Value]) -> String {
    use serde_json::json;
    let mut events = Vec::new();
    for (index, item) in items.iter().enumerate() {
        let mut added = item.clone();
        if item["type"] == "message" {
            added["content"] = json!([]);
            added["status"] = json!("in_progress");
        }
        events.push(
            json!({"type": "response.output_item.added", "output_index": index, "item": added}),
        );
        if item["type"] == "message" {
            events.push(json!({"type": "response.output_text.delta", "item_id": item["id"],
                               "output_index": index, "content_index": 0, "delta": item["content"][0]["text"]}));
        }
        if item["type"] == "function_call" {
            events.push(
                json!({"type": "response.function_call_arguments.delta", "item_id": item["id"],
                               "output_index": index, "delta": item["arguments"]}),
            );
        }
        events.push(
            json!({"type": "response.output_item.done", "output_index": index, "item": item}),
        );
    }
    events.push(json!({"type": "response.completed", "response": response_body(items)}));
    events
        .iter()
        .enumerate()
        .map(|(sequence, event)| {
            let mut event = event.clone();
            event["sequence_number"] = json!(sequence);
            format!("data: {event}\n\n")
        })
        .collect()
}

/// `turn` replayed on `wire` as history: the request's input items after
/// the first user message, up to the follow-up.
fn replayed_items(
    wire: &Responses,
    turn: Vec<message::AssistantContent>,
) -> Vec<serde_json::Value> {
    let request = CompletionRequest::new(Message::user("next")).messages([
        Message::user("first"),
        Message::Assistant {
            id: None,
            content: turn,
        },
    ]);
    let body = json_body(
        &wire
            .encode(request, Mode::Unary)
            .expect("the follow-up encodes")
            .request,
    );
    let input = body["input"].as_array().cloned().unwrap_or_default();
    input[1..input.len() - 1].to_vec()
}

#[tokio::test]
async fn every_output_item_survives_decode_history_and_replay_in_both_modes() {
    let mut covered = [false; OUTPUT_VARIANTS];
    for (item, canonical) in output_samples() {
        let typed: super::super::Output =
            serde_json::from_value(item.clone()).expect("every item decodes");
        let index = output_variant_index(&typed);
        assert!(
            index < OUTPUT_VARIANTS,
            "number the new variant within OUTPUT_VARIANTS"
        );
        covered[index] = true;

        let items = std::slice::from_ref(&item);
        let buffered = folded_unary(openai(), &response_body(items).to_string()).await;
        let streamed = folded_stream(openai(), &stream_body(items)).await;
        assert_eq!(
            buffered.choice, streamed.choice,
            "{}: both modes agree",
            item["type"]
        );
        assert_eq!(
            buffered.choice.len(),
            1,
            "{}: one part, never dropped",
            item["type"]
        );
        assert_eq!(
            matches!(buffered.choice[0], message::AssistantContent::Native(_)),
            !canonical,
            "{}: lifted as expected",
            item["type"]
        );

        let replayed = replayed_items(&openai(), buffered.choice);
        assert_eq!(replayed.len(), 1, "{}: one input item", item["type"]);
        if canonical {
            assert_eq!(replayed[0]["type"], item["type"]);
        } else {
            assert_eq!(replayed[0], item, "native items replay verbatim");
        }
    }
    assert!(
        covered.iter().all(|seen| *seen),
        "a sample for every variant: {covered:?}"
    );
}

/// Items replay in the order the turn produced them: reasoning is no longer
/// hoisted ahead of the items it followed (loss 6).
#[tokio::test]
async fn a_turn_mixing_every_item_replays_in_order() {
    let items: Vec<_> = output_samples().into_iter().map(|(item, _)| item).collect();
    let buffered = folded_unary(openai(), &response_body(&items).to_string()).await;
    let streamed = folded_stream(openai(), &stream_body(&items)).await;
    assert_eq!(buffered.choice, streamed.choice);
    let replayed = replayed_items(&openai(), buffered.choice);
    let types: Vec<_> = replayed.iter().map(|item| item["type"].clone()).collect();
    let expected: Vec<_> = items.iter().map(|item| item["type"].clone()).collect();
    assert_eq!(types, expected);
}

#[tokio::test]
async fn responses_native_items_never_reach_another_issuer_or_wire_format() {
    use serde_json::json;
    let compaction = json!({"type": "compaction", "id": "cmp_1", "encrypted_content": "opaque"});
    let turn = folded_unary(
        openai(),
        &response_body(std::slice::from_ref(&compaction)).to_string(),
    )
    .await
    .choice;

    // The same Responses format under another issuer (xAI): left out, and a
    // native-only turn leaves no assistant item at all.
    let xai = OpenAIConfig::with_key(&XAI, "k").responses("grok-4");
    assert!(replayed_items(&xai, turn.clone()).is_empty());

    // Another wire format: the Messages wire never sends it.
    let messages =
        crate::providers::anthropic::AnthropicConfig::new("k").completion("claude-haiku-4-5");
    let request = CompletionRequest::new(Message::user("next"))
        .messages([
            Message::user("first"),
            Message::Assistant {
                id: None,
                content: turn,
            },
        ])
        .max_tokens(16);
    let body = json_body(
        &messages
            .encode(request, Mode::Unary)
            .expect("encodes")
            .request,
    );
    assert!(!body.to_string().contains("compaction"), "{body}");
}
