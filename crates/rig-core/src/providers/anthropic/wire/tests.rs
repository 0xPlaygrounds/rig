//! The Messages wire, driven from recorded bytes and no socket.
//!
//! The unary and streamed bodies below are the two cells of
//! `crates/rig-cassette/fixtures/cassettes/anthropic/raw_completion_parity_matrix/` — the same turn
//! recorded both ways. Folding them through the same decoder is the property
//! this port exists for, and it is checked here without a transport so a
//! failure names the decoder rather than the harness.

use super::*;
use crate::message::AssistantContent;
use crate::test_utils::json_body;
use crate::wire::secret::tests::a_config_reloads_without_its_credential;
use crate::wire::{Framing, Mode, Wire};

/// `text_turn_parity.yaml`'s reply body, verbatim.
const UNARY: &str = r#"{"content":[{"text":"parity probe","type":"text"}],"id":"msg_REDACTED_1","model":"claude-haiku-4-5-20251001","role":"assistant","stop_details":null,"stop_reason":"end_turn","stop_sequence":null,"type":"message","usage":{"cache_creation":{"ephemeral_1h_input_tokens":0,"ephemeral_5m_input_tokens":0},"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"inference_geo":"not_available","input_tokens":14,"output_tokens":6,"service_tier":"standard"}}"#;

/// `streamed_text_turn_parity.yaml`'s reply body, verbatim.
const STREAMED: &str = concat!(
    "event: message_start\n",
    r#"data: {"message":{"content":[],"id":"msg_REDACTED_1","model":"claude-haiku-4-5-20251001","role":"assistant","stop_details":null,"stop_reason":null,"stop_sequence":null,"type":"message","usage":{"cache_creation":{"ephemeral_1h_input_tokens":0,"ephemeral_5m_input_tokens":0},"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"inference_geo":"not_available","input_tokens":14,"output_tokens":1,"service_tier":"standard"}},"type":"message_start"}"#,
    "\n\nevent: content_block_start\n",
    r#"data: {"content_block":{"text":"","type":"text"},"index":0,"type":"content_block_start"}"#,
    "\n\nevent: ping\ndata: {\"type\":\"ping\"}\n\n",
    "event: content_block_delta\n",
    r#"data: {"delta":{"text":"p","type":"text_delta"},"index":0,"type":"content_block_delta"}"#,
    "\n\nevent: content_block_delta\n",
    r#"data: {"delta":{"text":"arity probe","type":"text_delta"},"index":0,"type":"content_block_delta"}"#,
    "\n\nevent: content_block_stop\n",
    r#"data: {"index":0,"type":"content_block_stop"}"#,
    "\n\nevent: message_delta\n",
    r#"data: {"delta":{"stop_details":null,"stop_reason":"end_turn","stop_sequence":null},"type":"message_delta","usage":{"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"input_tokens":14,"output_tokens":6}}"#,
    "\n\n",
);

fn wire() -> Messages {
    AnthropicConfig::new("sk-test").completion("claude-haiku-4-5")
}

fn request() -> CompletionRequest {
    CompletionRequest::new("Reply with exactly: parity probe").max_tokens(32)
}

/// Fold a recorded reply body through the wire's own decoder, read in the
/// mode the driver would have read it in — which is also what decides the
/// framing, exactly as `encode` decides it.
fn fold(body: &str, mode: Mode) -> crate::completion::CompletionResponse {
    let frames: Vec<WireFrame> = match mode {
        Mode::Streaming => crate::http_client::framing::SseFramer::new()
            .push(body.as_bytes())
            .filter(|event| !event.data.trim().is_empty())
            .map(|event| WireFrame::Text(event.data))
            .collect(),
        Mode::Unary => vec![WireFrame::Text(body.to_owned())],
    };
    let mut response = crate::test_utils::decode_reply(
        &wire(),
        &request(),
        mode,
        frames,
        serde_json::from_str(body).unwrap_or(serde_json::Value::Null),
    )
    .expect("the recorded reply folds");
    response.provider_request_id = Some("req_REDACTED_1".to_owned());
    response
}

#[test]
fn the_same_turn_folds_identically_whether_it_was_buffered_or_streamed() {
    let buffered = fold(UNARY, Mode::Unary);
    let streamed = fold(STREAMED, Mode::Streaming);

    assert_eq!(buffered.choice, streamed.choice);
    assert_eq!(buffered.usage, streamed.usage);
    assert_eq!(buffered.model, streamed.model);
    assert_eq!(buffered.finish_reason(), streamed.finish_reason());
    assert_eq!(buffered.message_id, streamed.message_id);
    assert_eq!(buffered.provider_request_id, streamed.provider_request_id);
    assert_eq!(
        buffered.choice.first(),
        Some(&AssistantContent::text("parity probe"))
    );
    assert_eq!(buffered.usage.output_tokens, Some(6));
    assert_eq!(buffered.usage.input_tokens, Some(14));
}

#[test]
fn the_streaming_request_asks_for_a_stream_and_the_unary_one_does_not() {
    let unary = wire()
        .encode(request(), Mode::Unary)
        .expect("the request encodes");
    assert_eq!(unary.framing, Framing::Whole);
    let unary_body = json_body(&unary.request);
    assert_eq!(unary_body.get("stream"), None);

    let streaming = wire()
        .encode(request(), Mode::Streaming)
        .expect("the request encodes");
    assert_eq!(streaming.framing, Framing::Sse);
    assert_eq!(
        json_body(&streaming.request).get("stream"),
        Some(&serde_json::Value::Bool(true))
    );
}

#[test]
fn the_request_carries_the_key_version_and_endpoint() {
    let encoded = wire()
        .encode(request(), Mode::Unary)
        .expect("the request encodes");
    let request = &encoded.request;
    assert_eq!(request.uri().path(), "/v1/messages");
    assert_eq!(
        request
            .headers()
            .get("x-api-key")
            .and_then(|v| v.to_str().ok()),
        Some("sk-test")
    );
    assert_eq!(
        request
            .headers()
            .get("anthropic-version")
            .and_then(|v| v.to_str().ok()),
        Some(super::super::completion::ANTHROPIC_VERSION_LATEST)
    );
    assert_eq!(encoded.request_id_header, Some("request-id"));
}

#[test]
fn a_serialized_provider_never_carries_its_key() {
    let provider =
        AnthropicConfig::new("sk-live-do-not-leak").with_beta("prompt-caching-2024-07-31");
    a_config_reloads_without_its_credential(&provider, "sk-live-do-not-leak", |provider| {
        &provider.api_key
    });

    let wire = provider.completion("claude-haiku-4-5");
    let json = serde_json::to_string(&wire).expect("the wire serializes");
    assert!(!json.contains("sk-live-do-not-leak"));
    let restored: Messages = serde_json::from_str(&json).expect("the wire round-trips");
    assert_eq!(restored.model, "claude-haiku-4-5");
    assert_eq!(restored.provider.dialect, ANTHROPIC);
    assert!(restored.provider.api_key.is_empty());
}

#[test]
fn a_dialect_round_trips_by_name_and_rejects_an_unknown_one() {
    for dialect in [ANTHROPIC, ZAI, MINIMAX, MOONSHOT, XIAOMIMIMO] {
        let json = serde_json::to_string(&dialect).expect("a dialect serializes");
        assert_eq!(json, format!("\"{}\"", dialect.name));
        let restored: Dialect = serde_json::from_str(&json).expect("a dialect round-trips");
        assert_eq!(restored, dialect);
    }
    assert!(serde_json::from_str::<Dialect>("\"not-a-provider\"").is_err());
}

#[test]
fn a_gateway_defaults_max_tokens_to_its_one_documented_ceiling() {
    assert_eq!(
        ANTHROPIC.default_max_tokens("claude-haiku-4-5"),
        Some(64_000)
    );
    assert_eq!(ANTHROPIC.default_max_tokens("some-unknown-model"), None);
    // A gateway documents one ceiling rather than per-model limits, so an
    // unrecognized model still gets a usable default.
    assert_eq!(ZAI.default_max_tokens("some-unknown-model"), Some(4096));
    // `strict_tool_schemas` is a const quirk of a const dialect, so the
    // gateway's disagreement with Anthropic is a compile-time fact, not a
    // runtime one.
    const _: () = assert!(!ZAI.quirks.strict_tool_schemas);
    const _: () = assert!(ANTHROPIC.quirks.strict_tool_schemas);
}

#[test]
fn a_base_url_that_already_names_the_endpoint_is_trimmed() {
    for pasted in [
        "https://example.invalid/v1/messages",
        "https://example.invalid/messages",
        "https://example.invalid/v1",
        "https://example.invalid/",
    ] {
        assert_eq!(normalize_base_url(pasted), "https://example.invalid");
    }
}

// ── totality: every block survives decode, history and replay ───────────

/// The index of `content`'s variant. Exhaustive and wildcard-free, so a new
/// [`Content`] variant does not compile until it is numbered here, and the
/// coverage check below then fails until a sample decodes to it.
fn variant_index(content: &Content) -> usize {
    match content {
        Content::Text { .. } => 0,
        Content::Image { .. } => 1,
        Content::ToolUse { .. } => 2,
        Content::ToolResult { .. } => 3,
        Content::Document { .. } => 4,
        Content::Thinking { .. } => 5,
        Content::RedactedThinking { .. } => 6,
        Content::Unknown(_) => 7,
    }
}

const CONTENT_VARIANTS: usize = 8;

/// One block of a reply: the whole block a unary reply states, and the
/// opening block and deltas a stream states it with.
struct Sample {
    block: serde_json::Value,
    start: serde_json::Value,
    deltas: Vec<serde_json::Value>,
}

fn whole(block: serde_json::Value) -> Sample {
    Sample {
        start: block.clone(),
        block,
        deltas: Vec::new(),
    }
}

/// A block for every variant, every hosted-tool kind the context lists as
/// failing the reply, and a block type no Anthropic API has sent yet.
fn samples() -> Vec<Sample> {
    use serde_json::json;
    vec![
        Sample {
            block: json!({"type": "text", "text": "hi"}),
            start: json!({"type": "text", "text": ""}),
            deltas: vec![json!({"type": "text_delta", "text": "hi"})],
        },
        Sample {
            block: json!({"type": "thinking", "thinking": "hmm", "signature": "sig"}),
            start: json!({"type": "thinking", "thinking": "", "signature": ""}),
            deltas: vec![
                json!({"type": "thinking_delta", "thinking": "hmm"}),
                json!({"type": "signature_delta", "signature": "sig"}),
            ],
        },
        whole(json!({"type": "redacted_thinking", "data": "opaque"})),
        Sample {
            block: json!({"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {"q": "x"}}),
            start: json!({"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}),
            deltas: vec![json!({"type": "input_json_delta", "partial_json": "{\"q\":\"x\"}"})],
        },
        // Request-side kinds a reply should not carry: kept, not dropped.
        whole(
            json!({"type": "image", "source": {"type": "url", "url": "https://example.invalid/a.png"}}),
        ),
        whole(
            json!({"type": "tool_result", "tool_use_id": "toolu_0", "content": [{"type": "text", "text": "r"}]}),
        ),
        whole(
            json!({"type": "document", "source": {"type": "url", "url": "https://example.invalid/a.pdf"}}),
        ),
        // Hosted tools: a streamed call and the results that used to fail
        // the whole reply.
        Sample {
            block: json!({"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_fetch",
                          "input": {"url": "https://example.invalid"}, "caller": {"type": "direct"}}),
            start: json!({"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_fetch",
                          "input": {}, "caller": {"type": "direct"}}),
            deltas: vec![
                json!({"type": "input_json_delta", "partial_json": "{\"url\":\"https://example.invalid\"}"}),
            ],
        },
        whole(
            json!({"type": "web_fetch_tool_result", "tool_use_id": "srvtoolu_1",
                     "content": {"type": "web_fetch_result", "url": "https://example.invalid",
                                 "content": {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "page"}}}}),
        ),
        whole(
            json!({"type": "bash_code_execution_tool_result", "tool_use_id": "srvtoolu_2",
                     "content": {"type": "bash_code_execution_result", "stdout": "42\n", "stderr": "", "return_code": 0, "content": []}}),
        ),
        whole(
            json!({"type": "mcp_tool_use", "id": "mcptoolu_1", "name": "search", "server_name": "docs", "input": {"q": "rig"}}),
        ),
        whole(json!({"type": "container_upload", "file_id": "file_1"})),
        // A block type invented after this crate was written, with a text
        // field its stream fills by delta.
        Sample {
            block: json!({"type": "novel_block_2027", "id": "nb_1", "payload": {"nested": [1, 2]}, "text": "abc"}),
            start: json!({"type": "novel_block_2027", "id": "nb_1", "payload": {"nested": [1, 2]}, "text": ""}),
            deltas: vec![json!({"type": "text_delta", "text": "abc"})],
        },
    ]
}

/// `blocks` as one unary reply body.
fn unary_body(blocks: &[serde_json::Value]) -> String {
    serde_json::json!({
        "type": "message", "id": "msg_1", "model": "claude-haiku-4-5", "role": "assistant",
        "content": blocks, "stop_reason": "end_turn", "stop_sequence": null,
        "usage": {"input_tokens": 3, "output_tokens": 5}
    })
    .to_string()
}

/// `samples` as one SSE body, block by block.
fn streamed_body(samples: &[&Sample]) -> String {
    use serde_json::json;
    let mut events = vec![json!({"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-haiku-4-5",
        "content": [], "stop_reason": null, "stop_sequence": null,
        "usage": {"input_tokens": 3, "output_tokens": 1}}})];
    for (index, sample) in samples.iter().enumerate() {
        events.push(
            json!({"type": "content_block_start", "index": index, "content_block": sample.start}),
        );
        for delta in &sample.deltas {
            events.push(json!({"type": "content_block_delta", "index": index, "delta": delta}));
        }
        events.push(json!({"type": "content_block_stop", "index": index}));
    }
    events.push(
        json!({"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": null},
                       "usage": {"output_tokens": 5}}),
    );
    events.push(json!({"type": "message_stop"}));
    events
        .iter()
        .map(|event| {
            format!(
                "event: {}\ndata: {event}\n\n",
                event["type"].as_str().unwrap_or_default()
            )
        })
        .collect()
}

/// `turn` replayed on `wire` as history, and the assistant message it sent.
fn replayed(wire: &Messages, turn: Vec<AssistantContent>) -> Option<serde_json::Value> {
    let request = CompletionRequest::new(Message::user("next"))
        .messages([
            Message::user("first"),
            Message::Assistant {
                id: None,
                content: turn,
            },
        ])
        .max_tokens(32);
    let body = json_body(
        &wire
            .encode(request, Mode::Unary)
            .expect("the follow-up encodes")
            .request,
    );
    body["messages"]
        .as_array()
        .and_then(|messages| {
            messages
                .iter()
                .find(|message| message["role"] == "assistant")
        })
        .cloned()
}

use super::super::completion::Content;
use crate::completion::Message;

#[test]
fn every_block_survives_decode_history_and_replay_in_both_modes() {
    let samples = samples();
    let mut covered = [false; CONTENT_VARIANTS];
    for sample in &samples {
        let typed: Content =
            serde_json::from_value(sample.block.clone()).expect("every block decodes");
        let index = variant_index(&typed);
        assert!(
            index < CONTENT_VARIANTS,
            "number the new variant within CONTENT_VARIANTS"
        );
        covered[index] = true;

        let buffered = fold(
            &unary_body(std::slice::from_ref(&sample.block)),
            Mode::Unary,
        );
        let streamed = fold(&streamed_body(&[sample]), Mode::Streaming);
        assert_eq!(
            buffered.choice, streamed.choice,
            "{}: both modes agree",
            sample.block["type"]
        );
        assert_eq!(
            buffered.choice.len(),
            1,
            "{}: one part, never dropped",
            sample.block["type"]
        );

        let assistant = replayed(&wire(), buffered.choice).expect("the turn replays");
        assert_eq!(
            assistant["content"],
            serde_json::json!([sample.block]),
            "replayed verbatim"
        );
    }
    assert!(
        covered.iter().all(|seen| *seen),
        "a sample for every variant: {covered:?}"
    );
}

#[test]
fn a_reply_mixing_every_block_keeps_their_order() {
    let samples = samples();
    let blocks: Vec<_> = samples.iter().map(|sample| sample.block.clone()).collect();
    let buffered = fold(&unary_body(&blocks), Mode::Unary);
    let streamed = fold(
        &streamed_body(&samples.iter().collect::<Vec<_>>()),
        Mode::Streaming,
    );
    assert_eq!(buffered.choice, streamed.choice);
    let assistant = replayed(&wire(), buffered.choice).expect("the turn replays");
    assert_eq!(assistant["content"], serde_json::Value::Array(blocks));
}

#[test]
fn native_items_never_reach_another_issuer_or_wire_format() {
    let hosted = samples()
        .into_iter()
        .find(|sample| sample.block["type"] == "web_fetch_tool_result")
        .expect("a hosted-tool sample");
    let turn = fold(
        &unary_body(&[
            hosted.block.clone(),
            serde_json::json!({"type": "text", "text": "answer"}),
        ]),
        Mode::Unary,
    )
    .choice;
    assert!(matches!(turn[0], AssistantContent::Native(_)));

    // The same Messages format under another issuer: the item is left out.
    let zai = AnthropicConfig::with_key(&ZAI, "k").completion("glm-4.6");
    let assistant = replayed(&zai, turn.clone()).expect("the text still replays");
    assert_eq!(
        assistant["content"],
        serde_json::json!([{"type": "text", "text": "answer"}])
    );

    // A native-only turn has nothing for that issuer: the message is gone.
    let native_only = vec![turn[0].clone()];
    assert_eq!(replayed(&zai, native_only.clone()), None);

    // Another wire format with the same issuer name never opens it either.
    let responses = crate::providers::openai::OpenAIConfig::new("k").responses("gpt-5.2");
    let request = CompletionRequest::new(Message::user("next")).messages([
        Message::user("first"),
        Message::Assistant {
            id: None,
            content: turn,
        },
    ]);
    let body = json_body(
        &responses
            .encode(request, Mode::Unary)
            .expect("encodes")
            .request,
    );
    assert!(
        !body.to_string().contains("web_fetch_tool_result"),
        "a Messages item never reaches a Responses request: {body}"
    );
}
