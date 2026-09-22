//! The Messages wire, driven from recorded bytes and no socket.
//!
//! The unary and streamed bodies below are the two cells of
//! `crates/rig-cassette/fixtures/cassettes/anthropic/raw_completion_parity_matrix/` — the same turn
//! recorded both ways. Folding them through the same decoder is the property
//! this port exists for, and it is checked here without a transport so a
//! failure names the decoder rather than the harness.

use super::*;
use crate::driver::WireDriver;
use crate::message::AssistantContent;
use crate::streaming::StreamEvent;
use crate::wire::secret::tests::a_config_reloads_without_its_credential;
use crate::wire::{Fold, Framing, Mode, Wire};

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
    Anthropic::new("sk-test").messages("claude-haiku-4-5")
}

fn request() -> CompletionRequest {
    CompletionRequest {
        model: None,
        chat_history: vec![crate::message::Message::user(
            "Reply with exactly: parity probe",
        )],
        documents: Vec::new(),
        tools: Vec::new(),
        temperature: None,
        max_tokens: Some(32),
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    }
}

/// Fold a recorded reply body through the wire's own decoder, read in the
/// mode the driver would have read it in — which is also what decides the
/// framing, exactly as `encode` decides it.
fn fold(body: &str, mode: Mode) -> crate::completion::CompletionResponse {
    let wire = wire();
    let mut driver = WireDriver::<Completion, _>::new(wire.decoder(mode));
    match mode {
        Mode::Streaming => {
            let mut framer = crate::http_client::framing::SseFramer::new();
            for event in framer.push(body.as_bytes()) {
                if !event.data.trim().is_empty() {
                    driver.push(WireFrame::Text(event.data));
                }
            }
        }
        Mode::Unary => driver.push(WireFrame::Text(body.to_owned())),
    }
    driver.finish();
    let mut fold = <Completion as crate::wire::Operation>::fold(&request());
    for item in driver.drain() {
        let event = item.expect("the recorded reply decodes without an in-band error");
        fold.absorb(event).expect("the fold accepts every event");
    }
    Fold::<Completion>::finish(
        fold,
        crate::wire::Reply {
            provider: "anthropic".to_owned(),
            raw: serde_json::from_str(body).unwrap_or(serde_json::Value::Null),
            provider_request_id: Some("req_REDACTED_1".to_owned()),
        },
    )
    .expect("the fold produces a response")
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
fn the_buffered_reply_is_one_frame_whose_terminal_is_unconditional() {
    let wire = wire();
    let mut driver = WireDriver::<Completion, _>::new(wire.decoder(Mode::Unary));
    driver.push(WireFrame::Text(UNARY.to_owned()));
    let items: Vec<_> = driver.drain().collect();
    assert!(
        items
            .iter()
            .any(|item| matches!(item, Ok(StreamEvent::Final(_)))),
        "a whole message is a complete turn, so it always ends in a terminal"
    );
    assert!(driver.done(), "the terminal stops the driver");
}

#[test]
fn the_streaming_request_asks_for_a_stream_and_the_unary_one_does_not() {
    let unary = wire()
        .encode(request(), Mode::Unary)
        .expect("the request encodes");
    assert_eq!(unary.framing, Framing::Whole);
    let unary_body = body_of(&unary);
    assert_eq!(unary_body.get("stream"), None);

    let streaming = wire()
        .encode(request(), Mode::Streaming)
        .expect("the request encodes");
    assert_eq!(streaming.framing, Framing::Sse);
    assert_eq!(
        body_of(&streaming).get("stream"),
        Some(&serde_json::Value::Bool(true))
    );
}

#[test]
fn the_request_carries_the_key_version_and_endpoint() {
    let encoded = wire()
        .encode(request(), Mode::Unary)
        .expect("the request encodes");
    let request = encoded.requests.first().expect("one request");
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

fn body_of(encoded: &Encoded) -> serde_json::Value {
    let request = encoded.requests.first().expect("one request");
    match request.body() {
        Body::Bytes(bytes) => {
            serde_json::from_slice(bytes).expect("the body is the JSON the wire built")
        }
        Body::Multipart(_) => panic!("the Messages endpoint takes JSON"),
    }
}

#[test]
fn a_serialized_provider_never_carries_its_key() {
    let provider = Anthropic::new("sk-live-do-not-leak").with_beta("prompt-caching-2024-07-31");
    a_config_reloads_without_its_credential(&provider, "sk-live-do-not-leak", |provider| {
        &provider.api_key
    });

    let wire = provider.messages("claude-haiku-4-5");
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
