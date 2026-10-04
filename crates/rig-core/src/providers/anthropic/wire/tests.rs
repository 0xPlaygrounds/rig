//! The Messages wire, driven from recorded bytes and no socket.
//!
//! The unary and streamed bodies below are one recorded Messages turn, taken
//! both ways. Folding them through the same decoder is the property
//! this port exists for, and it is checked here without a transport so a
//! failure names the decoder rather than the harness.

use super::*;
use crate::message::AssistantContent;
use crate::test_utils::json_body;
use crate::wire::secret::tests::a_config_reloads_without_its_credential;
use crate::wire::{Framing, Mode, Wire, WireFrame};

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
    assert_eq!(buffered.model(), streamed.model());
    assert_eq!(buffered.finish_reason(), streamed.finish_reason());
    assert_eq!(buffered.response_id(), streamed.response_id());
    assert_eq!(buffered.provider_request_id, streamed.provider_request_id);
    assert_eq!(
        buffered.choice.first().map(AssistantContent::canonical),
        Some(AssistantContent::text("parity probe"))
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

/// pi's rule for a call id another model issued: characters outside
/// `[a-zA-Z0-9_-]` become `_`, and the id keeps at most 64 of them.
#[test]
fn a_foreign_call_id_is_normalized_to_anthropic_spelling() {
    use crate::completion::ReplayTarget;

    let wire = wire();
    assert_eq!(
        wire.normalize_tool_call_id("call_1|fc_2.x", wire.model(), None),
        "call_1_fc_2_x"
    );
    let long = "a".repeat(80);
    assert_eq!(
        wire.normalize_tool_call_id(&long, wire.model(), None).len(),
        64
    );
    assert_eq!(
        wire.normalize_tool_call_id("toolu_01-Ab", wire.model(), None),
        "toolu_01-Ab"
    );
}

/// Each dialect reads images on its documented vision models only, in user
/// turns and tool results; no Messages model reads assistant images.
#[test]
fn each_dialect_reads_images_on_its_vision_models() {
    use crate::completion::ReplayTarget;

    for (dialect, model, images) in [
        (&ANTHROPIC, "claude-sonnet-4-6", true),
        (&ZAI, "glm-4.6", false),
        (&ZAI, "glm-4.5-air", false),
        (&ZAI, "glm-4.5v", true),
        (&ZAI, "glm-4.6v-flash", true),
        (&ZAI, "glm-5v-turbo", true),
        (&MOONSHOT, "kimi-k2-thinking", false),
        (&MOONSHOT, "kimi-k2-0905-preview", false),
        (&MOONSHOT, "moonshot-v1-8k", false),
        (&MOONSHOT, "moonshot-v1-8k-vision-preview", true),
        (&MOONSHOT, "kimi-k2.6", true),
        (&MOONSHOT, "kimi-k3", true),
        (&MINIMAX, "MiniMax-M2.7", false),
        (&MINIMAX, "MiniMax-M3", true),
        (&XIAOMIMIMO, "mimo-v2-flash", false),
        (&XIAOMIMIMO, "mimo-v2-pro", false),
        (&XIAOMIMIMO, "mimo-v2.5-pro", false),
        (&XIAOMIMIMO, "mimo-v2-omni", true),
        (&XIAOMIMIMO, "mimo-v2.5", true),
        (&XIAOMIMIMO, "mimo-v2.6-flash", true),
    ] {
        // The model the request addresses decides, not the wire's own.
        let accepts = AnthropicConfig::with_key(dialect, "sk-test")
            .completion("another-model")
            .accepts(model);
        assert_eq!(accepts.user_images, images, "{model}");
        assert_eq!(accepts.tool_result_images, images, "{model}");
        assert!(!accepts.assistant_images, "{model}");
        assert!(accepts.tools, "{model}");
    }
}
