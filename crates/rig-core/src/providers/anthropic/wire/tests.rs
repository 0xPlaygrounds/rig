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

/// A conversation whose assistant turn carries `reasoning`, issued by
/// `issuer`, followed by a new user turn: the shape a replayed tool loop has.
fn replaying(issuer: &'static str, reasoning: crate::message::Reasoning) -> CompletionRequest {
    request().messages([
        crate::message::Message::user("first"),
        crate::message::Message::Assistant {
            id: None,
            content: vec![
                AssistantContent::Reasoning(reasoning.sealed(issuer)),
                AssistantContent::text("answer"),
            ],
        },
    ])
}

/// A request replaying one signed Anthropic thinking block.
fn replaying_thinking() -> CompletionRequest {
    replaying(
        "anthropic",
        crate::message::Reasoning::new_with_signature("thought", Some("sig".into())),
    )
}

fn binding_wire(model: &str) -> Messages {
    AnthropicConfig::new("sk-test").completion(model)
}

fn beta_header(encoded: &Encoded) -> Option<&str> {
    encoded
        .request
        .headers()
        .get("anthropic-beta")
        .and_then(|value| value.to_str().ok())
}

const DROP_BLOCK: &str = r#"{"block_binding":{"prefix_mismatch_behavior":"drop_block"}}"#;

#[test]
fn a_binding_model_replaying_thinking_drops_stale_blocks_in_both_modes() {
    use super::super::completion::{CLAUDE_FABLE_5, CLAUDE_FABLE_5_1, CLAUDE_OPUS_5_5};
    let drop_block: serde_json::Value = serde_json::from_str(DROP_BLOCK).expect("JSON");
    for model in [
        CLAUDE_OPUS_5_5,
        CLAUDE_FABLE_5_1,
        CLAUDE_FABLE_5,
        "claude-opus-5-5-20260901",
    ] {
        for mode in [Mode::Unary, Mode::Streaming] {
            let encoded = binding_wire(model)
                .encode(replaying_thinking(), mode)
                .expect("the request encodes");
            assert_eq!(
                json_body(&encoded.request)["thinking"],
                drop_block,
                "{model}"
            );
            assert_eq!(
                beta_header(&encoded),
                Some(THINKING_BINDING_BETA),
                "{model}"
            );
        }
    }
}

#[test]
fn a_replayed_redacted_thinking_block_is_bound_too() {
    let encoded = binding_wire(super::super::completion::CLAUDE_OPUS_5_5)
        .encode(
            replaying("anthropic", crate::message::Reasoning::redacted("opaque")),
            Mode::Unary,
        )
        .expect("the request encodes");
    let body = json_body(&encoded.request);
    assert_eq!(
        body["messages"][1]["content"][0]["type"],
        "redacted_thinking"
    );
    assert_eq!(
        body["thinking"]["block_binding"]["prefix_mismatch_behavior"],
        "drop_block"
    );
    assert_eq!(beta_header(&encoded), Some(THINKING_BINDING_BETA));
}

#[test]
fn a_request_without_replayed_thinking_is_unchanged() {
    let opus = binding_wire(super::super::completion::CLAUDE_OPUS_5_5);
    // Nothing to replay.
    let encoded = opus
        .encode(request(), Mode::Unary)
        .expect("the request encodes");
    assert_eq!(json_body(&encoded.request).get("thinking"), None);
    assert_eq!(beta_header(&encoded), None);

    // Reasoning another provider issued is not replayed, so nothing is bound.
    let encoded = opus
        .encode(
            replaying(
                "openai",
                crate::message::Reasoning::new_with_signature("thought", Some("sig".into())),
            ),
            Mode::Unary,
        )
        .expect("the request encodes");
    let body = json_body(&encoded.request);
    assert_eq!(body.get("thinking"), None);
    assert_eq!(beta_header(&encoded), None);
}

#[test]
fn a_model_that_does_not_bind_thinking_blocks_is_unchanged() {
    use super::super::completion::{
        CLAUDE_HAIKU_4_5, CLAUDE_OPUS_4_8, CLAUDE_OPUS_5, CLAUDE_SONNET_4_6, CLAUDE_SONNET_5,
        CLAUDE_SONNET_5_5,
    };
    for model in [
        CLAUDE_SONNET_5_5,
        CLAUDE_OPUS_5,
        CLAUDE_SONNET_5,
        CLAUDE_OPUS_4_8,
        CLAUDE_SONNET_4_6,
        CLAUDE_HAIKU_4_5,
        // A later model whose id merely starts with a binding model's.
        "claude-opus-5-5-mini",
    ] {
        for behavior in [
            ThinkingPrefixMismatch::DropBlock,
            ThinkingPrefixMismatch::Reject,
        ] {
            let encoded = AnthropicConfig::new("sk-test")
                .with_thinking_prefix_mismatch(behavior)
                .completion(model)
                .encode(replaying_thinking(), Mode::Streaming)
                .expect("the request encodes");
            assert_eq!(json_body(&encoded.request).get("thinking"), None, "{model}");
            assert_eq!(beta_header(&encoded), None, "{model}");
        }
    }
}

#[test]
fn reject_sends_the_request_without_a_binding() {
    let config = AnthropicConfig::new("sk-test")
        .with_thinking_prefix_mismatch(ThinkingPrefixMismatch::Reject);
    let model = super::super::completion::CLAUDE_OPUS_5_5;
    let rejecting = config.completion(model);
    let encoded = rejecting
        .encode(replaying_thinking(), Mode::Unary)
        .expect("the request encodes");
    assert_eq!(json_body(&encoded.request).get("thinking"), None);
    assert_eq!(beta_header(&encoded), None);

    // The default is `DropBlock`, and choosing it is the same request.
    let explicit = config
        .with_thinking_prefix_mismatch(ThinkingPrefixMismatch::DropBlock)
        .completion(model)
        .encode(replaying_thinking(), Mode::Unary)
        .expect("the request encodes");
    let default = binding_wire(model)
        .encode(replaying_thinking(), Mode::Unary)
        .expect("the request encodes");
    assert_eq!(json_body(&explicit.request), json_body(&default.request));
    assert_eq!(beta_header(&explicit), Some(THINKING_BINDING_BETA));
}

#[test]
fn the_callers_thinking_settings_are_kept() {
    let wire = binding_wire(super::super::completion::CLAUDE_OPUS_5_5);

    // The binding merges into the caller's `thinking`.
    let encoded = wire
        .encode(
            replaying_thinking()
                .additional_params(serde_json::json!({ "thinking": { "type": "adaptive" } })),
            Mode::Unary,
        )
        .expect("the request encodes");
    let body = json_body(&encoded.request);
    assert_eq!(body["thinking"]["type"], "adaptive");
    assert_eq!(
        body["thinking"]["block_binding"]["prefix_mismatch_behavior"],
        "drop_block"
    );

    // The caller's own `block_binding` wins, and still gets its beta flag.
    let encoded = wire
        .encode(
            replaying_thinking().additional_params(serde_json::json!({
                "thinking": { "block_binding": { "prefix_mismatch_behavior": "error" } }
            })),
            Mode::Streaming,
        )
        .expect("the request encodes");
    assert_eq!(
        json_body(&encoded.request)["thinking"],
        serde_json::json!({ "block_binding": { "prefix_mismatch_behavior": "error" } })
    );
    assert_eq!(beta_header(&encoded), Some(THINKING_BINDING_BETA));

    // Disabled thinking rejects the field, so it is left alone.
    let encoded = wire
        .encode(
            replaying_thinking()
                .additional_params(serde_json::json!({ "thinking": { "type": "disabled" } })),
            Mode::Unary,
        )
        .expect("the request encodes");
    assert_eq!(
        json_body(&encoded.request)["thinking"],
        serde_json::json!({ "type": "disabled" })
    );
    assert_eq!(beta_header(&encoded), None);
}

#[test]
fn the_binding_beta_joins_the_callers_flags_once() {
    let model = super::super::completion::CLAUDE_OPUS_5_5;
    let encoded = AnthropicConfig::new("sk-test")
        .with_beta("other-beta")
        .completion(model)
        .encode(replaying_thinking(), Mode::Unary)
        .expect("the request encodes");
    assert_eq!(
        beta_header(&encoded),
        Some("other-beta,thinking-binding-controls-2026-08-01")
    );

    for betas in [
        vec![THINKING_BINDING_BETA],
        vec!["other-beta,thinking-binding-controls-2026-08-01"],
    ] {
        let config = betas
            .iter()
            .fold(AnthropicConfig::new("sk-test"), |config, beta| {
                config.with_beta(*beta)
            });
        let encoded = config
            .completion(model)
            .encode(replaying_thinking(), Mode::Unary)
            .expect("the request encodes");
        assert_eq!(beta_header(&encoded), Some(betas.join(",").as_str()));
    }
}

#[test]
fn a_gateway_sends_no_binding_unless_its_quirks_opt_in() {
    let gateway = AnthropicConfig::with_key(&MINIMAX, "sk-test")
        .completion(super::super::completion::CLAUDE_OPUS_5_5);
    let replaying_minimax = || {
        replaying(
            MINIMAX.name,
            crate::message::Reasoning::new_with_signature("thought", Some("sig".into())),
        )
    };
    let encoded = gateway
        .encode(replaying_minimax(), Mode::Unary)
        .expect("the request encodes");
    let body = json_body(&encoded.request);
    assert_eq!(body["messages"][1]["content"][0]["type"], "thinking");
    assert_eq!(body.get("thinking"), None);
    assert_eq!(beta_header(&encoded), None);
    const _: () = assert!(!MINIMAX.quirks.thinking_block_binding);
    const _: () = assert!(ANTHROPIC.quirks.thinking_block_binding);

    let mut quirks = Quirks::gateway();
    quirks.thinking_block_binding = true;
    let opted_in = Dialect { quirks, ..MINIMAX };
    let encoded = AnthropicConfig::with_key(&opted_in, "sk-test")
        .completion(super::super::completion::CLAUDE_OPUS_5_5)
        .encode(replaying_minimax(), Mode::Unary)
        .expect("the request encodes");
    assert_eq!(
        json_body(&encoded.request)["thinking"]["block_binding"]["prefix_mismatch_behavior"],
        "drop_block"
    );
    assert_eq!(beta_header(&encoded), Some(THINKING_BINDING_BETA));
}

#[test]
fn the_default_binding_is_left_out_of_a_serialized_config() {
    let config = AnthropicConfig::new("sk-test");
    let json = serde_json::to_value(&config).expect("the config serializes");
    assert_eq!(json.get("thinking_prefix_mismatch"), None);

    let rejecting = config.with_thinking_prefix_mismatch(ThinkingPrefixMismatch::Reject);
    let json = serde_json::to_value(&rejecting).expect("the config serializes");
    assert_eq!(json["thinking_prefix_mismatch"], "reject");
    let restored: AnthropicConfig = serde_json::from_value(json).expect("the config loads");
    assert_eq!(
        restored.thinking_prefix_mismatch,
        ThinkingPrefixMismatch::Reject
    );
}
