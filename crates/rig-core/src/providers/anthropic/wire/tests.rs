//! The Messages wire, driven from recorded bytes and no socket.
//!
//! The unary and streamed bodies below are the two cells of
//! `crates/rig-cassette/fixtures/cassettes/anthropic/raw_completion_parity_matrix/` — the same turn
//! recorded both ways. Folding them through the same decoder is the property
//! this port exists for, and it is checked here without a transport so a
//! failure names the decoder rather than the harness.

use super::*;
use crate::driver::WireDriver;
use crate::error::ProviderError;
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
    AnthropicConfig::new("sk-test").completion("claude-haiku-4-5")
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
    let mut fold = crate::test_utils::fold_for(&request(), &wire, mode);
    for item in driver.drain() {
        let event = item.expect("the recorded reply decodes without an in-band error");
        fold.absorb(&event).expect("the fold accepts every event");
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

// Encoded-request cells for the model-specific request rules below. They pin
// what Rig refuses to send or how it merges settings before any traffic, so
// no recorded reply can witness them.

fn encoded_for(wire: &Messages, request: CompletionRequest) -> Result<Encoded, EncodeError> {
    wire.encode(request, Mode::Unary)
}

fn beta_header(encoded: &Encoded) -> Option<String> {
    encoded
        .requests
        .first()
        .and_then(|request| request.headers().get("anthropic-beta"))
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
}

fn request_with_tool_choice(choice: crate::message::ToolChoice) -> CompletionRequest {
    let mut request = request();
    request.tools = vec![crate::completion::ToolDefinition {
        name: "lookup".to_owned(),
        description: "Look something up.".to_owned(),
        parameters: serde_json::json!({"type": "object", "properties": {}}),
    }];
    request.tool_choice = Some(choice);
    request
}

#[test]
fn forced_tool_choice_fails_to_encode_for_models_that_reject_it() {
    use crate::message::ToolChoice;
    use crate::providers::anthropic::completion::{CLAUDE_FABLE_5_1, CLAUDE_OPUS_5_5};

    for model in [CLAUDE_OPUS_5_5, CLAUDE_FABLE_5_1, "claude-mythos-5-1"] {
        let wire = AnthropicConfig::new("sk-test").completion(model);
        for choice in [
            ToolChoice::Required,
            ToolChoice::Specific {
                function_names: vec!["lookup".to_owned()],
            },
        ] {
            for mode in [Mode::Unary, Mode::Streaming] {
                let error = wire
                    .encode(request_with_tool_choice(choice.clone()), mode)
                    .expect_err("a forced tool choice must not reach the API");
                let message = ProviderError::from(error).to_string();
                assert!(message.contains(model), "{message}");
                assert!(message.contains("ToolChoice::Auto"), "{message}");
            }
        }
        for choice in [ToolChoice::Auto, ToolChoice::None] {
            let encoded = encoded_for(&wire, request_with_tool_choice(choice))
                .expect("auto and none are accepted");
            let body = body_of(&encoded);
            assert!(matches!(
                body["tool_choice"]["type"].as_str(),
                Some("auto" | "none")
            ));
        }
    }
}

#[test]
fn forced_tool_choice_still_encodes_for_models_that_accept_it() {
    use crate::providers::anthropic::completion::{CLAUDE_FABLE_5, CLAUDE_OPUS_5};

    // `claude-opus-5` and `claude-fable-5` prefix the rejecting IDs; the list
    // is exact, so they keep forced tool use.
    for model in [CLAUDE_OPUS_5, CLAUDE_FABLE_5, "claude-opus-5-5-preview"] {
        let wire = AnthropicConfig::new("sk-test").completion(model);
        let encoded = encoded_for(
            &wire,
            request_with_tool_choice(crate::message::ToolChoice::Required),
        )
        .expect("forced tool use is accepted");
        assert_eq!(body_of(&encoded)["tool_choice"]["type"], "any");
    }
}

#[test]
fn describe_reports_the_forced_tool_choice_capability_per_model() {
    use crate::providers::anthropic::completion::{
        CLAUDE_FABLE_5, CLAUDE_FABLE_5_1, CLAUDE_OPUS_5, CLAUDE_OPUS_5_5,
    };

    let accepts = |model: &str| {
        AnthropicConfig::new("sk-test")
            .completion(model)
            .describe()
            .capabilities
            .completion
            .accepts_forced_tool_choice
    };
    assert!(!accepts(CLAUDE_OPUS_5_5));
    assert!(!accepts(CLAUDE_FABLE_5_1));
    assert!(accepts(CLAUDE_OPUS_5));
    assert!(accepts(CLAUDE_FABLE_5));
}

fn schema_request(additional_params: Option<serde_json::Value>) -> CompletionRequest {
    let mut request = request();
    request.output_schema = Some(crate::schemars::schema_for!(String));
    request.additional_params = additional_params;
    request
}

fn encoded_body_bytes(encoded: &Encoded) -> String {
    match encoded.requests.first().map(http::Request::body) {
        Some(Body::Bytes(bytes)) => String::from_utf8(bytes.to_vec()).expect("utf-8 JSON"),
        _ => panic!("the Messages endpoint takes JSON bytes"),
    }
}

#[test]
fn output_config_merges_effort_additional_params_and_the_schema_format_into_one_key() {
    use crate::providers::anthropic::completion::{CLAUDE_OPUS_5_5, Effort};

    let wire = AnthropicConfig::new("sk-test")
        .completion(CLAUDE_OPUS_5_5)
        .with_effort(Effort::High);

    // Typed effort and the schema's format share one object.
    let encoded = encoded_for(&wire, schema_request(None)).expect("encodes");
    let body = body_of(&encoded);
    assert_eq!(body["output_config"]["effort"], "high");
    assert_eq!(body["output_config"]["format"]["type"], "json_schema");
    assert_eq!(
        encoded_body_bytes(&encoded)
            .matches("\"output_config\"")
            .count(),
        1
    );

    // A request's effort overrides the wire default, and unknown keys pass through.
    let encoded = encoded_for(
        &wire,
        schema_request(Some(serde_json::json!({
            "output_config": {"effort": "low", "task_budget": {"tokens": 1000}}
        }))),
    )
    .expect("encodes");
    let body = body_of(&encoded);
    assert_eq!(body["output_config"]["effort"], "low");
    assert_eq!(body["output_config"]["task_budget"]["tokens"], 1000);
    assert_eq!(body["output_config"]["format"]["type"], "json_schema");
    assert_eq!(
        encoded_body_bytes(&encoded)
            .matches("\"output_config\"")
            .count(),
        1
    );

    // Without typed settings, `additional_params` alone still lands once.
    let plain = AnthropicConfig::new("sk-test").completion(CLAUDE_OPUS_5_5);
    let mut effort_only = request();
    effort_only.additional_params = Some(serde_json::json!({"output_config": {"effort": "max"}}));
    let body = body_of(&encoded_for(&plain, effort_only).expect("encodes"));
    assert_eq!(body["output_config"], serde_json::json!({"effort": "max"}));

    // No settings at all sends no `output_config`.
    let body = body_of(&encoded_for(&plain, request()).expect("encodes"));
    assert_eq!(body.get("output_config"), None);
}

#[test]
fn output_config_rejects_a_format_that_conflicts_with_the_output_schema() {
    use crate::providers::anthropic::completion::CLAUDE_OPUS_5_5;

    let wire = AnthropicConfig::new("sk-test").completion(CLAUDE_OPUS_5_5);
    let conflicting = schema_request(Some(serde_json::json!({
        "output_config": {"format": {"type": "json_schema", "schema": {"type": "object"}}}
    })));
    let error = encoded_for(&wire, conflicting).expect_err("two formats cannot both win");
    assert!(
        ProviderError::from(error)
            .to_string()
            .contains("output_schema")
    );

    // The same format from both sources is not a conflict.
    let rig_format = body_of(&encoded_for(&wire, schema_request(None)).expect("encodes"))
        ["output_config"]["format"]
        .clone();
    let agreeing = schema_request(Some(
        serde_json::json!({"output_config": {"format": rig_format.clone()}}),
    ));
    let body = body_of(&encoded_for(&wire, agreeing).expect("encodes"));
    assert_eq!(body["output_config"]["format"], rig_format);

    // A format without an output schema passes through.
    let mut format_only = request();
    format_only.additional_params = Some(serde_json::json!({
        "output_config": {"format": {"type": "json_schema", "schema": {"type": "object"}}}
    }));
    let body = body_of(&encoded_for(&wire, format_only).expect("encodes"));
    assert_eq!(body["output_config"]["format"]["schema"]["type"], "object");
}

#[test]
fn invalid_effort_or_thinking_in_additional_params_fails_to_encode() {
    use crate::providers::anthropic::completion::CLAUDE_OPUS_5_5;

    let wire = AnthropicConfig::new("sk-test").completion(CLAUDE_OPUS_5_5);
    for params in [
        serde_json::json!({"output_config": {"effort": "adaptive"}}),
        serde_json::json!({"output_config": "high"}),
        serde_json::json!({"thinking": {"type": "sometimes"}}),
        serde_json::json!({"thinking": {"display": "updates"}}),
    ] {
        let mut request = request();
        request.additional_params = Some(params.clone());
        assert!(
            encoded_for(&wire, request).is_err(),
            "{params} should not reach the API"
        );
    }
}

#[test]
fn thinking_merges_typed_defaults_with_additional_params() {
    use crate::providers::anthropic::completion::{
        CLAUDE_OPUS_5_5, THINKING_DISPLAY_UPDATES_BETA, Thinking, ThinkingDisplay,
    };

    let wire = AnthropicConfig::new("sk-test")
        .completion(CLAUDE_OPUS_5_5)
        .with_thinking(Thinking::adaptive().with_display(ThinkingDisplay::Summarized));

    let encoded = encoded_for(&wire, request()).expect("encodes");
    assert_eq!(
        body_of(&encoded)["thinking"],
        serde_json::json!({"type": "adaptive", "display": "summarized"})
    );
    assert_eq!(beta_header(&encoded), None);

    // Request keys override the default's; unknown keys pass through.
    let mut overriding = request();
    overriding.additional_params = Some(serde_json::json!({
        "thinking": {
            "display": "updates",
            "block_binding": {"prefix_mismatch_behavior": "error"}
        }
    }));
    let encoded = encoded_for(&wire, overriding).expect("encodes");
    assert_eq!(
        body_of(&encoded)["thinking"],
        serde_json::json!({
            "type": "adaptive",
            "display": "updates",
            "block_binding": {"prefix_mismatch_behavior": "error"}
        })
    );
    assert_eq!(
        beta_header(&encoded).as_deref(),
        Some(THINKING_DISPLAY_UPDATES_BETA)
    );
    assert_eq!(
        encoded_body_bytes(&encoded).matches("\"thinking\"").count(),
        1
    );

    // A different type replaces the default whole: `disabled` has no display.
    let mut disabling = request();
    disabling.additional_params = Some(serde_json::json!({"thinking": {"type": "disabled"}}));
    let body = body_of(&encoded_for(&wire, disabling).expect("encodes"));
    assert_eq!(body["thinking"], serde_json::json!({"type": "disabled"}));
}

#[test]
fn updates_display_adds_its_beta_flag_once_beside_configured_flags() {
    use crate::providers::anthropic::completion::{
        CLAUDE_OPUS_5_5, THINKING_DISPLAY_UPDATES_BETA, Thinking, ThinkingDisplay,
    };

    let updates = Thinking::adaptive().with_display(ThinkingDisplay::Updates);
    let configured = AnthropicConfig::new("sk-test").with_beta("files-api-2025-04-14");
    let encoded = encoded_for(
        &configured
            .completion(CLAUDE_OPUS_5_5)
            .with_thinking(updates.clone()),
        request(),
    )
    .expect("encodes");
    assert_eq!(
        beta_header(&encoded),
        Some(format!(
            "files-api-2025-04-14,{THINKING_DISPLAY_UPDATES_BETA}"
        ))
    );

    let already = AnthropicConfig::new("sk-test").with_beta(THINKING_DISPLAY_UPDATES_BETA);
    let encoded = encoded_for(
        &already.completion(CLAUDE_OPUS_5_5).with_thinking(updates),
        request(),
    )
    .expect("encodes");
    assert_eq!(
        beta_header(&encoded).as_deref(),
        Some(THINKING_DISPLAY_UPDATES_BETA)
    );

    // Streaming requests carry it too.
    let mut streamed = request();
    streamed.additional_params =
        Some(serde_json::json!({"thinking": {"type": "adaptive", "display": "updates"}}));
    let encoded = AnthropicConfig::new("sk-test")
        .completion(CLAUDE_OPUS_5_5)
        .encode(streamed, Mode::Streaming)
        .expect("encodes");
    assert_eq!(
        beta_header(&encoded).as_deref(),
        Some(THINKING_DISPLAY_UPDATES_BETA)
    );
}

#[test]
fn typed_thinking_and_effort_serialize_to_the_documented_wire_values() {
    use crate::providers::anthropic::completion::{Effort, Thinking, ThinkingDisplay};

    assert_eq!(
        serde_json::to_value([
            Effort::Low,
            Effort::Medium,
            Effort::High,
            Effort::Xhigh,
            Effort::Max
        ])
        .expect("serializes"),
        serde_json::json!(["low", "medium", "high", "xhigh", "max"])
    );
    assert_eq!(
        serde_json::to_value(Thinking::enabled(2048).with_display(ThinkingDisplay::Omitted))
            .expect("serializes"),
        serde_json::json!({"type": "enabled", "budget_tokens": 2048, "display": "omitted"})
    );
    assert_eq!(
        serde_json::to_value(Thinking::Disabled.with_display(ThinkingDisplay::Summarized))
            .expect("serializes"),
        serde_json::json!({"type": "disabled"})
    );
    assert_eq!(
        serde_json::to_value(Thinking::adaptive()).expect("serializes"),
        serde_json::json!({"type": "adaptive"})
    );
}
