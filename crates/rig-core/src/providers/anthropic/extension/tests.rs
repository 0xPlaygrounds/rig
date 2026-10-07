//! Encoder tests build the body the Messages wire sends: a typed option is
//! definitory (the body key it writes), so these are unit tests. Extras over
//! recorded replies live in rig-cassette's Anthropic target; the ones here
//! cover fields no unary recording holds.

use serde_json::{Value, json};

use super::*;
use crate::completion::{CompletionResponse, Effort, Message, OnUnsupported};
use crate::error::ProviderError;
use crate::operation::Completion;
use crate::providers::anthropic::completion::{
    CLAUDE_HAIKU_4_5, CLAUDE_OPUS_4_8, CLAUDE_OPUS_5_5, CLAUDE_SONNET_4_6, CLAUDE_SONNET_5_5,
};
use crate::providers::anthropic::wire::{AnthropicConfig, Dialect, MINIMAX, Messages, ZAI};
use crate::wire::{Body, Mode, Operation, Wire, WireFrame};

fn wire(model: &str) -> Messages {
    AnthropicConfig::new("sk-test").completion(model)
}

/// The body `wire` sends for `request`, prepared as the driver prepares it.
fn sent_on(wire: &Messages, request: CompletionRequest) -> Result<Value, ProviderError> {
    let request = Completion::prepare(request, &wire.describe())?;
    let encoded = wire.encode(request, Mode::Unary)?;
    let Body::Bytes(bytes) = encoded.request.body() else {
        return Err(ProviderError::request("a Messages body is JSON"));
    };
    Ok(serde_json::from_slice(bytes)?)
}

/// `request` carrying `options` as the Anthropic entry.
fn with(request: CompletionRequest, options: &AnthropicOptions) -> CompletionRequest {
    request.provider_option(options.clone())
}

/// The body Anthropic's `model` gets for a short prompt with `options`.
fn sent(model: &str, options: AnthropicOptions) -> Value {
    sent_on(&wire(model), with(CompletionRequest::new("hi"), &options)).expect("the body encodes")
}

/// The refused option's name.
fn refused(result: Result<Value, ProviderError>) -> String {
    match result {
        Err(ProviderError::UnsupportedOption(option)) => option.option.into_owned(),
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn top_k_is_sent_at_the_top_level() {
    let body = sent(CLAUDE_HAIKU_4_5, AnthropicOptions::default().top_k(40));
    assert_eq!(body["top_k"], 40);
}

/// A model that fixes its sampling takes no `top_k`: refused by name under
/// `Error`, left out under `Ignore`.
#[test]
fn top_k_is_refused_on_a_model_that_fixes_its_sampling() {
    let options = AnthropicOptions::default().top_k(40);
    let request = with(CompletionRequest::new("hi"), &options);
    assert_eq!(
        refused(sent_on(&wire(CLAUDE_OPUS_4_8), request.clone())),
        "anthropic.*.top_k"
    );
    let ignored = request.on_unsupported(OnUnsupported::Ignore);
    let body = sent_on(&wire(CLAUDE_OPUS_4_8), ignored).expect("ignored, not refused");
    assert!(body.get("top_k").is_none(), "{body}");
}

#[test]
fn metadata_user_id_is_sent_under_metadata() {
    let body = sent(
        CLAUDE_SONNET_4_6,
        AnthropicOptions::default().metadata_user_id("u-1"),
    );
    assert_eq!(body["metadata"], json!({"user_id": "u-1"}));
}

#[test]
fn inference_geo_is_sent() {
    for (geo, word) in [(InferenceGeo::Us, "us"), (InferenceGeo::Global, "global")] {
        let body = sent(
            CLAUDE_SONNET_4_6,
            AnthropicOptions::default().inference_geo(geo),
        );
        assert_eq!(body["inference_geo"], word);
    }
}

/// Fast mode is sent on the models that offer it and refused on a listed
/// model that does not; the standard speed goes anywhere.
#[test]
fn speed_is_sent_where_the_model_offers_it() {
    let body = sent(
        CLAUDE_OPUS_5_5,
        AnthropicOptions::default().speed(Speed::Fast),
    );
    assert_eq!(body["speed"], "fast");
    let fast = with(
        CompletionRequest::new("hi"),
        &AnthropicOptions::default().speed(Speed::Fast),
    );
    assert_eq!(
        refused(sent_on(&wire(CLAUDE_SONNET_5_5), fast)),
        "anthropic.*.speed"
    );
    let body = sent(
        CLAUDE_SONNET_5_5,
        AnthropicOptions::default().speed(Speed::Standard),
    );
    assert_eq!(body["speed"], "standard");
}

/// The budget joins the mapped effort and the base's output format in one
/// `output_config`.
#[test]
fn task_budget_merges_into_output_config() {
    let schema: crate::schemars::Schema = serde_json::from_value(
        json!({"type": "object", "properties": {"answer": {"type": "string"}}}),
    )
    .expect("a schema");
    let request = CompletionRequest::new("hi")
        .reasoning(Effort::High)
        .output_schema(schema);
    let options = AnthropicOptions::default().task_budget(TaskBudget::new(20_000).remaining(5_000));
    let body = sent_on(&wire(CLAUDE_OPUS_4_8), with(request, &options)).expect("the body encodes");
    let config = &body["output_config"];
    assert_eq!(config["effort"], "high");
    assert!(config["format"].is_object(), "{config}");
    assert_eq!(
        config["task_budget"],
        json!({"type": "tokens", "total": 20000, "remaining": 5000})
    );
    let body = sent(
        CLAUDE_OPUS_4_8,
        AnthropicOptions::default().task_budget(TaskBudget::new(64_000)),
    );
    assert_eq!(
        body["output_config"],
        json!({"task_budget": {"type": "tokens", "total": 64000}})
    );
}

#[test]
fn fallbacks_are_sent_in_both_documented_forms() {
    let body = sent(
        CLAUDE_OPUS_5_5,
        AnthropicOptions::default().fallbacks(Fallbacks::Default),
    );
    assert_eq!(body["fallbacks"], "default");
    let body = sent(
        CLAUDE_OPUS_5_5,
        AnthropicOptions::default().fallbacks(Fallbacks::Models(vec![CLAUDE_OPUS_4_8.to_owned()])),
    );
    assert_eq!(body["fallbacks"], json!([{"model": CLAUDE_OPUS_4_8}]));
}

/// The history after a turn that ran in container `c-hist`.
fn container_history() -> Vec<Message> {
    let reply = json!({
        "type": "message", "id": "msg_1", "model": CLAUDE_SONNET_4_6, "role": "assistant",
        "stop_reason": "end_turn", "stop_sequence": null,
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "container": {"id": "c-hist", "expires_at": "2026-10-03T00:00:00Z"},
        "content": [{"type": "text", "text": "done"}]
    });
    let response = crate::test_utils::decode_reply(
        &wire(CLAUDE_SONNET_4_6),
        &CompletionRequest::new("run it"),
        Mode::Unary,
        [WireFrame::Text(reply.to_string())],
        reply.clone(),
    )
    .expect("the reply folds");
    let turn = response.message().expect("an assistant turn");
    vec![Message::user("run it"), turn, Message::user("again")]
}

/// A typed container replaces the one the history names; without it the
/// history's container is sent.
#[test]
fn a_typed_container_replaces_the_history_container() {
    let request = CompletionRequest::from(container_history());
    let body = sent_on(&wire(CLAUDE_SONNET_4_6), request.clone()).expect("the body encodes");
    assert_eq!(body["container"], "c-hist");

    let typed = with(
        request.clone(),
        &AnthropicOptions::default().container("c-typed"),
    );
    let body = sent_on(&wire(CLAUDE_SONNET_4_6), typed).expect("the body encodes");
    assert_eq!(body["container"], "c-typed");

    let skills = ContainerParam::Skills {
        id: None,
        skills: vec![
            Skill::new(SkillKind::Anthropic, "xlsx"),
            Skill::new(SkillKind::Custom, "skill_abc123").version("latest"),
        ],
    };
    let typed = with(request, &AnthropicOptions::default().container(skills));
    let body = sent_on(&wire(CLAUDE_SONNET_4_6), typed).expect("the body encodes");
    assert_eq!(
        body["container"],
        json!({"skills": [
            {"type": "anthropic", "skill_id": "xlsx"},
            {"type": "custom", "skill_id": "skill_abc123", "version": "latest"}
        ]})
    );
}

#[test]
fn context_management_is_sent() {
    let edits = ContextManagement::new(vec![json!({"type": "clear_tool_uses_20250919"})]);
    let body = sent(
        CLAUDE_OPUS_5_5,
        AnthropicOptions::default().context_management(edits),
    );
    assert_eq!(
        body["context_management"],
        json!({"edits": [{"type": "clear_tool_uses_20250919"}]})
    );
}

#[test]
fn mcp_servers_are_sent() {
    let body = sent(
        CLAUDE_OPUS_5_5,
        AnthropicOptions::default()
            .mcp_server(McpServer::new("https://mcp.example.com/sse", "example"))
            .mcp_server(
                McpServer::new("https://mcp.example.org/sse", "private").authorization_token("t"),
            ),
    );
    assert_eq!(
        body["mcp_servers"],
        json!([
            {"type": "url", "url": "https://mcp.example.com/sse", "name": "example"},
            {"type": "url", "url": "https://mcp.example.org/sse", "name": "private",
                "authorization_token": "t"}
        ])
    );
}

/// `Some(None)` sends `null`, which the API reads as a conversation's first
/// request.
#[test]
fn diagnostics_previous_message_id_sends_null_or_an_id() {
    let body = sent(
        CLAUDE_OPUS_5_5,
        AnthropicOptions::default().diagnostics_previous_message_id(None),
    );
    assert_eq!(body["diagnostics"], json!({"previous_message_id": null}));
    let body = sent(
        CLAUDE_OPUS_5_5,
        AnthropicOptions::default().diagnostics_previous_message_id(Some("msg_1".to_owned())),
    );
    assert_eq!(body["diagnostics"], json!({"previous_message_id": "msg_1"}));
}

/// `additional_params` sits above the typed layer, leaf by leaf.
#[test]
fn raw_additional_params_beat_typed_options() {
    let options = AnthropicOptions::default()
        .top_k(40)
        .metadata_user_id("typed");
    let request = with(CompletionRequest::new("hi"), &options)
        .additional_params(json!({"top_k": 5, "metadata": {"user_id": "raw"}}));
    let body = sent_on(&wire(CLAUDE_HAIKU_4_5), request).expect("the body encodes");
    assert_eq!(body["top_k"], 5);
    assert_eq!(body["metadata"], json!({"user_id": "raw"}));
}

/// Every option set at once writes no leaf a mapped option, the request or
/// the wire owns.
#[test]
fn no_field_writes_a_reserved_leaf() {
    let options = AnthropicOptions::default()
        .top_k(1)
        .metadata_user_id("u")
        .inference_geo(InferenceGeo::Us)
        .speed(Speed::Fast)
        .task_budget(TaskBudget::new(20_000).remaining(1))
        .fallbacks(Fallbacks::Default)
        .container("c")
        .context_management(ContextManagement::new(vec![json!({"type": "x"})]))
        .mcp_server(McpServer::new("u", "n"))
        .diagnostics_previous_message_id(None);
    let value = serde_json::to_value(&options).expect("options serialize");
    let shared = &value["*"];
    assert_eq!(
        value.as_object().map(|sections| sections.len()),
        Some(1),
        "{value}"
    );
    for pointer in [
        "/model",
        "/messages",
        "/max_tokens",
        "/system",
        "/temperature",
        "/top_p",
        "/stop_sequences",
        "/stream",
        "/tools",
        "/tool_choice",
        "/cache_control",
        "/service_tier",
        "/thinking",
        "/output_config/effort",
        "/output_config/format",
    ] {
        assert!(shared.pointer(pointer).is_none(), "{pointer} in {shared}");
    }
    assert_eq!(
        shared.as_object().map(|fields| fields.len()),
        Some(10),
        "every field is set: {shared}"
    );
}

/// A gateway speaking the Messages format reads its own entry, never
/// Anthropic's.
#[test]
fn an_anthropic_entry_is_not_read_by_a_dialect() {
    let options = AnthropicOptions::default()
        .top_k(40)
        .metadata_user_id("u-1");
    for (dialect, model) in [(&ZAI, "glm-4.6"), (&MINIMAX, "MiniMax-M2.7")] {
        let gateway =
            |dialect: &Dialect| AnthropicConfig::with_key(dialect, "sk-test").completion(model);
        let plain = sent_on(&gateway(dialect), CompletionRequest::new("hi")).expect("encodes");
        let typed = sent_on(
            &gateway(dialect),
            with(CompletionRequest::new("hi"), &options),
        )
        .expect("encodes");
        assert_eq!(plain, typed, "{}", dialect.name);
    }
}

/// The response `reply` folds into on `wire`, with `reply` as its raw
/// document.
fn folded(wire: &Messages, reply: &Value) -> CompletionResponse {
    crate::test_utils::decode_reply(
        wire,
        &CompletionRequest::new("hi"),
        Mode::Unary,
        [WireFrame::Text(reply.to_string())],
        reply.clone(),
    )
    .expect("the reply folds")
}

/// A Messages reply with `content` and `usage`, and `extra` merged in.
fn reply(content: Value, usage: Value, extra: Value) -> Value {
    let mut reply = json!({
        "type": "message", "id": "msg_1", "model": CLAUDE_OPUS_5_5, "role": "assistant",
        "content": content, "stop_reason": "end_turn", "stop_sequence": null,
        "usage": usage,
    });
    if let (Some(reply), Value::Object(extra)) = (reply.as_object_mut(), extra) {
        reply.extend(extra);
    }
    reply
}

/// Built here, not recorded: no unary recording holds `speed`, a refusal's
/// `stop_details` or a `fallback` block.
#[test]
fn extras_read_speed_stop_details_and_a_leading_fallback() {
    let refusal = reply(
        json!([
            {"type": "fallback", "from": {"model": CLAUDE_OPUS_5_5}, "to": {"model": CLAUDE_OPUS_4_8}},
            {"type": "text", "text": "ok"}
        ]),
        json!({"input_tokens": 1, "output_tokens": 1, "speed": "fast"}),
        json!({"stop_details": {"type": "refusal", "category": "cyber", "explanation": "no"}}),
    );
    let extras = folded(&wire(CLAUDE_OPUS_5_5), &refusal)
        .extras::<AnthropicExt>()
        .expect("an Anthropic reply")
        .expect("the extras read");
    assert_eq!(extras.speed.as_deref(), Some("fast"));
    assert_eq!(
        extras.stop_details,
        Some(StopDetails {
            kind: "refusal".to_owned(),
            category: Some("cyber".to_owned()),
            explanation: Some("no".to_owned()),
        })
    );
    assert_eq!(extras.fallback_model.as_deref(), Some(CLAUDE_OPUS_4_8));
    assert_eq!(extras.stop_reason.as_deref(), Some("end_turn"));

    let late = reply(
        json!([
            {"type": "text", "text": "ok"},
            {"type": "fallback", "from": {"model": CLAUDE_OPUS_5_5}, "to": {"model": CLAUDE_OPUS_4_8}}
        ]),
        json!({"input_tokens": 1, "output_tokens": 1}),
        json!({}),
    );
    let extras = AnthropicExtras::from_reply(&Api::from_static(MESSAGES_API), &late)
        .expect("the extras read");
    assert_eq!(extras.fallback_model, None, "only a leading block switches");
    assert_eq!(extras.speed, None);
}

/// Extras belong to the provider and route that produced the reply.
#[test]
fn extras_are_none_for_another_provider_and_an_error_for_another_route() {
    let body = reply(
        json!([{"type": "text", "text": "ok"}]),
        json!({"input_tokens": 1, "output_tokens": 1}),
        json!({}),
    );
    let gateway = AnthropicConfig::with_key(&MINIMAX, "sk-test").completion("MiniMax-M2.7");
    assert!(folded(&gateway, &body).extras::<AnthropicExt>().is_none());
    assert!(AnthropicExtras::from_reply(&Api::from_static("openai.chat"), &body).is_err());
}
