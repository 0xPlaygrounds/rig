use serde_json::{Value, json};

use super::*;
use crate::completion::{GenerationOptions, OnUnsupported, Verbosity};
use crate::error::ProviderError;
use crate::operation::Completion;
use crate::providers::anthropic::completion::{
    CLAUDE_FABLE_5, CLAUDE_HAIKU_4_5, CLAUDE_OPUS_4_6, CLAUDE_OPUS_4_8, CLAUDE_OPUS_5,
    CLAUDE_OPUS_5_5, CLAUDE_SONNET_4_6, CLAUDE_SONNET_5, CLAUDE_SONNET_5_5,
};
use crate::providers::anthropic::wire::AnthropicConfig;
use crate::wire::{Body, Mode, Operation, Wire};

fn wire(model: &str) -> Messages {
    AnthropicConfig::new("sk-test").completion(model)
}

fn gateway(dialect: &crate::providers::anthropic::Dialect, model: &str) -> Messages {
    AnthropicConfig::with_key(dialect, "sk-test").completion(model)
}

/// The body `wire` sends for `request`, prepared as the driver prepares it.
fn sent(wire: &Messages, request: CompletionRequest, mode: Mode) -> Result<Value, ProviderError> {
    let request = Completion::prepare(request, &wire.describe())?;
    let encoded = wire.encode(request, mode)?;
    let Body::Bytes(bytes) = encoded.request.body() else {
        return Err(ProviderError::request("a Messages body is JSON"));
    };
    Ok(serde_json::from_slice(bytes)?)
}

fn request(options: GenerationOptions) -> CompletionRequest {
    CompletionRequest::new("Answer the question")
        .max_tokens(2048)
        .options(options)
}

fn schema() -> crate::schemars::Schema {
    serde_json::from_value(json!({"type": "object", "properties": {"answer": {"type": "string"}}}))
        .expect("a schema")
}

fn refused(result: Result<Value, ProviderError>) -> &'static str {
    match result {
        Err(ProviderError::UnsupportedOption(option)) => option.option,
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn claude_class_reads_every_spelling_of_a_model() {
    assert_eq!(claude_class(CLAUDE_HAIKU_4_5), Some(ClaudeClass::A0));
    assert_eq!(
        claude_class("claude-haiku-4-5-20251001"),
        Some(ClaudeClass::A0)
    );
    assert_eq!(
        claude_class("us.anthropic.claude-haiku-4-5-20251001-v1:0"),
        Some(ClaudeClass::A0)
    );
    assert_eq!(
        claude_class("anthropic/claude-opus-5.5"),
        Some(ClaudeClass::A7)
    );
    assert_eq!(
        claude_class("us.anthropic.claude-sonnet-5"),
        Some(ClaudeClass::A4)
    );
    assert_eq!(claude_class(CLAUDE_SONNET_5_5), Some(ClaudeClass::A6));
    assert_eq!(claude_class(CLAUDE_OPUS_5), Some(ClaudeClass::A5));
    assert_eq!(claude_class(CLAUDE_FABLE_5), Some(ClaudeClass::A8));
    assert_eq!(claude_class("claude-opus-5-50"), None);
    assert_eq!(claude_class("custom-model"), None);
}

/// Ported from #2616 (`typed_thinking_and_effort_serialize_to_the_documented_wire_values`):
/// each class's thinking and effort, as the mapping table states them.
#[test]
fn reasoning_takes_each_classes_documented_shape() {
    let reasoning = |model: &str, reasoning: Reasoning| {
        sent(
            &wire(model),
            request(GenerationOptions::default().reasoning(reasoning)),
            Mode::Unary,
        )
    };
    for (model, effort) in [
        (CLAUDE_OPUS_4_6, Effort::Max),
        (CLAUDE_OPUS_4_8, Effort::XHigh),
        (CLAUDE_SONNET_5, Effort::Low),
        (CLAUDE_OPUS_5_5, Effort::Medium),
        ("custom-model", Effort::High),
    ] {
        let body = reasoning(model, Reasoning::Effort(effort)).expect("the level is taken");
        assert_eq!(body["thinking"]["type"], "adaptive", "{model}");
        assert_eq!(body["output_config"]["effort"], effort.as_str(), "{model}");
    }
    let body = reasoning("claude-opus-4-5", Reasoning::Effort(Effort::High))
        .expect("Opus 4.5 takes an effort");
    assert_eq!(body["output_config"], json!({"effort": "high"}));
    assert!(body.get("thinking").is_none(), "{body}");

    for (model, off) in [
        (CLAUDE_HAIKU_4_5, json!({"type": "disabled"})),
        (CLAUDE_OPUS_5, json!({"type": "disabled"})),
        (CLAUDE_SONNET_5_5, json!({"type": "between_tools"})),
    ] {
        let body = reasoning(model, Reasoning::Off).expect("thinking can be turned off");
        assert_eq!(body["thinking"], off, "{model}");
    }
    let body = reasoning(CLAUDE_OPUS_4_6, Reasoning::Budget { tokens: 1024 })
        .expect("Opus 4.6 takes a budget");
    assert_eq!(
        body["thinking"],
        json!({"type": "enabled", "budget_tokens": 1024})
    );

    for (model, refused_reasoning) in [
        (CLAUDE_OPUS_5_5, Reasoning::Off),
        (CLAUDE_FABLE_5, Reasoning::Off),
        (CLAUDE_HAIKU_4_5, Reasoning::Effort(Effort::High)),
        (CLAUDE_OPUS_4_6, Reasoning::Effort(Effort::XHigh)),
        (CLAUDE_OPUS_4_8, Reasoning::Effort(Effort::Minimal)),
        (CLAUDE_OPUS_4_8, Reasoning::Budget { tokens: 2048 }),
        (CLAUDE_HAIKU_4_5, Reasoning::Budget { tokens: 512 }),
        (CLAUDE_HAIKU_4_5, Reasoning::Budget { tokens: 4096 }),
    ] {
        assert_eq!(
            refused(reasoning(model, refused_reasoning)),
            "reasoning",
            "{model}: {refused_reasoning:?}"
        );
    }
}

/// Ported from #1480 (`adaptive_parameters_validate_legacy_models_without_rejecting_newer_ones`):
/// a model without adaptive thinking refuses an effort level, and the newer
/// models, or one the table does not name, take every level they list.
#[test]
fn legacy_models_refuse_an_effort_newer_ones_take_it() {
    for model in ["claude-sonnet-4-5", CLAUDE_HAIKU_4_5] {
        let result = sent(
            &wire(model),
            request(GenerationOptions::default().reasoning(Effort::Max)),
            Mode::Unary,
        );
        assert_eq!(refused(result), "reasoning", "{model}");
    }
    for model in [
        CLAUDE_OPUS_4_6,
        CLAUDE_SONNET_4_6,
        CLAUDE_OPUS_4_8,
        "custom-model",
    ] {
        let body = sent(
            &wire(model),
            request(GenerationOptions::default().reasoning(Effort::Max)),
            Mode::Unary,
        )
        .unwrap_or_else(|error| panic!("{model}: {error}"));
        assert_eq!(body["output_config"]["effort"], "max", "{model}");
    }
}

#[test]
fn the_remaining_options_take_their_documented_cells() {
    let one =
        |options: GenerationOptions| sent(&wire(CLAUDE_OPUS_4_6), request(options), Mode::Unary);
    let body = one(GenerationOptions::default().top_p(0.5)).expect("Opus 4.6 takes top_p");
    assert_eq!(body["top_p"], 0.5);
    assert_eq!(
        refused(one(GenerationOptions::default().top_p(0.5)).and_then(|_| {
            sent(
                &wire(CLAUDE_OPUS_4_6),
                request(GenerationOptions::default().top_p(0.5)).temperature(0.2),
                Mode::Unary,
            )
        })),
        "top_p"
    );
    let body = one(GenerationOptions::default().stop(["END"])).expect("stop sequences");
    assert_eq!(body["stop_sequences"], json!(["END"]));
    let body =
        one(GenerationOptions::default().service_tier(ServiceTier::Auto)).expect("the auto tier");
    assert_eq!(body["service_tier"], "auto");
    for options in [
        GenerationOptions::default().service_tier(ServiceTier::Priority),
        GenerationOptions::default().service_tier(ServiceTier::Flex),
        GenerationOptions::default().verbosity(Verbosity::Low),
        GenerationOptions::default().seed(1),
    ] {
        assert!(matches!(
            one(options),
            Err(ProviderError::UnsupportedOption(_))
        ));
    }

    // `parallel_tool_calls: false` joins the request's tool choice.
    let tool = crate::completion::ToolDefinition::new(
        crate::message::ToolName::new("lookup").expect("a name"),
        "Look a word up.",
        json!({"type": "object", "properties": {}}),
    );
    let body = sent(
        &wire(CLAUDE_OPUS_4_6),
        request(GenerationOptions::default().parallel_tool_calls(false)).tool(tool.clone()),
        Mode::Unary,
    )
    .expect("parallel tool use can be turned off");
    assert_eq!(
        body["tool_choice"],
        json!({"type": "auto", "disable_parallel_tool_use": true})
    );
    let body = sent(
        &wire(CLAUDE_OPUS_4_6),
        request(GenerationOptions::default().parallel_tool_calls(false))
            .tool(tool)
            .tool_choice(ToolChoice::Required),
        Mode::Unary,
    )
    .expect("a required choice takes the flag");
    assert_eq!(
        body["tool_choice"],
        json!({"type": "any", "disable_parallel_tool_use": true})
    );
}

/// Ported from #2616 (`output_config_merges_effort_additional_params_and_the_schema_format_into_one_key`).
#[test]
fn output_config_merges_effort_additional_params_and_the_schema_format_into_one_key() {
    let wire = wire(CLAUDE_OPUS_5_5);
    let high = || GenerationOptions::default().reasoning(Effort::High);

    // The mapped effort and the schema's format share one object.
    let body = sent(&wire, request(high()).output_schema(schema()), Mode::Unary).expect("encodes");
    assert_eq!(body["output_config"]["effort"], "high");
    assert_eq!(body["output_config"]["format"]["type"], "json_schema");

    // A raw effort overrides the mapped one, and unknown keys pass through.
    let body = sent(
        &wire,
        request(high())
            .output_schema(schema())
            .additional_params(json!({
                "output_config": {"effort": "low", "task_budget": {"tokens": 1000}}
            })),
        Mode::Unary,
    )
    .expect("encodes");
    assert_eq!(body["output_config"]["effort"], "low");
    assert_eq!(body["output_config"]["task_budget"]["tokens"], 1000);
    assert_eq!(body["output_config"]["format"]["type"], "json_schema");

    // `additional_params` alone still lands once.
    let body = sent(
        &wire,
        request(GenerationOptions::default())
            .additional_params(json!({"output_config": {"effort": "max"}})),
        Mode::Unary,
    )
    .expect("encodes");
    assert_eq!(body["output_config"], json!({"effort": "max"}));

    // No setting sends no `output_config`.
    let body = sent(&wire, request(GenerationOptions::default()), Mode::Unary).expect("encodes");
    assert!(body.get("output_config").is_none(), "{body}");
}

/// Replaces #2616's `output_config_rejects_a_format_that_conflicts_with_the_output_schema`:
/// `additional_params` is above the request's own fields, so a raw format
/// merges over the schema's key by key, beside the mapped effort.
#[test]
fn a_raw_output_format_is_above_the_schemas() {
    let raw = json!({"type": "json_schema", "schema": {"additionalProperties": true}});
    let body = sent(
        &wire(CLAUDE_OPUS_5_5),
        request(GenerationOptions::default().reasoning(Effort::High))
            .output_schema(schema())
            .additional_params(json!({"output_config": {"format": raw}})),
        Mode::Unary,
    )
    .expect("encodes");
    assert_eq!(
        body["output_config"]["format"]["schema"]["additionalProperties"],
        true
    );
    assert_eq!(
        body["output_config"]["format"]["schema"]["properties"]["answer"]["type"],
        "string"
    );
    assert_eq!(body["output_config"]["effort"], "high");
}

/// Ported from #2616 (`thinking_merges_typed_defaults_with_additional_params`):
/// raw thinking keys merge into the mapped thinking, and a raw `type`
/// replaces the mapped one.
#[test]
fn thinking_merges_the_mapped_thinking_with_additional_params() {
    let wire = wire(CLAUDE_OPUS_5_5);
    let high = || GenerationOptions::default().reasoning(Effort::High);
    let body = sent(
        &wire,
        request(high()).additional_params(json!({
            "thinking": {
                "display": "summarized",
                "block_binding": {"prefix_mismatch_behavior": "error"}
            }
        })),
        Mode::Unary,
    )
    .expect("encodes");
    assert_eq!(
        body["thinking"],
        json!({
            "type": "adaptive",
            "display": "summarized",
            "block_binding": {"prefix_mismatch_behavior": "error"}
        })
    );

    let body = sent(
        &wire,
        request(high()).additional_params(json!({"thinking": {"type": "disabled"}})),
        Mode::Unary,
    )
    .expect("encodes");
    assert_eq!(body["thinking"], json!({"type": "disabled"}));
}

/// Ported from #1480 (`adaptive_parameters_merge_schema_effort_tools_and_passthrough`).
#[test]
fn adaptive_parameters_merge_schema_effort_tools_and_passthrough() {
    let body = sent(
        &wire(CLAUDE_SONNET_4_6),
        request(GenerationOptions::default().reasoning(Effort::Max))
            .output_schema(schema())
            .additional_params(json!({
                "thinking": {"display": "omitted"},
                "output_config": {"custom_option": true},
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                "metadata": {"user_id": "test-only"}
            })),
        Mode::Unary,
    )
    .expect("encodes");
    assert_eq!(
        body["thinking"],
        json!({"type": "adaptive", "display": "omitted"})
    );
    assert_eq!(body["output_config"]["effort"], "max");
    assert_eq!(body["output_config"]["custom_option"], true);
    assert_eq!(body["output_config"]["format"]["schema"]["type"], "object");
    assert_eq!(body["tools"].as_array().map(Vec::len), Some(1));
    assert_eq!(body["metadata"]["user_id"], "test-only");
}

/// Ported from #1480 (`streaming_adaptive_effort_and_schema_share_blocking_conversion`).
#[test]
fn a_streamed_body_is_the_unary_one_with_stream_set() {
    let wire = wire(CLAUDE_OPUS_4_8);
    let request = || {
        request(GenerationOptions::default().reasoning(Effort::High))
            .output_schema(schema())
            .additional_params(json!({"metadata": {"user_id": "test-only"}}))
    };
    let streamed = sent(&wire, request(), Mode::Streaming).expect("encodes");
    let mut unary = sent(&wire, request(), Mode::Unary).expect("encodes");
    unary["stream"] = json!(true);
    assert_eq!(streamed, unary);
    assert_eq!(
        streamed["output_config"]["format"]["schema"]["type"],
        "object"
    );
}

/// Replaces #2616's `gateways_pass_thinking_and_effort_through_unchecked`:
/// a gateway refuses a reasoning option no vendor page or recording
/// confirms, and passes raw thinking and effort through unchecked.
#[test]
fn gateways_refuse_unverified_reasoning_and_pass_raw_keys_through() {
    use crate::providers::anthropic::wire::ZAI;
    let zai = gateway(&ZAI, "glm-5");
    assert_eq!(
        refused(sent(
            &zai,
            request(GenerationOptions::default().reasoning(Effort::High)),
            Mode::Unary
        )),
        "reasoning"
    );
    let body = sent(
        &zai,
        request(GenerationOptions::default()).additional_params(json!({
            "thinking": {"type": "enabled"},
            "output_config": {"effort": "turbo"}
        })),
        Mode::Unary,
    )
    .expect("a gateway passes raw keys through");
    assert_eq!(body["thinking"], json!({"type": "enabled"}));
    assert_eq!(body["output_config"]["effort"], "turbo");
}

#[test]
fn minimax_places_block_markers_for_a_short_cache() {
    use crate::providers::anthropic::wire::MINIMAX;
    let minimax = gateway(&MINIMAX, "MiniMax-M2.7");
    let body = sent(
        &minimax,
        CompletionRequest::new("hi")
            .preamble("Be brief.")
            .max_tokens(64)
            .options(GenerationOptions::default().cache(CacheRetention::Short)),
        Mode::Unary,
    )
    .expect("encodes");
    assert!(body.get("cache_control").is_none(), "{body}");
    assert_eq!(
        body["system"][0]["cache_control"],
        json!({"type": "ephemeral"})
    );
    assert_eq!(
        body["messages"][0]["content"][0]["cache_control"],
        json!({"type": "ephemeral"})
    );
    assert_eq!(
        refused(sent(
            &minimax,
            request(GenerationOptions::default().cache(CacheRetention::Long)),
            Mode::Unary
        )),
        "cache"
    );
}

#[test]
fn a_cache_none_is_refused_when_the_wire_places_markers() {
    let placing = wire(CLAUDE_OPUS_4_6).with_prompt_caching();
    assert_eq!(
        refused(sent(
            &placing,
            request(GenerationOptions::default().cache(CacheRetention::None)),
            Mode::Unary
        )),
        "cache"
    );
    let body = sent(
        &wire(CLAUDE_OPUS_4_6),
        request(GenerationOptions::default().cache(CacheRetention::None)),
        Mode::Unary,
    )
    .expect("no marker is the default");
    assert!(body.get("cache_control").is_none(), "{body}");

    // Under `Ignore` a refused option is skipped with a warning.
    let body = sent(
        &placing,
        request(
            GenerationOptions::default()
                .cache(CacheRetention::None)
                .on_unsupported(OnUnsupported::Ignore),
        ),
        Mode::Unary,
    )
    .expect("skipped");
    assert!(body.get("cache_control").is_none(), "{body}");
}

/// Tools that only arrive through `additional_params.tools` still count:
/// `parallel_tool_calls(false)` sets the flag rather than being dropped.
#[test]
fn raw_tools_count_for_parallel_tool_calls() {
    let raw =
        json!({"tools": [{"name": "t", "description": "d", "input_schema": {"type": "object"}}]});
    let body = sent(
        &wire(CLAUDE_OPUS_4_6),
        request(GenerationOptions::default().parallel_tool_calls(false))
            .additional_params(raw.clone()),
        Mode::Unary,
    )
    .expect("raw tools can be called in parallel");
    assert_eq!(
        body["tool_choice"],
        json!({"type": "auto", "disable_parallel_tool_use": true})
    );
    assert_eq!(body["tools"].as_array().map(Vec::len), Some(1));

    let body = sent(
        &gateway(&XIAOMIMIMO, "mimo-v2-flash"),
        request(GenerationOptions::default().parallel_tool_calls(false)).additional_params(raw),
        Mode::Unary,
    )
    .expect("MiMo takes the flag for raw tools too");
    assert_eq!(
        body["tool_choice"],
        json!({"type": "auto", "disable_parallel_tool_use": true})
    );
}
