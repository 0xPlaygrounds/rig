use std::collections::BTreeSet;

use super::*;
use crate::completion::{AMAZON_NOVA_LITE, ANTHROPIC_CLAUDE_SONNET_4_5, Converse};
use crate::streaming::tests::{CLAUDE, NOVA};
use rig_core::completion::{
    CacheRetention, CompletionRequest, Effort, GenerationOptions, OnUnsupported, ProviderOptions,
    Reasoning, ServiceTier, ToolDefinition, Verbosity,
};
use rig_core::message::{ToolChoice, ToolName};
use rig_core::operation::Completion;
use rig_core::wire::{Mode, Operation, Wire};
use serde_json::json;

/// A provider that is not Bedrock but takes Bedrock's options type.
enum Other {}

impl ProviderExtension for Other {
    const PROVIDER: &'static str = "other";
    type Options = BedrockOptions;
    type Extras = BedrockExtras;
}

fn every_field() -> BedrockOptions {
    BedrockOptions::default()
        .guardrail(Guardrail::new("g1", "DRAFT").trace(GuardrailTrace::EnabledFull))
        .performance_latency(PerformanceLatency::Optimized)
        .request_metadata("team", "search")
        .additional_response_field_path("/stop_sequence")
        .anthropic_beta("context-1m-2025-08-07")
        .top_k(5)
}

fn with(options: &BedrockOptions) -> CompletionRequest {
    CompletionRequest::new("q").provider_options(
        ProviderOptions::new()
            .with::<BedrockExt>(options)
            .expect("the options serialize"),
    )
}

/// The body `request` sends to `model` in `mode`, prepared as the driver
/// prepares it.
fn encoded(model: &str, request: CompletionRequest, mode: Mode) -> Value {
    let wire = Converse::new(model);
    let request = Completion::prepare(request, &wire.describe()).expect("prepares");
    serde_json::to_value(wire.encode(request, mode).expect("encodes").body).expect("serializes")
}

fn sent(options: &BedrockOptions) -> Value {
    encoded(NOVA, with(options), Mode::Unary)
}

#[test]
fn guardrail_is_sent_on_both_modes() {
    let options = BedrockOptions::default()
        .guardrail(Guardrail::new("g1", "DRAFT").trace(GuardrailTrace::Enabled));
    for mode in [Mode::Unary, Mode::Streaming] {
        assert_eq!(
            encoded(NOVA, with(&options), mode)["guardrailConfig"],
            json!({"guardrailIdentifier": "g1", "guardrailVersion": "DRAFT", "trace": "enabled"}),
            "{mode:?}"
        );
    }
    let untraced = BedrockOptions::default().guardrail(Guardrail::new("g1", "3"));
    assert_eq!(
        sent(&untraced)["guardrailConfig"],
        json!({"guardrailIdentifier": "g1", "guardrailVersion": "3"})
    );
}

#[test]
fn performance_latency_goes_to_performance_config() {
    let options = BedrockOptions::default().performance_latency(PerformanceLatency::Optimized);
    assert_eq!(
        sent(&options)["performanceConfig"],
        json!({"latency": "optimized"})
    );
}

#[test]
fn request_metadata_is_sent_as_given() {
    let options = BedrockOptions::default()
        .request_metadata("team", "search")
        .request_metadata("run", "42");
    assert_eq!(
        sent(&options)["requestMetadata"],
        json!({"team": "search", "run": "42"})
    );
}

#[test]
fn response_field_paths_are_sent_as_given() {
    let options = BedrockOptions::default()
        .additional_response_field_path("/stop_sequence")
        .additional_response_field_path("/amazon-bedrock-trace");
    assert_eq!(
        sent(&options)["additionalModelResponseFieldPaths"],
        json!(["/stop_sequence", "/amazon-bedrock-trace"])
    );
}

/// The betas join the mapped thinking in the model's own request fields.
#[test]
fn anthropic_beta_joins_the_model_fields() {
    let options = BedrockOptions::default().anthropic_beta("context-1m-2025-08-07");
    let request = with(&options).options(GenerationOptions::default().reasoning(Effort::High));
    let body = encoded(CLAUDE, request, Mode::Unary);
    assert_eq!(
        body["additionalModelRequestFields"]["anthropic_beta"],
        json!(["context-1m-2025-08-07"])
    );
    assert_eq!(
        body["additionalModelRequestFields"]["thinking"],
        json!({"type": "adaptive"})
    );
}

/// `top_k` goes in the model's own request fields, where a raw `top_k`
/// replaces it.
#[test]
fn top_k_goes_to_the_model_fields_and_raw_beats_it() {
    let options = BedrockOptions::default().top_k(5);
    assert_eq!(
        encoded(CLAUDE, with(&options), Mode::Unary)["additionalModelRequestFields"],
        json!({"top_k": 5})
    );
    let mut request = with(&options);
    request.additional_params = Some(json!({"top_k": 9}));
    assert_eq!(
        encoded(CLAUDE, request, Mode::Unary)["additionalModelRequestFields"],
        json!({"top_k": 9})
    );
}

#[test]
fn another_providers_entry_is_not_read() {
    let request = CompletionRequest::new("q").provider_options(
        ProviderOptions::new()
            .with::<Other>(&every_field())
            .expect("the options serialize"),
    );
    assert_eq!(
        encoded(NOVA, request, Mode::Unary),
        encoded(NOVA, CompletionRequest::new("q"), Mode::Unary)
    );
}

/// Every JSON pointer to a leaf of `value`, an empty object or array being
/// a leaf.
fn leaves(value: &Value, at: &str, out: &mut BTreeSet<String>) {
    match value {
        Value::Object(fields) if !fields.is_empty() => {
            for (key, field) in fields {
                leaves(field, &format!("{at}/{key}"), out);
            }
        }
        _ => {
            out.insert(at.to_owned());
        }
    }
}

/// No field writes a leaf that a mapped option or the request's own fields
/// write, or a leaf above or below one, on any model family.
#[test]
fn no_field_writes_a_reserved_leaf() {
    let sections = ProviderOptions::new()
        .with::<BedrockExt>(&every_field())
        .expect("the options serialize");
    let mut provider = BTreeSet::new();
    leaves(
        &sections.get::<BedrockExt>().expect("an entry")[rig_core::completion::SHARED],
        "",
        &mut provider,
    );
    let reasonings = [
        Reasoning::Off,
        Reasoning::Effort(Effort::High),
        Reasoning::Budget { tokens: 2048 },
    ];
    for model in [CLAUDE, ANTHROPIC_CLAUDE_SONNET_4_5, AMAZON_NOVA_LITE] {
        for reasoning in reasonings {
            let mut request = CompletionRequest::new("q")
                .options(
                    GenerationOptions::default()
                        .reasoning(reasoning)
                        .cache(CacheRetention::Short)
                        .service_tier(ServiceTier::Priority)
                        .verbosity(Verbosity::Low)
                        .parallel_tool_calls(true)
                        .top_p(0.5)
                        .seed(1)
                        .stop(["END"])
                        .on_unsupported(OnUnsupported::Ignore),
                )
                .preamble("be brief".to_owned());
            request.temperature = Some(0.5);
            request.max_tokens = Some(4096);
            request.tools = vec![ToolDefinition::new(
                ToolName::new("lookup").expect("a tool name"),
                "a tool",
                json!({"type": "object"}),
            )];
            request.tool_choice = Some(ToolChoice::Auto);
            request.output_schema = Some(schemars::json_schema!({"type": "object"}));
            let mut owned = BTreeSet::new();
            leaves(&encoded(model, request, Mode::Unary), "", &mut owned);
            for leaf in &provider {
                for reserved in &owned {
                    assert!(
                        !(leaf == reserved
                            || leaf.starts_with(&format!("{reserved}/"))
                            || reserved.starts_with(&format!("{leaf}/"))),
                        "{model}: {leaf} meets {reserved}"
                    );
                }
            }
        }
    }
}

/// Not a cassette test: no Bedrock recording carries `serviceTier`,
/// `performanceConfig`, `cacheDetails`, a prompt router's trace or
/// `additionalModelResponseFields`, so this document is built here.
#[test]
fn extras_read_every_field_of_a_converse_document() {
    let raw = json!({
        "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 10,
            "outputTokens": 2,
            "totalTokens": 12,
            "cacheDetails": [{"inputTokens": 1024, "ttl": "1h"}]
        },
        "metrics": {"latencyMs": 321},
        "serviceTier": {"type": "priority"},
        "performanceConfig": {"latency": "optimized"},
        "trace": {"promptRouter": {"invokedModelId": "arn:model/claude"}},
        "additionalModelResponseFields": {"stop_sequence": null}
    });
    let api = Api::from_static("bedrock.converse");
    let extras = BedrockExtras::from_reply(&api, &raw).expect("the document reads");
    assert_eq!(extras.stop_reason.as_deref(), Some("end_turn"));
    assert_eq!(extras.latency_ms, Some(321));
    assert_eq!(
        extras.cache_details,
        Some(vec![CacheDetail {
            input_tokens: Some(1024),
            ttl: Some("1h".to_owned()),
        }])
    );
    assert_eq!(extras.service_tier.as_deref(), Some("priority"));
    assert_eq!(extras.performance_latency.as_deref(), Some("optimized"));
    assert_eq!(extras.invoked_model_id.as_deref(), Some("arn:model/claude"));
    assert_eq!(extras.trace, Some(raw["trace"].clone()));
    assert_eq!(
        extras.additional_model_response_fields,
        Some(json!({"stop_sequence": null}))
    );

    let bare = BedrockExtras::from_reply(&api, &json!({"stopReason": "end_turn"}))
        .expect("the document reads");
    assert_eq!(
        bare,
        BedrockExtras {
            stop_reason: Some("end_turn".to_owned()),
            ..BedrockExtras::default()
        }
    );
    assert!(BedrockExtras::from_reply(&api, &json!({"metrics": {"latencyMs": "fast"}})).is_err());
}
