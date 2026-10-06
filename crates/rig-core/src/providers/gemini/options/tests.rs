use serde_json::{Value, json};

use crate::completion::{
    CacheRetention, CompletionRequest, Effort, GenerationOptions, OnUnsupported, Reasoning,
    ServiceTier,
};
use crate::error::ProviderError;
use crate::operation::Completion;
use crate::providers::gemini::GeminiConfig;
use crate::providers::gemini::completion::GenerateContent;
use crate::providers::gemini::interactions_api::{InteractionResume, Interactions};
use crate::wire::{Body, Mode, Operation, Wire};

fn sent<W>(wire: &W, request: CompletionRequest) -> Result<Value, ProviderError>
where
    W: Wire<Op = Completion, Payload = crate::wire::Encoded>,
{
    let request = Completion::prepare(request, &wire.describe())?;
    let encoded = wire.encode(request, Mode::Unary)?;
    let Body::Bytes(bytes) = encoded.request.body() else {
        return Err(ProviderError::request("a Gemini body is JSON"));
    };
    if bytes.is_empty() {
        return Ok(Value::Null);
    }
    Ok(serde_json::from_slice(bytes)?)
}

fn with(options: GenerationOptions) -> CompletionRequest {
    CompletionRequest::new("hi").options(options)
}

fn refused(result: Result<Value, ProviderError>) -> Option<&'static str> {
    match result {
        Err(ProviderError::UnsupportedOption(option)) => Some(option.option),
        _ => None,
    }
}

fn rest(model: &str) -> GenerateContent {
    GenerateContent::new(GeminiConfig::new("k"), model)
}

#[test]
fn generate_content_reasoning_follows_the_models_thinking() {
    let body = sent(
        &rest("gemini-2.5-flash"),
        with(GenerationOptions::default().reasoning(Reasoning::Off)),
    )
    .expect("2.5 Flash turns thinking off with a zero budget");
    assert_eq!(
        body["generationConfig"]["thinkingConfig"],
        json!({"thinkingBudget": 0})
    );
    let body = sent(
        &rest("gemini-2.5-pro"),
        with(GenerationOptions::default().reasoning(Reasoning::Budget { tokens: 4096 })),
    )
    .expect("2.5 Pro takes a budget in range");
    assert_eq!(
        body["generationConfig"]["thinkingConfig"],
        json!({"thinkingBudget": 4096})
    );
    let body = sent(
        &rest("gemini-3.8-flash"),
        with(GenerationOptions::default().reasoning(Effort::Low)),
    )
    .expect("3.8 Flash takes `low`");
    assert_eq!(
        body["generationConfig"]["thinkingConfig"],
        json!({"thinkingLevel": "low"})
    );
    for (model, reasoning) in [
        ("gemini-3.8-flash", Reasoning::Effort(Effort::Minimal)),
        ("gemini-3.8-flash", Reasoning::Off),
        ("gemini-3.8-flash", Reasoning::Effort(Effort::XHigh)),
        ("gemini-2.5-pro", Reasoning::Off),
        ("gemini-2.5-pro", Reasoning::Budget { tokens: 64 }),
        ("gemini-2.5-flash", Reasoning::Effort(Effort::High)),
    ] {
        assert_eq!(
            refused(sent(
                &rest(model),
                with(GenerationOptions::default().reasoning(reasoning))
            )),
            Some("reasoning"),
            "{model}: {reasoning:?}"
        );
    }
}

/// A raw `generationConfig` merges over the typed fields and the mapped
/// options, key by key: a key set both ways is the raw one.
#[test]
fn a_raw_generation_config_beats_the_typed_fields_and_keeps_the_rest() {
    let body = sent(
        &rest("gemini-3-flash-preview"),
        with(GenerationOptions::default().top_p(0.5).seed(7))
            .temperature(0.2)
            .max_tokens(64)
            .additional_params(json!({"generationConfig": {"temperature": 0.9, "topK": 3}})),
    )
    .expect("encodes");
    assert_eq!(
        body["generationConfig"],
        json!({"temperature": 0.9, "maxOutputTokens": 64, "topP": 0.5, "seed": 7, "topK": 3})
    );
}

#[test]
fn tiers_follow_the_route() {
    let body = sent(
        &rest("gemini-3-flash-preview"),
        with(GenerationOptions::default().service_tier(ServiceTier::Flex)),
    )
    .expect("the Gemini API takes flex");
    assert_eq!(body["serviceTier"], "flex");
    let body = sent(
        &rest("gemini-3-flash-preview"),
        with(GenerationOptions::default().service_tier(ServiceTier::Auto)),
    )
    .expect("auto is the default");
    assert!(body.get("serviceTier").is_none(), "{body}");
    assert_eq!(
        refused(sent(
            &rest("gemini-3-flash-preview"),
            with(GenerationOptions::default().seed(u64::from(u32::MAX)))
        )),
        Some("seed")
    );
}

#[test]
fn interactions_cells() {
    let wire = Interactions::new(GeminiConfig::new("k"), "gemini-3-flash-preview");
    let body = sent(
        &wire,
        with(
            GenerationOptions::default()
                .reasoning(Effort::High)
                .seed(3)
                .stop(["END"])
                .service_tier(ServiceTier::Priority),
        )
        .temperature(0.1),
    )
    .expect("Interactions options");
    assert_eq!(
        body["generation_config"],
        json!({"temperature": 0.1, "thinking_level": "high", "seed": 3, "stop_sequences": ["END"]})
    );
    assert_eq!(body["service_tier"], "priority");
    for options in [
        GenerationOptions::default().reasoning(Reasoning::Off),
        GenerationOptions::default().top_p(0.5),
        GenerationOptions::default().cache(CacheRetention::Long),
    ] {
        assert!(refused(sent(&wire, with(options))).is_some());
    }
}

#[test]
fn a_resumed_interaction_refuses_options_and_raw_params() {
    let resume = InteractionResume::new(GeminiConfig::new("k"), "interaction-1");
    assert_eq!(
        refused(sent(&resume, with(GenerationOptions::default().seed(1)))),
        Some("seed")
    );
    let body = sent(
        &resume,
        with(
            GenerationOptions::default()
                .seed(1)
                .on_unsupported(OnUnsupported::Ignore),
        ),
    )
    .expect("an ignored option leaves an empty read");
    assert_eq!(body, Value::Null);
    let error = sent(
        &resume,
        CompletionRequest::new("hi").additional_params(json!({"stream": true})),
    )
    .expect_err("raw parameters have nothing to reach");
    assert!(error.to_string().contains("additional_params"), "{error}");
}
