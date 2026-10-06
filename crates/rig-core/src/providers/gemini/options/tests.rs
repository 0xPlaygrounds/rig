use serde_json::{Value, json};

use crate::completion::{
    CacheRetention, CompletionRequest, Effort, GenerationOptions, OnUnsupported,
    ProviderToolDefinition, Reasoning, ServiceTier,
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
        Err(ProviderError::UnsupportedOption(option)) => match option.option {
            std::borrow::Cow::Borrowed(name) => Some(name),
            std::borrow::Cow::Owned(name) => panic!("a GenerationOptions field, got {name}"),
        },
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

/// On GenerateContent the encoder refuses a reasoning option exactly when
/// `ModelSpec::validate` does, for every Gemini API row.
#[test]
fn generate_content_reasoning_agrees_with_validate() {
    let reasonings = [
        Reasoning::Off,
        Reasoning::Effort(Effort::Minimal),
        Reasoning::Effort(Effort::Low),
        Reasoning::Effort(Effort::Medium),
        Reasoning::Effort(Effort::High),
        Reasoning::Effort(Effort::XHigh),
        Reasoning::Budget { tokens: 0 },
        Reasoning::Budget { tokens: 1024 },
        Reasoning::Budget { tokens: 30_000 },
    ];
    let mut checked = 0;
    for spec in crate::catalog::Catalog::builtin()
        .iter()
        .filter(|spec| spec.provider.vendor() == "gcp.gemini")
    {
        for reasoning in &reasonings {
            let options = GenerationOptions::default().reasoning(*reasoning);
            let encoded = sent(&rest(&spec.id), with(options.clone()));
            assert_eq!(
                refused(encoded).is_some(),
                spec.validate(&options).is_err(),
                "{}: {reasoning:?}",
                spec.id
            );
            checked += 1;
        }
    }
    assert!(checked > 300, "{checked}");
}

/// Gemma 4 turns thinking on with level `high` and off with `minimal`; the
/// image models think as their model pages say.
#[test]
fn gemma_and_the_image_models_think_as_documented() {
    for model in ["gemma-4-26b-a4b-it", "gemma-4-31b-it"] {
        let body = sent(
            &rest(model),
            with(GenerationOptions::default().reasoning(Reasoning::Off)),
        )
        .expect("Gemma 4 turns thinking off");
        assert_eq!(
            body["generationConfig"]["thinkingConfig"],
            json!({"thinkingLevel": "minimal"})
        );
        let body = sent(
            &rest(model),
            with(GenerationOptions::default().reasoning(Effort::High)),
        )
        .expect("Gemma 4 thinks at `high`");
        assert_eq!(
            body["generationConfig"]["thinkingConfig"],
            json!({"thinkingLevel": "high"})
        );
    }
    let body = sent(
        &rest("gemini-3-pro-image-preview"),
        with(GenerationOptions::default().reasoning(Effort::Low)),
    )
    .expect("the preview takes the GA model's levels");
    assert_eq!(
        body["generationConfig"]["thinkingConfig"],
        json!({"thinkingLevel": "low"})
    );
    let body = sent(
        &rest("gemini-2.5-flash-image"),
        with(GenerationOptions::default().reasoning(Reasoning::Off)),
    )
    .expect("2.5 Flash Image does not think");
    assert!(
        body["generationConfig"].get("thinkingConfig").is_none(),
        "{body}"
    );
}

/// An id the catalog does not list takes the thinking of the model it
/// versions: a `-001` revision or a dated preview.
#[test]
fn a_versioned_gemini_id_takes_its_models_thinking() {
    let body = sent(
        &rest("gemini-2.5-flash-preview-09-2025"),
        with(GenerationOptions::default().reasoning(Reasoning::Off)),
    )
    .expect("a 2.5 Flash preview turns thinking off");
    assert_eq!(
        body["generationConfig"]["thinkingConfig"],
        json!({"thinkingBudget": 0})
    );
    assert_eq!(
        refused(sent(
            &rest("models/gemini-2.5-flash-lite-preview-06-17"),
            with(GenerationOptions::default().reasoning(Reasoning::Budget { tokens: 100 })),
        )),
        Some("reasoning"),
        "2.5 Flash-Lite's budget starts at 512"
    );
    let body = sent(
        &rest("gemini-2.0-flash-001"),
        with(GenerationOptions::default().reasoning(Reasoning::Off)),
    )
    .expect("2.0 Flash does not think");
    assert!(
        body.get("generationConfig")
            .is_none_or(|config| config.get("thinkingConfig").is_none()),
        "{body}"
    );
    let body = sent(
        &rest("gemini-1.5-pro-002"),
        with(GenerationOptions::default().reasoning(Reasoning::Off)),
    )
    .expect("an unlisted model before 2.5 does not think");
    assert!(
        body.get("generationConfig")
            .is_none_or(|config| config.get("thinkingConfig").is_none()),
        "{body}"
    );
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

/// A raw `generationConfig: null` is absent, as it was before the merge:
/// the typed fields and the mapped thinking are still sent.
#[test]
fn a_raw_null_generation_config_keeps_the_typed_fields() {
    let schema: schemars::Schema =
        serde_json::from_value(json!({"type": "object"})).expect("a schema");
    let body = sent(
        &rest("gemini-2.5-flash"),
        with(GenerationOptions::default().reasoning(Reasoning::Off))
            .temperature(0.2)
            .max_tokens(64)
            .output_schema(schema)
            .additional_params(json!({"generationConfig": null})),
    )
    .expect("encodes");
    assert_eq!(
        body["generationConfig"],
        json!({
            "responseMimeType": "application/json",
            "responseJsonSchema": {"type": "object"},
            "temperature": 0.2,
            "maxOutputTokens": 64,
            "thinkingConfig": {"thinkingBudget": 0},
        })
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
    // Raw tools leave the raw layer for the base to append; the resume base
    // refuses them rather than drop them.
    for request in [
        CompletionRequest::new("hi").additional_params(json!({"tools": [{"googleSearch": {}}]})),
        CompletionRequest::new("hi").provider_tool(ProviderToolDefinition::new("googleSearch")),
    ] {
        let error = sent(&resume, request).expect_err("raw tools have nothing to reach");
        assert!(
            error.to_string().contains("additional_params.tools"),
            "{error}"
        );
    }
}
