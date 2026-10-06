use serde_json::{Value, json};

use super::*;
use crate::completion::{GenerationOptions, Verbosity};
use crate::error::ProviderError;
use crate::operation::Completion;
use crate::providers::openai::wire::{
    AZURE, Chat, DEEPSEEK, Dialect, GROQ, MISTRAL, MOONSHOT, OLLAMA, OPENAI, OPENROUTER,
    OpenAIConfig, VENICE,
};
use crate::wire::{Body, Mode, Operation, Wire};

fn chat(dialect: &Dialect, model: &str) -> Chat {
    Chat::new(OpenAIConfig::with_key(dialect, "sk-test"), model)
}

/// The body `chat` sends for `options` alone, prepared as the driver
/// prepares it.
fn sent(chat: &Chat, options: GenerationOptions) -> Result<Value, ProviderError> {
    let request = CompletionRequest::new("hi").max_tokens(16).options(options);
    let request = Completion::prepare(request, &chat.describe())?;
    let encoded = chat.encode(request, Mode::Unary)?;
    let Body::Bytes(bytes) = encoded.request.body() else {
        return Err(ProviderError::request("a chat body is JSON"));
    };
    Ok(serde_json::from_slice(bytes)?)
}

fn refused(result: Result<Value, ProviderError>) -> Option<&'static str> {
    match result {
        Err(ProviderError::UnsupportedOption(option)) => Some(option.option),
        _ => None,
    }
}

#[test]
fn gpt_versions_read_every_spelling() {
    assert_eq!(gpt_version("gpt-5"), Some((5, 0)));
    assert_eq!(gpt_version("gpt-5.5-pro"), Some((5, 5)));
    assert_eq!(gpt_version("openai/gpt-5.6-sol"), Some((5, 6)));
    assert_eq!(gpt_version("gpt-6-sol"), Some((6, 0)));
    assert_eq!(gpt_version("gpt-4o-mini"), Some((4, 0)));
    assert_eq!(gpt_version("o3"), None);
    assert_eq!(reasons("o3-mini"), Some(true));
    assert_eq!(reasons("gpt-4.1"), Some(false));
    assert_eq!(reasons("my-deployment"), None);
}

#[test]
fn openai_chat_cells() {
    let gpt = |model| chat(&OPENAI, model);
    let body = sent(
        &gpt("gpt-5.2"),
        GenerationOptions::default().reasoning(Reasoning::Off),
    )
    .expect("GPT-5.2 turns reasoning off");
    assert_eq!(body["reasoning_effort"], "none");
    assert_eq!(
        refused(sent(
            &gpt("gpt-5"),
            GenerationOptions::default().reasoning(Reasoning::Off)
        )),
        Some("reasoning")
    );
    let body = sent(
        &gpt("gpt-4o"),
        GenerationOptions::default().reasoning(Reasoning::Off),
    )
    .expect("a model that does not reason is already off");
    assert!(body.get("reasoning_effort").is_none(), "{body}");
    assert_eq!(
        refused(sent(
            &gpt("gpt-4o"),
            GenerationOptions::default().reasoning(Effort::High)
        )),
        Some("reasoning")
    );

    let body = sent(
        &gpt("gpt-5.2"),
        GenerationOptions::default().cache(CacheRetention::Short),
    )
    .expect("in-memory retention");
    assert_eq!(body["prompt_cache_retention"], "in_memory");
    let body = sent(
        &gpt("gpt-5.2"),
        GenerationOptions::default().cache(CacheRetention::Long),
    )
    .expect("extended retention");
    assert_eq!(body["prompt_cache_retention"], "24h");
    let body = sent(
        &gpt("gpt-5.6"),
        GenerationOptions::default().cache(CacheRetention::None),
    )
    .expect("explicit mode with no breakpoints");
    assert_eq!(body["prompt_cache_options"], json!({"mode": "explicit"}));
    assert_eq!(
        refused(sent(
            &gpt("gpt-5.6"),
            GenerationOptions::default().cache(CacheRetention::Long)
        )),
        Some("cache")
    );
    assert_eq!(
        refused(sent(
            &gpt("gpt-5.5"),
            GenerationOptions::default().cache(CacheRetention::Short)
        )),
        Some("cache")
    );

    let body = sent(
        &gpt("gpt-5.2"),
        GenerationOptions::default()
            .service_tier(ServiceTier::Flex)
            .verbosity(Verbosity::High)
            .parallel_tool_calls(true)
            .top_p(0.3)
            .seed(4)
            .stop(["a", "b"]),
    )
    .expect("every other option");
    assert_eq!(body["service_tier"], "flex");
    assert_eq!(body["verbosity"], "high");
    assert_eq!(body["parallel_tool_calls"], true);
    assert_eq!(body["top_p"], 0.3);
    assert_eq!(body["seed"], 4);
    assert_eq!(body["stop"], json!(["a", "b"]));
    assert_eq!(
        refused(sent(
            &gpt("gpt-5.2"),
            GenerationOptions::default().stop(["a", "b", "c", "d", "e"])
        )),
        Some("stop")
    );
}

#[test]
fn azure_refuses_what_its_default_api_version_predates() {
    let azure = chat(&AZURE, "prod-deployment");
    assert_eq!(
        refused(sent(
            &azure,
            GenerationOptions::default().reasoning(Effort::High)
        )),
        Some("reasoning")
    );
    assert_eq!(
        refused(sent(
            &azure,
            GenerationOptions::default().service_tier(ServiceTier::Auto)
        )),
        Some("service_tier")
    );
    let mut later = azure.clone();
    later.provider.api_version = Some("2025-04-01-preview".to_owned());
    let body = sent(&later, GenerationOptions::default().reasoning(Effort::High))
        .expect("a later api-version takes reasoning");
    assert_eq!(body["reasoning_effort"], "high");
    assert_eq!(
        refused(sent(
            &later,
            GenerationOptions::default().reasoning(Effort::Max)
        )),
        Some("reasoning")
    );
}

#[test]
fn dialect_cells() {
    let body = sent(
        &chat(&OPENROUTER, "openai/gpt-5.2"),
        GenerationOptions::default()
            .reasoning(Reasoning::Off)
            .cache(CacheRetention::Short)
            .service_tier(ServiceTier::Auto),
    )
    .expect("OpenRouter options");
    assert_eq!(body["reasoning"], json!({"effort": "none"}));
    assert!(
        body.get("cache_control").is_none(),
        "an automatic upstream: {body}"
    );
    assert!(body.get("service_tier").is_none(), "{body}");
    let body = sent(
        &chat(&OPENROUTER, "anthropic/claude-sonnet-4.5"),
        GenerationOptions::default()
            .reasoning(Reasoning::Budget { tokens: 2048 })
            .cache(CacheRetention::Long),
    )
    .expect("an Anthropic upstream takes a budget and a marker");
    assert_eq!(body["reasoning"], json!({"max_tokens": 2048}));
    assert_eq!(
        body["cache_control"],
        json!({"type": "ephemeral", "ttl": "1h"})
    );

    let body = sent(
        &chat(&DEEPSEEK, "deepseek-v4-flash"),
        GenerationOptions::default().reasoning(Reasoning::Off),
    )
    .expect("DeepSeek turns thinking off");
    assert_eq!(body["thinking"], json!({"type": "disabled"}));
    assert_eq!(
        refused(sent(
            &chat(&DEEPSEEK, "deepseek-v4-flash"),
            GenerationOptions::default().reasoning(Effort::Medium)
        )),
        Some("reasoning")
    );

    let body = sent(
        &chat(&MISTRAL, "mistral-large"),
        GenerationOptions::default().seed(3),
    )
    .expect("Mistral's seed");
    assert_eq!(body["random_seed"], 3);
    assert!(body.get("seed").is_none(), "{body}");

    let body = sent(
        &chat(&GROQ, "qwen/qwen3.8-27b"),
        GenerationOptions::default().service_tier(ServiceTier::Priority),
    )
    .expect("Groq's tiers");
    assert_eq!(body["service_tier"], "performance");
    assert_eq!(
        refused(sent(
            &chat(&GROQ, "openai/gpt-oss-120b"),
            GenerationOptions::default().reasoning(Reasoning::Off)
        )),
        Some("reasoning")
    );

    let body = sent(
        &chat(&VENICE, "venice-uncensored"),
        GenerationOptions::default().cache(CacheRetention::Long),
    )
    .expect("Venice's retention");
    assert_eq!(body["prompt_cache_retention"], "24h");

    let body = sent(
        &chat(&MOONSHOT, "kimi-k2.6"),
        GenerationOptions::default()
            .reasoning(Reasoning::Off)
            .cache(CacheRetention::Short),
    )
    .expect("Kimi K2.6");
    assert_eq!(body["thinking"], json!({"type": "disabled"}));
    assert_eq!(
        body["prompt_cache_options"],
        json!({"mode": "implicit", "ttl": "5m"})
    );

    let body = sent(
        &chat(&OLLAMA, "qwen3:4b"),
        GenerationOptions::default().reasoning(Effort::Low),
    )
    .expect("Ollama's /v1 effort");
    assert_eq!(body["reasoning_effort"], "low");
}

#[test]
fn a_gateway_rig_does_not_know_refuses_every_option() {
    const GATEWAY: Dialect = Dialect::gateway("mygw", "https://gateway.example/v1", "MYGW_KEY");
    let gateway = chat(&GATEWAY, "model");
    assert_eq!(
        refused(sent(&gateway, GenerationOptions::default().top_p(0.5))),
        Some("top_p")
    );
    let body = sent(&gateway, GenerationOptions::default()).expect("no option set");
    assert!(body.get("top_p").is_none(), "{body}");
}
