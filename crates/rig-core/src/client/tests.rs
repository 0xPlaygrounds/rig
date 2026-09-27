//! Each provider's configuration serializes to the JSON #2600's
//! configuration of the same name did, written out here, and reads back
//! from it. The credential is never written.

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use crate::providers::{anthropic, cohere, copilot, gemini, ollama, openai, voyageai};

fn round_trips<T>(config: &T, expected: Value)
where
    T: Serialize + DeserializeOwned + std::fmt::Debug,
{
    let written = serde_json::to_value(config).expect("the configuration serializes");
    assert_eq!(written, expected);
    let read: T = serde_json::from_value(expected.clone()).expect("the JSON reads back");
    assert_eq!(
        serde_json::to_value(&read).expect("the read configuration serializes"),
        expected
    );
}

#[test]
fn openai_config_json_is_unchanged() {
    round_trips(
        &openai::OpenAIConfig::new("sk-secret"),
        json!({
            "api_key": "[redacted]",
            "base_url": "https://api.openai.com/v1",
            "dialect": "openai",
            "auth": "Bearer",
        }),
    );
}

#[test]
fn anthropic_config_json_is_unchanged() {
    round_trips(
        &anthropic::AnthropicConfig::new("sk-secret"),
        json!({
            "api_key": "[redacted]",
            "base_url": "https://api.anthropic.com",
            "version": "2023-06-01",
            "betas": [],
            "dialect": "anthropic",
        }),
    );
}

#[test]
fn gemini_config_json_is_unchanged() {
    round_trips(
        &gemini::GeminiConfig::new("sk-secret"),
        json!({
            "api_key": "[redacted]",
            "base_url": "https://generativelanguage.googleapis.com",
        }),
    );
}

#[test]
fn cohere_config_json_is_unchanged() {
    round_trips(
        &cohere::CohereConfig::new("sk-secret"),
        json!({"api_key": "[redacted]", "base_url": "https://api.cohere.ai"}),
    );
}

#[test]
fn ollama_config_json_is_unchanged() {
    round_trips(
        &ollama::OllamaConfig::new().with_api_key("sk-secret"),
        json!({"base_url": "http://localhost:11434", "api_key": "[redacted]"}),
    );
}

#[test]
fn copilot_config_json_is_unchanged() {
    round_trips(
        &copilot::CopilotConfig::new("sk-secret"),
        json!({"api_key": "[redacted]", "base_url": "https://api.githubcopilot.com"}),
    );
}

#[test]
fn voyageai_config_json_is_unchanged() {
    round_trips(
        &voyageai::VoyageAiConfig::new("sk-secret"),
        json!({"api_key": "[redacted]", "base_url": "https://api.voyageai.com/v1"}),
    );
}

/// A client keeps its configuration, and a configuration put on a transport
/// comes back unchanged.
#[test]
fn a_client_holds_the_configuration_it_was_built_from() {
    let config = openai::OpenAIConfig::new("sk-secret").with_base_url("http://localhost:1/v1");
    let client = config
        .clone()
        .connect(crate::test_utils::RecordingHttpClient::new("{}"));
    assert_eq!(client.config(), &config);
}
