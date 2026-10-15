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

/// A token exchange whose recorded reply names an enterprise API root.
const EXCHANGE: &str = r#"{"token":"tid=session","expires_at":4102444800,"endpoints":{"api":"https://api.enterprise.githubcopilot.com"}}"#;

fn github_token_sign_in() -> copilot::auth::Authenticator {
    copilot::auth::Authenticator::new(
        copilot::auth::AuthSource::GitHubAccessToken("gho_token".into()),
        copilot::auth::MemoryTokenStore::default(),
        copilot::auth::DeviceCodeHandler::default(),
        false,
    )
}

/// Copilot's token exchange sends through the client's own transport, and
/// the client takes the session token and the API root the exchange names.
#[tokio::test]
async fn copilot_signs_in_through_its_own_transport() {
    let http = crate::test_utils::RecordingHttpClient::new(EXCHANGE);
    let copilot = copilot::CopilotConfig::new("")
        .connect(http.clone())
        .authenticate(&github_token_sign_in())
        .await
        .expect("the exchange succeeds");

    let requests = http.requests();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests.first().map(|request| request.uri.as_str()),
        Some("https://api.github.com/copilot_internal/v2/token")
    );
    assert_eq!(copilot.config().api_key.expose(), "tid=session");
    assert_eq!(
        copilot.config().base_url,
        "https://api.enterprise.githubcopilot.com"
    );
}

/// An API root set on the client before signing in is kept.
#[tokio::test]
async fn copilot_sign_in_keeps_an_explicit_api_root() {
    let copilot = copilot::CopilotConfig::new("")
        .with_base_url("http://localhost:4141")
        .connect(crate::test_utils::RecordingHttpClient::new(EXCHANGE))
        .authenticate(&github_token_sign_in())
        .await
        .expect("the exchange succeeds");
    assert_eq!(copilot.config().api_key.expose(), "tid=session");
    assert_eq!(copilot.config().base_url, "http://localhost:4141");
}

/// ChatGPT's sign-in keeps the configuration and takes the token and the
/// account it resolves.
#[tokio::test]
async fn chatgpt_sign_in_sets_the_token_and_account() {
    use crate::providers::chatgpt;
    let authenticator = chatgpt::auth::Authenticator::new(
        chatgpt::auth::AuthSource::AccessToken {
            access_token: "chatgpt-token".into(),
            account_id: Some("acct_1".into()),
        },
        chatgpt::auth::MemoryTokenStore::default(),
        chatgpt::auth::DeviceCodeHandler::default(),
        false,
    );
    let chatgpt = openai::OpenAIConfig::with_key(&chatgpt::DIALECT, "")
        .with_base_url("http://localhost:1/backend-api/codex")
        .connect(crate::test_utils::RecordingHttpClient::new("{}"))
        .authenticate(&authenticator)
        .await
        .expect("an access token needs no exchange");
    assert_eq!(chatgpt.config().api_key.expose(), "chatgpt-token");
    assert_eq!(chatgpt.config().account_id.as_deref(), Some("acct_1"));
    assert_eq!(
        chatgpt.config().base_url,
        "http://localhost:1/backend-api/codex"
    );
}

/// The ChatGPT sign-in model says what its model leaves out without
/// signing in: a check is a dry run, so no credential is fetched and
/// nothing is sent.
#[test]
fn a_signed_in_model_checks_without_a_credential() {
    use crate::providers::chatgpt;
    use crate::serve::ErasedHandler;
    let authenticator = chatgpt::auth::Authenticator::new(
        chatgpt::auth::AuthSource::OAuth,
        chatgpt::auth::MemoryTokenStore::default(),
        chatgpt::auth::DeviceCodeHandler::default(),
        false,
    );
    let spec = crate::catalog::Catalog::builtin()
        .resolve("chatgpt/gpt-5.3-codex-spark")
        .expect("in the catalog")
        .spec
        .clone();
    let http = crate::test_utils::RecordingHttpClient::new("{}");
    let model = chatgpt::SignedInModel::new(
        spec,
        authenticator,
        crate::http_client::DynHttpClient::new(http.clone()),
    )
    .expect("connects");
    let kind = crate::effect::EffectKind::Completion {
        request: crate::completion::CompletionRequest::new("hi").top_p(0.9),
        stream: true,
    };
    let left_out: Vec<String> = ErasedHandler::new(model)
        .check(&kind)
        .iter()
        .map(|refusal| refusal.option.to_string())
        .collect();
    assert_eq!(left_out, ["top_p"]);
    assert!(http.requests().is_empty(), "a check sends nothing");
}
