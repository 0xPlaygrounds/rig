//! Each provider's configuration serializes to the JSON #2600's
//! configuration of the same name did, written out here, and reads back
//! from it. The credential is never written.

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

use crate::providers::{anthropic, cohere, gemini, ollama, openai, voyageai};

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
fn voyageai_config_json_is_unchanged() {
    round_trips(
        &voyageai::VoyageAiConfig::new("sk-secret"),
        json!({"api_key": "[redacted]", "base_url": "https://api.voyageai.com/v1"}),
    );
}

/// One signed-in request, served through a recording transport whose every
/// reply is `reply`: what the sign-in and the model sent.
async fn signed_in_requests(
    reference: &str,
    credentials: impl crate::client::signed_in::CredentialSource,
    reply: &'static str,
) -> Vec<crate::test_utils::CapturedHttpRequest> {
    use crate::serve::{Dispatch, ErasedHandler};
    let spec = crate::catalog::Catalog::builtin()
        .resolve(reference)
        .expect("in the catalog")
        .spec
        .clone();
    let http = crate::test_utils::RecordingHttpClient::new(reply);
    let model = crate::client::signed_in::SignedInModel::new(
        spec,
        credentials,
        crate::http_client::DynHttpClient::new(http.clone()),
    )
    .expect("connects");
    let kind = crate::effect::EffectKind::Completion {
        request: crate::completion::CompletionRequest::new("hi").into(),
        stream: false,
    };
    // The reply need not decode: the requests are the evidence.
    let _ = ErasedHandler::new(model)
        .handle(
            kind,
            Dispatch::new(crate::effect::EffectId::from_raw(1), false),
        )
        .await
        .into_outcome()
        .await;
    http.requests()
}

/// A ChatGPT sign-in sends the token and the account it resolves.
#[tokio::test]
async fn a_chatgpt_sign_in_sends_the_token_and_account() {
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
    let requests = signed_in_requests("chatgpt/gpt-6.1-sol", authenticator, "{}").await;
    let request = requests.first().expect("the model sent its request");
    assert_eq!(request.headers["authorization"], "Bearer chatgpt-token");
    assert_eq!(request.headers["chatgpt-account-id"], "acct_1");
}

/// A Copilot sign-in exchanges the GitHub token through the model's own
/// transport, then sends the session token to the API root the exchange
/// names.
#[tokio::test]
async fn a_copilot_sign_in_sends_to_the_api_root_its_exchange_names() {
    use crate::providers::copilot;
    let authenticator = copilot::auth::Authenticator::new(
        copilot::auth::AuthSource::GitHubAccessToken("gho_token".into()),
        copilot::auth::MemoryTokenStore::default(),
        copilot::auth::DeviceCodeHandler::default(),
        false,
    );
    let exchange = r#"{"token":"tid=session","expires_at":4102444800,"endpoints":{"api":"https://api.enterprise.githubcopilot.com"}}"#;
    let requests = signed_in_requests("copilot/gpt-4.1", authenticator, exchange).await;
    let uris: Vec<&str> = requests
        .iter()
        .map(|request| request.uri.as_str())
        .collect();
    assert_eq!(
        uris,
        [
            "https://api.github.com/copilot_internal/v2/token",
            "https://api.enterprise.githubcopilot.com/chat/completions",
        ]
    );
    assert_eq!(requests[1].headers["authorization"], "Bearer tid=session");
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
        .resolve("chatgpt/gpt-6.1-sol")
        .expect("in the catalog")
        .spec
        .clone();
    let http = crate::test_utils::RecordingHttpClient::new("{}");
    let model = crate::client::signed_in::SignedInModel::new(
        spec,
        authenticator,
        crate::http_client::DynHttpClient::new(http.clone()),
    )
    .expect("connects");
    let kind = crate::effect::EffectKind::Completion {
        request: crate::completion::CompletionRequest::new("hi")
            .top_p(0.9)
            .into(),
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
