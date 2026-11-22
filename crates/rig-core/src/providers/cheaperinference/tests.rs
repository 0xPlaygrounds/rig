use anyhow::{Result, anyhow, ensure};
use serde_json::json;

use crate::completion::CompletionRequest;
use crate::providers::openai::{OpenAIConfig, wire::CHEAPERINFERENCE};
use crate::providers::registry::{ProviderConfig, ProviderId};
use crate::test_utils::json_body;
use crate::wire::{Mode, Wire};

/// Endpoint and authentication are preset configuration, so these assertions
/// exercise request construction without inventing a provider response cassette.
#[test]
fn chat_requests_use_the_gateway_endpoint_and_bearer_key() -> Result<()> {
    let model = OpenAIConfig::with_key(&CHEAPERINFERENCE, "ci-test").chat("deepseek-v4-flash");
    for mode in [Mode::Unary, Mode::Streaming] {
        let encoded = model.encode(CompletionRequest::new("hello"), mode)?;
        ensure!(encoded.request.uri() == "https://api.cheaperinference.com/v1/chat/completions");
        ensure!(
            encoded
                .request
                .headers()
                .get(http::header::AUTHORIZATION)
                .map(|value| value.to_str())
                .transpose()?
                == Some("Bearer ci-test")
        );
        let body = json_body(&encoded.request);
        ensure!(body.get("model") == Some(&json!("deepseek-v4-flash")));
        ensure!(
            body.get("stream").and_then(|value| value.as_bool())
                == matches!(mode, Mode::Streaming).then_some(true)
        );
    }
    Ok(())
}

/// Registration and serialization are local contracts, independent of replies
/// from the remote service.
#[test]
fn registered_configurations_reload_without_credentials() -> Result<()> {
    let id = ProviderId::resolve(super::PROVIDER_NAME)?;
    let config = id
        .config("ci-secret")
        .ok_or_else(|| anyhow!("the provider must have a registered configuration"))?;
    let json = serde_json::to_string(&config)?;
    ensure!(!json.contains("ci-secret"));
    let restored: ProviderConfig = serde_json::from_str(&json)?;
    ensure!(restored.id() == Some(id));
    let ProviderConfig::OpenAi(restored) = restored else {
        return Err(anyhow!("Cheaper Inference uses the OpenAI wire format"));
    };
    ensure!(restored.api_key.is_empty());
    ensure!(restored.base_url == "https://api.cheaperinference.com/v1");
    Ok(())
}
