//! Smoke test for OpenRouter model fallback routing.

use rig::OneOrMany;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{CompletionModel, CompletionRequest, Prompt};
use rig::providers::openrouter::{self, ModelFallbacks};

use crate::support::assert_nonempty_response;

const PRIMARY_MODEL: &str = "openai/gpt-4o-mini";
const FALLBACK_MODEL: &str = "anthropic/claude-3-5-haiku";

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn model_fallbacks_routes_and_reports_response_model() {
    let client = openrouter::Client::from_env().expect("client should build");
    let fallbacks = ModelFallbacks::new([FALLBACK_MODEL]).expect("fallback list should be valid");
    let model = client
        .completion_model(PRIMARY_MODEL)
        .with_model_fallbacks(fallbacks);

    let response = model
        .completion(CompletionRequest {
            model: None,
            preamble: None,
            chat_history: OneOrMany::one("Reply with one word: hello".into()),
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
        })
        .await
        .expect("completion should succeed");

    assert!(
        response.response_model.is_some(),
        "OpenRouter should report the routed model"
    );
    assert_eq!(
        response.response_model.as_deref(),
        Some(response.raw_response.model.as_str())
    );

    let routed = response.response_model.as_deref().unwrap_or_default();
    assert!(
        routed == PRIMARY_MODEL || routed == FALLBACK_MODEL,
        "routed model should be primary or configured fallback, got {routed}"
    );
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn model_fallbacks_agent_extended_details_expose_response_model() {
    let client = openrouter::Client::from_env().expect("client should build");
    let fallbacks = ModelFallbacks::new([FALLBACK_MODEL]).expect("fallback list should be valid");
    let agent = client
        .agent(PRIMARY_MODEL)
        .additional_params(fallbacks.to_json())
        .build();

    let response = agent
        .prompt("Reply with one word: hello")
        .extended_details()
        .await
        .expect("prompt should succeed");

    assert_nonempty_response(&response.output);
    assert!(
        response.response_model.is_some(),
        "agent extended details should expose routed model"
    );
    assert_eq!(
        response.response_model.as_deref(),
        response
            .completion_calls()
            .last()
            .and_then(|call| call.response_model.as_deref())
    );
}
