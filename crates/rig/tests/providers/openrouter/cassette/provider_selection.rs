//! Cassette-backed OpenRouter provider selection scenarios.

use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::openrouter::{MaxPrice, ProviderPreferences, ProviderSortStrategy};

use crate::support::assert_nonempty_response;

use super::super::support::with_openrouter_cassette;

const DEEPSEEK_V3_2: &str = "deepseek/deepseek-v3.2";

#[tokio::test]
async fn provider_selection_scenarios() {
    with_openrouter_cassette(
        "provider_selection/provider_selection_scenarios",
        |client| async move {
            let scenarios = [
                (
                    "hello",
                    ProviderPreferences::new()
                        .order(["DeepInfra", "DeepSeek", "Chutes"])
                        .allow_fallbacks(true)
                        .to_json(),
                ),
                (
                    "planet",
                    ProviderPreferences::new()
                        .ignore(["Google Vertex"])
                        .to_json(),
                ),
                (
                    "french hello",
                    ProviderPreferences::new()
                        .sort(ProviderSortStrategy::Latency)
                        .to_json(),
                ),
                (
                    "sky color",
                    ProviderPreferences::new()
                        .require_parameters(true)
                        .to_json(),
                ),
                (
                    "country",
                    ProviderPreferences::new()
                        .max_price(MaxPrice::new().prompt(0.30).completion(0.50))
                        .to_json(),
                ),
            ];

            for (prompt, params) in scenarios {
                let agent = client
                    .agent(DEEPSEEK_V3_2)
                    .preamble("You are a helpful assistant.")
                    .additional_params(params)
                    .build();
                let response = agent.prompt(prompt).await.expect("prompt should succeed");
                assert_nonempty_response(&response);
            }
        },
    )
    .await;
}
