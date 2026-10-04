//! Cassette-backed OpenRouter provider selection scenarios.

use serde_json::json;

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
                    json!({"provider": {
                        "order": ["DeepInfra", "DeepSeek", "Chutes"],
                        "allow_fallbacks": true,
                    }}),
                ),
                ("planet", json!({"provider": {"ignore": ["Google Vertex"]}})),
                ("french hello", json!({"provider": {"sort": "latency"}})),
                (
                    "sky color",
                    json!({"provider": {"require_parameters": true}}),
                ),
                (
                    "country",
                    json!({"provider": {"max_price": {"prompt": 0.30, "completion": 0.50}}}),
                ),
            ];

            for (prompt, params) in scenarios {
                let agent = rig::AgentBuilder::new(client.completion(DEEPSEEK_V3_2))
                    .preamble("You are a helpful assistant.")
                    .additional_params(params)
                    .build();
                let response = agent.prompt(prompt).await.expect("prompt should succeed");
                assert_nonempty_response(&response.output());
            }
        },
    )
    .await;
}
