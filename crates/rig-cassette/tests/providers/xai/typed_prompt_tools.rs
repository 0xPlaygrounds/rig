//! xAI live coverage for combining `prompt_typed()` with tool calling.

use anyhow::Result;
use rig::providers::xai;

use super::support::with_xai_cassette_result;
use crate::support::assert_weather_tool_roundtrip_response;
use crate::weather_tool::{WeatherResponse, WeatherTool};

#[tokio::test]
async fn prompt_typed_with_tool_call_roundtrip() -> Result<()> {
    with_xai_cassette_result(
        "typed_prompt_tools/prompt_typed_with_tool_call_roundtrip",
        |client| async move {
            let weather = WeatherTool::default();
            let agent = rig::AgentBuilder::new(client.completion(xai::GROK_4))
                .preamble(
                    "You are a helpful assistant. When asked about weather, use the weather tool to get the current conditions. \
                     After calling the tool, respond with ONLY minified JSON matching this schema: \
                     {\"city\": string, \"weather\": string}. \
                     DO NOT wrap the JSON in markdown or add explanatory text. \
                     DO NOT modify the weather description from the tool result.",
                )
                .tool(weather.clone())
                .default_max_turns(2)
                .additional_params(serde_json::json!({ "store": false }))
                .build();

            let response: WeatherResponse = agent
                .prompt_typed("Hello, whats the weather in London?")
                .await?
                .output;

            anyhow::ensure!(
                weather.calls() >= 1,
                "expected the weather tool to be executed at least once"
            );
            assert_weather_tool_roundtrip_response(&response.city, &response.weather, "London");

            Ok(())
        },
    )
    .await
}
