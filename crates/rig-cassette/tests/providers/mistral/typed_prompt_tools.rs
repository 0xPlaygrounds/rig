//! Mistral live coverage for combining `prompt_typed()` with tool calling.

use anyhow::Result;
use rig::providers::openai::wire::MISTRAL;
use rig_test_support::cassette_models::OpenAiModels;

use crate::support::assert_weather_tool_roundtrip_response;
use crate::weather_tool::{WeatherResponse, WeatherTool};

use super::TOOL_MODEL;

#[tokio::test]
#[ignore = "requires MISTRAL_API_KEY"]
async fn prompt_typed_with_tool_call_roundtrip() -> Result<()> {
    let weather = WeatherTool::default();
    let client = OpenAiModels::from_env_for(&MISTRAL).expect("MISTRAL_API_KEY should be set");
    let agent = rig::AgentBuilder::new(client.completion(TOOL_MODEL))
        .preamble(
            "You are a helpful assistant. When asked about weather, use the weather tool to get the current conditions. \
             After calling the tool, respond with ONLY minified JSON matching this schema: \
             {\"city\": string, \"weather\": string}. \
             DO NOT wrap the JSON in markdown or add explanatory text. \
             DO NOT modify the weather description from the tool result.",
        )
        .max_tokens(256)
        .tool(weather.clone())
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
}
