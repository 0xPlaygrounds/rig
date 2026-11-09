//! Live Copilot coverage for combining `prompt_typed()` with tool calling.

use anyhow::Result;

use crate::copilot::{live_responses_model, with_copilot_cassette_result};
use crate::support::assert_weather_tool_roundtrip_response;
use crate::weather_tool::{WeatherResponse, WeatherTool};

#[tokio::test]
async fn prompt_typed_with_tool_call_roundtrip() -> Result<()> {
    with_copilot_cassette_result("typed_prompt_tools/prompt_typed_with_tool_call_roundtrip", |client| async move {
        let weather = WeatherTool::default();
        let agent = rig::AgentBuilder::new(client.completion(live_responses_model()))
            .preamble(
                "You are a helpful assistant. When asked about weather, use the weather tool to get the current conditions. \
                 After calling the tool, return a JSON response with the city name and the weather description. \
                 DO NOT modify the description from the tool result.",
            )
            .tool(weather.clone())
            .default_max_turns(2)
            .build();

        let response: WeatherResponse = agent
            .prompt_typed("Hello, what's the weather in London?")
            .await?
            .output;

        anyhow::ensure!(
            weather.calls() >= 1,
            "expected the weather tool to be executed at least once"
        );
        assert_weather_tool_roundtrip_response(&response.city, &response.weather, "London");

        Ok(())
    }).await
}
