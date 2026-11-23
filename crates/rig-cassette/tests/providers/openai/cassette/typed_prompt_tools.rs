//! Live OpenAI coverage for combining `prompt_typed()` with tool calling.

use anyhow::Result;

use super::super::support::with_openai_completions_cassette_result;
use crate::support::assert_weather_tool_roundtrip_response;
use crate::weather_tool::{WeatherResponse, WeatherTool};

#[tokio::test]
async fn prompt_typed_with_tool_call_roundtrip() -> Result<()> {
    with_openai_completions_cassette_result(
        "typed_prompt_tools/prompt_typed_with_tool_call_roundtrip",
        |client| async move {
            let weather = WeatherTool::default();
            let agent = rig::AgentBuilder::new(client.completion(rig::providers::openai::GPT_4O))
                .preamble(
                    "You are a helpful assistant. When asked about weather, use the weather tool to get the current conditions. \
                     After calling the tool, return a JSON response with the city name and the weather description. \
                     DO NOT modify the description from the tool result.",
                )
                .tool(weather.clone())
                .default_max_turns(2)
                .build();

            let result = agent
                .prompt_typed("Hello, whats the weather in London?")
                .await;

            println!("prompt_typed result: {result:#?}");

            let response: WeatherResponse = result?.output;
            println!("agent response: {response:#?}");

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
