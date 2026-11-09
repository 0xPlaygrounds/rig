//! Cassette-backed Venice typed-output plus tool-call coverage.

use anyhow::Result;

use super::super::{TOOL_MODEL, support::with_venice_cassette_result};
use crate::support::assert_weather_tool_roundtrip_response;
use crate::weather_tool::{WeatherResponse, WeatherTool};

#[tokio::test]
async fn prompt_typed_with_tool_call_roundtrip() -> Result<()> {
    with_venice_cassette_result(
        "typed_prompt_tools/prompt_typed_with_tool_call_roundtrip",
        |client| async move {
            let weather = WeatherTool::default();
            let agent = rig::AgentBuilder::new(client.completion(TOOL_MODEL))
                .preamble(
                    "When asked about weather, use the weather tool. Afterward return only JSON \
                     matching {\"city\": string, \"weather\": string}, preserving the tool result.",
                )
                .tool(weather.clone())
                .default_max_turns(2)
                .build();
            let response: WeatherResponse = agent
                .prompt_typed("What is the weather in London?")
                .await?
                .output;
            anyhow::ensure!(weather.calls() >= 1);
            assert_weather_tool_roundtrip_response(&response.city, &response.weather, "London");
            Ok(())
        },
    )
    .await
}
