//! Live Copilot coverage for combining `prompt_typed()` with tool calling.

use anyhow::Result;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rig_core::client::CompletionClient;
use rig_core::completion::{ToolDefinition, TypedPrompt};
use rig_core::tool::Tool;

use crate::copilot::{live_client, live_responses_model};
use crate::support::assert_weather_tool_roundtrip_response;

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct WeatherResponse {
    city: String,
    weather: String,
}

#[derive(Debug, Deserialize)]
struct WeatherArgs {
    city: String,
}

#[derive(Clone)]
struct WeatherTool {
    call_count: Arc<AtomicUsize>,
}

impl WeatherTool {
    fn new(call_count: Arc<AtomicUsize>) -> Self {
        Self { call_count }
    }
}

impl Tool for WeatherTool {
    const NAME: &'static str = "weather";

    type Error = std::io::Error;
    type Args = WeatherArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Get the current weather for a city.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                },
                "required": ["city"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(format!(
            "The weather in {} is all fire and brimstone",
            args.city
        ))
    }
}

#[tokio::test]
#[ignore = "requires Copilot credentials or existing OAuth cache"]
async fn prompt_typed_with_tool_call_roundtrip() -> Result<()> {
    let call_count = Arc::new(AtomicUsize::new(0));
    let agent = live_client()
        .agent(live_responses_model())
        .preamble(
            "You are a helpful assistant. When asked about weather, use the weather tool to get the current conditions. \
             After calling the tool, return a JSON response with the city name and the weather description. \
             DO NOT modify the description from the tool result.",
        )
        .tool(WeatherTool::new(call_count.clone()))
        .build();

    let response: WeatherResponse = agent
        .prompt_typed("Hello, what's the weather in London?")
        .await?;

    anyhow::ensure!(
        call_count.load(Ordering::SeqCst) >= 1,
        "expected the weather tool to be executed at least once"
    );
    assert_weather_tool_roundtrip_response(&response.city, &response.weather, "London");

    Ok(())
}
