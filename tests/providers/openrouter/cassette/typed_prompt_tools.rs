//! Cassette-backed OpenRouter coverage for combining `prompt_typed()` with tool calling.

use anyhow::Result;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rig::client::CompletionClient;
use rig::completion::TypedPrompt;
use rig::tool::Tool;

use crate::support::assert_weather_tool_roundtrip_response;

use super::super::{TOOL_MODEL, support::with_openrouter_cassette_result};

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
    type Args = WeatherArgs;
    type Output = String;

    fn description(&self) -> String {
        "Get the current weather for a city.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "city": { "type": "string" }
            },
            "required": ["city"]
        })
    }

    fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> impl std::future::Future<Output = Result<Self::Output, rig::tool::ToolExecutionError>> + Send
    {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        std::future::ready(Ok(format!(
            "The weather in {} is all fire and brimstone",
            args.city
        )))
    }
}

#[tokio::test]
async fn prompt_typed_with_tool_call_roundtrip() -> Result<()> {
    with_openrouter_cassette_result(
        "typed_prompt_tools/prompt_typed_with_tool_call_roundtrip",
        |client| async move {
            let call_count = Arc::new(AtomicUsize::new(0));
            let agent = client
                .agent(TOOL_MODEL)
                .preamble(
                    "You are a helpful assistant. When asked about weather, use the weather tool to get the current conditions. \
                     After calling the tool, respond with ONLY minified JSON matching this schema: \
                     {\"city\": string, \"weather\": string}. \
                     DO NOT wrap the JSON in markdown or add explanatory text. \
                     DO NOT modify the weather description from the tool result.",
                )
                .tool(WeatherTool::new(call_count.clone()))
                .default_max_turns(2)
                .build();

            let response: WeatherResponse = agent
                .prompt_typed("Hello, whats the weather in London?")
                .await?;

            anyhow::ensure!(
                call_count.load(Ordering::SeqCst) >= 1,
                "expected the weather tool to be executed at least once"
            );
            assert_weather_tool_roundtrip_response(&response.city, &response.weather, "London");

            Ok(())
        },
    )
    .await
}
