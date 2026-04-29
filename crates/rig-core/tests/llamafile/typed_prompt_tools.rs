//! Llamafile coverage for combining `prompt_typed()` with tool calling.

use anyhow::Result;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rig_core::client::CompletionClient;
use rig_core::completion::{ToolDefinition, TypedPrompt};
use rig_core::tool::Tool;

use crate::support::assert_weather_tool_roundtrip_response;

use super::support;

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
#[ignore = "requires a local llamafile server at http://localhost:8080"]
async fn prompt_typed_with_tool_call_roundtrip() -> Result<()> {
    if support::skip_if_server_unavailable() {
        return Ok(());
    }

    let call_count = Arc::new(AtomicUsize::new(0));
    let client = support::client();
    let agent = client
        .agent(support::model_name())
        .preamble(
            "You are a helpful assistant. When asked about weather, call the `weather` tool exactly once with the requested city. \
             The only valid tool name is `weather`; never invent or call any other tool. \
             After receiving the weather tool result, do not call any more tools. \
             Then respond with ONLY minified JSON matching exactly this schema: \
             {\"city\": string, \"weather\": string}. \
             The `city` field is required and must contain the requested city exactly. \
             The `weather` field is required and must contain the tool result verbatim. \
             For the prompt asking about London, a valid final answer looks like: \
             {\"city\":\"London\",\"weather\":\"The weather in London is all fire and brimstone\"}. \
             DO NOT wrap the JSON in markdown or add explanatory text.",
        )
        .tool(WeatherTool::new(call_count.clone()))
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
}
