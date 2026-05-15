//! Live OpenAI coverage for combining `prompt_typed()` with tool calling.

use anyhow::Result;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rig::client::CompletionClient;
use rig::completion::{ToolDefinition, TypedPrompt};
use rig::tool::Tool;

use super::super::support::with_openai_completions_cassette_result;
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

    fn definition(
        &self,
        _prompt: String,
    ) -> impl std::future::Future<Output = ToolDefinition> + Send + Sync {
        std::future::ready(ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Get the current weather for a city.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" }
                },
                "required": ["city"]
            }),
        })
    }

    fn call(
        &self,
        args: Self::Args,
    ) -> impl std::future::Future<Output = Result<Self::Output, Self::Error>> + Send {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        std::future::ready(Ok(format!(
            "The weather in {} is all fire and brimstone",
            args.city
        )))
    }
}

#[tokio::test]
async fn prompt_typed_with_tool_call_roundtrip() -> Result<()> {
    with_openai_completions_cassette_result(
        "typed_prompt_tools/prompt_typed_with_tool_call_roundtrip",
        |client| async move {
            let call_count = Arc::new(AtomicUsize::new(0));
            let agent = client
                .agent(rig::providers::openai::GPT_4O)
                .preamble(
                    "You are a helpful assistant. When asked about weather, use the weather tool to get the current conditions. \
                     After calling the tool, return a JSON response with the city name and the weather description. \
                     DO NOT modify the description from the tool result.",
                )
                .tool(WeatherTool::new(call_count.clone()))
                .build();

            let result = agent
                .prompt_typed("Hello, whats the weather in London?")
                .await;

            println!("prompt_typed result: {result:#?}");

            let response: WeatherResponse = result?;
            println!("agent response: {response:#?}");

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
