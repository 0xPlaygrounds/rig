//! Cassette-backed Doubleword typed-output plus tool-call coverage.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use rig::client::CompletionClient;
use rig::completion::TypedPrompt;
use rig::tool::Tool;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::super::{TOOL_MODEL, support::with_doubleword_cassette_result};
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

impl Tool for WeatherTool {
    const NAME: &'static str = "weather";
    type Error = std::io::Error;
    type Args = WeatherArgs;
    type Output = String;

    fn description(&self) -> String {
        "Get the current weather for a city.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(format!(
            "The weather in {} is all fire and brimstone",
            args.city
        ))
    }
}

#[tokio::test]
async fn prompt_typed_with_tool_call_roundtrip() -> Result<()> {
    with_doubleword_cassette_result(
        "typed_prompt_tools/prompt_typed_with_tool_call_roundtrip",
        |client| async move {
            let call_count = Arc::new(AtomicUsize::new(0));
            let agent = client
                .agent(TOOL_MODEL)
                .preamble(
                    "When asked about weather, use the weather tool. Afterward return only JSON \
                     matching {\"city\": string, \"weather\": string}, preserving the tool result.",
                )
                .tool(WeatherTool {
                    call_count: call_count.clone(),
                })
                .default_max_turns(2)
                .build();
            let response: WeatherResponse =
                agent.prompt_typed("What is the weather in London?").await?;
            anyhow::ensure!(call_count.load(Ordering::SeqCst) >= 1);
            assert_weather_tool_roundtrip_response(&response.city, &response.weather, "London");
            Ok(())
        },
    )
    .await
}
