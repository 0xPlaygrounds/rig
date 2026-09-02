//! Smoke coverage for issue #1604 against a local llama.cpp OpenAI-compatible server.

use anyhow::Result;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rig::agent::{
    AgentHook, CompletionCallAction, CompletionCallEvent, CompletionResponseEvent,
    ObservationAction, ToolCall as ToolCallEvent, ToolCallAction, ToolResultAction,
    ToolResultEvent,
};
use rig::prelude::*;
use rig::tool::Tool;

use super::super::cassette_support::*;
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

#[derive(Clone, Default)]
struct StepLogger {
    completion_calls: Arc<AtomicUsize>,
    tool_calls: Arc<AtomicUsize>,
}

impl StepLogger {
    fn next_completion_call(&self) -> usize {
        self.completion_calls.fetch_add(1, Ordering::SeqCst) + 1
    }

    fn current_completion_call(&self) -> usize {
        self.completion_calls.load(Ordering::SeqCst)
    }

    fn next_tool_call(&self) -> usize {
        self.tool_calls.fetch_add(1, Ordering::SeqCst) + 1
    }
}

impl AgentHook for StepLogger {
    async fn on_completion_call(
        &self,
        _ctx: &rig::agent::HookContext,
        event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        let call_no = self.next_completion_call();
        println!("\n=== completion call #{call_no}: model input ===");
        println!("history:\n{}", pretty_json(event.history));
        println!("prompt:\n{}", pretty_json(event.prompt));
        CompletionCallAction::continue_run()
    }

    async fn on_completion_response(
        &self,
        _ctx: &rig::agent::HookContext,
        event: CompletionResponseEvent<'_>,
    ) -> ObservationAction {
        let call_no = self.current_completion_call();
        println!("\n=== completion response #{call_no}: normalized choice ===");
        println!("{}", pretty_json(event.content));
        println!("usage: {:?}", event.usage);
        println!("message_id: {:?}", event.identity.message_id);
        ObservationAction::continue_run()
    }

    async fn on_tool_call(
        &self,
        _ctx: &rig::agent::HookContext,
        event: ToolCallEvent<'_>,
    ) -> ToolCallAction {
        let tool_no = self.next_tool_call();
        println!("\n=== tool call #{tool_no}: model requested tool ===");
        println!("tool_name: {}", event.tool_name);
        println!("tool_call_id: {:?}", event.tool_call_id);
        println!("block_id: {}", event.block_id);
        println!("args: {}", event.args);
        ToolCallAction::run()
    }

    async fn on_tool_result(
        &self,
        _ctx: &rig::agent::HookContext,
        event: ToolResultEvent<'_>,
    ) -> ToolResultAction {
        println!("\n=== tool result: tool returned ===");
        println!("tool_name: {}", event.tool_name);
        println!("tool_call_id: {:?}", event.tool_call_id);
        println!("block_id: {}", event.block_id);
        println!("args: {}", event.args);
        println!("result: {}", event.presentation.render());
        ToolResultAction::keep()
    }
}

fn pretty_json<T>(value: &T) -> String
where
    T: Serialize + ?Sized,
{
    serde_json::to_string_pretty(value)
        .unwrap_or_else(|err| format!("<failed to serialize debug payload as JSON: {err}>"))
}

impl Tool for WeatherTool {
    const NAME: &'static str = "weather";
    type Error = std::io::Error;
    type Args = WeatherArgs;
    type Output = String;

    fn description(&self) -> String {
        "Get the current weather for a city".to_string()
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
    ) -> impl std::future::Future<Output = Result<Self::Output, Self::Error>> + Send {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        let result = format!("The weather in {} is all fire and brimstone", args.city);

        println!("\n=== weather tool implementation ===");
        println!(
            "{}",
            pretty_json(&serde_json::json!({
                "args": { "city": args.city },
                "returned": result,
            }))
        );

        std::future::ready(Ok(result))
    }
}

#[tokio::test]
async fn prompt_typed_with_tool_call_verbatim_roundtrip() -> Result<()> {
    with_llamacpp_cassette_result("typed_prompt_tools/prompt_typed_with_tool_call_verbatim_roundtrip", |client| async move {
        let model = CASSETTE_MODEL;
        let hook = StepLogger::default();

        let call_count = Arc::new(AtomicUsize::new(0));

        let agent = client
            .agent(model)
            .tool(WeatherTool::new(call_count.clone()))
            .preamble(
                "You are a helpful assistant. When asked about weather, use the weather tool to get the current conditions. After calling the tool, return a JSON response with the city name and the weather description. DO NOT modify the description from the tool result.",
            )
            .build();

        // The tool call needs a second turn to become an answer; the default
        // is one, which is why this cell had never passed.
        let result = agent
            .prompt_typed::<WeatherResponse>("Hello, whats the weather in London?")
            .add_hook(hook)
            .max_turns(4)
            .await;

        println!("prompt_typed result: {result:#?}");

        let response = result?;
        println!("agent response: {response:#?}");

        anyhow::ensure!(
            call_count.load(Ordering::SeqCst) >= 1,
            "expected the weather tool to be executed at least once"
        );
        crate::support::assert_weather_tool_roundtrip_response(
            &response.output.city,
            &response.output.weather,
            "London",
        );

        Ok(())
    })
    .await
}

#[tokio::test]
async fn prompt_typed_with_tool_call_roundtrip() -> Result<()> {
    with_llamacpp_cassette_result("typed_prompt_tools/prompt_typed_with_tool_call_roundtrip", |client| async move {

        let call_count = Arc::new(AtomicUsize::new(0));
        let agent = client
            .agent(CASSETTE_MODEL)
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

        // The tool call needs a second turn to become an answer; the default
        // is one, which is why this cell had never passed.
        let response: WeatherResponse = agent
            .prompt_typed("Hello, whats the weather in London?")
            .max_turns(4)
            .await?.output;

        anyhow::ensure!(
            call_count.load(Ordering::SeqCst) >= 1,
            "expected the weather tool to be executed at least once"
        );
        assert_weather_tool_roundtrip_response(&response.city, &response.weather, "London");

        Ok(())
    })
    .await
}
