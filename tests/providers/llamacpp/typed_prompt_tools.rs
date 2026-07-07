//! Smoke coverage for issue #1604 against a local llama.cpp OpenAI-compatible server.

use anyhow::Result;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rig::agent::{AgentHook, Flow, StepEvent};
use rig::client::CompletionClient;
use rig::completion::{CompletionModel, ToolDefinition, TypedPrompt};
use rig::tool::Tool;

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

impl<M> AgentHook<M> for StepLogger
where
    M: CompletionModel,
    M::Response: Serialize,
{
    async fn on_event(&self, _ctx: &rig::agent::HookContext, event: StepEvent<'_, M>) -> Flow {
        match event {
            StepEvent::CompletionCall {
                prompt, history, ..
            } => {
                let call_no = self.next_completion_call();

                println!("\n=== completion call #{call_no}: model input ===");
                println!("history:\n{}", pretty_json(history));
                println!("prompt:\n{}", pretty_json(prompt));

                Flow::cont()
            }
            StepEvent::CompletionResponse { response, .. } => {
                let call_no = self.current_completion_call();

                println!("\n=== completion response #{call_no}: normalized choice ===");
                println!("{}", pretty_json(&response.choice));
                println!("\n=== completion response #{call_no}: raw provider payload ===");
                println!("{}", pretty_json(&response.raw_response));

                Flow::cont()
            }
            StepEvent::ToolCall {
                tool_name,
                tool_call_id,
                internal_call_id,
                args,
            } => {
                let tool_no = self.next_tool_call();

                println!("\n=== tool call #{tool_no}: model requested tool ===");
                println!("tool_name: {tool_name}");
                println!("tool_call_id: {tool_call_id:?}");
                println!("internal_call_id: {internal_call_id}");
                println!("args: {args}");

                Flow::cont()
            }
            StepEvent::ToolResult {
                tool_name,
                tool_call_id,
                internal_call_id,
                args,
                result,
                ..
            } => {
                println!("\n=== tool result: tool returned ===");
                println!("tool_name: {tool_name}");
                println!("tool_call_id: {tool_call_id:?}");
                println!("internal_call_id: {internal_call_id}");
                println!("args: {args}");
                println!("result: {result}");

                Flow::cont()
            }
            _ => Flow::cont(),
        }
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

    fn definition(&self) -> impl std::future::Future<Output = ToolDefinition> + Send + Sync {
        std::future::ready(ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Get the current weather for a city".to_string(),
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
#[ignore = "requires a local llama.cpp OpenAI-compatible server"]
async fn prompt_typed_with_tool_call_verbatim_roundtrip() -> Result<()> {
    let model = support::model_name();
    let hook = StepLogger::default();

    let call_count = Arc::new(AtomicUsize::new(0));
    let client = support::completions_client();

    let agent = client
        .agent(model)
        .tool(WeatherTool::new(call_count.clone()))
        .preamble(
            "You are a helpful assistant. When asked about weather, use the weather tool to get the current conditions. After calling the tool, return a JSON response with the city name and the weather description. DO NOT modify the description from the tool result.",
        )
        .build();

    let result = agent
        .prompt_typed::<WeatherResponse>("Hello, whats the weather in London?")
        .add_hook(hook)
        .await;

    println!("prompt_typed result: {result:#?}");

    let response = result?;
    println!("agent response: {response:#?}");

    anyhow::ensure!(
        call_count.load(Ordering::SeqCst) >= 1,
        "expected the weather tool to be executed at least once"
    );
    crate::support::assert_weather_tool_roundtrip_response(
        &response.city,
        &response.weather,
        "London",
    );

    Ok(())
}
