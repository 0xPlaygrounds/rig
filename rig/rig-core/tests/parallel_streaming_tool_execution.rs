//! Integration test: parallel tool execution in streaming multi-turn agents.
//!
//! Verifies that `with_tool_concurrency(N)` causes multiple tool calls from a
//! single model response to be dispatched concurrently rather than sequentially.
//!
//! The test uses two deliberately slow tools (100 ms each). With sequential
//! execution the combined wall time would be ~200 ms+; with parallel execution
//! it should be ~100 ms+ (i.e. the max of the individual durations).
//!
//! Run:
//!   source .env && cargo test -p rig-core --test parallel_streaming_tool_execution -- --ignored --nocapture

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use futures::StreamExt;
use rig::agent::MultiTurnStreamItem;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Message;
use rig::completion::request::ToolDefinition;
use rig::streaming::{StreamedAssistantContent, StreamedUserContent, StreamingChat};
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::json;

// ── Slow tools ──────────────────────────────────────────────

const TOOL_DELAY_MS: u64 = 500;

#[derive(Debug, thiserror::Error)]
#[error("Tool error")]
struct ToolError;

#[derive(Deserialize)]
struct CityArgs {
    city: String,
}

/// A weather tool that sleeps to simulate latency, then returns canned data.
struct SlowWeatherTool {
    call_count: Arc<AtomicUsize>,
}

impl Tool for SlowWeatherTool {
    const NAME: &'static str = "get_weather";
    type Error = ToolError;
    type Args = CityArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "get_weather".to_string(),
            description: "Get current weather for a city. Always call this for weather questions."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string", "description": "City name" }
                },
                "required": ["city"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(TOOL_DELAY_MS)).await;
        Ok(format!("{}: 72°F, sunny", args.city))
    }
}

/// A population tool that sleeps to simulate latency, then returns canned data.
struct SlowPopulationTool {
    call_count: Arc<AtomicUsize>,
}

impl Tool for SlowPopulationTool {
    const NAME: &'static str = "get_population";
    type Error = ToolError;
    type Args = CityArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "get_population".to_string(),
            description: "Get the population of a city. Always call this for population questions."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string", "description": "City name" }
                },
                "required": ["city"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(TOOL_DELAY_MS)).await;
        Ok(format!("{}: 13.96 million", args.city))
    }
}

// ── Shared constants ────────────────────────────────────────

const PREAMBLE: &str = "\
You are a helpful assistant with access to get_weather and get_population tools. \
When asked about a city, you MUST call BOTH tools in the SAME response (parallel tool use). \
Never guess data; always use the tools.";

const PROMPT: &str = "\
What is the weather and population of Tokyo? \
Call both get_weather and get_population at the same time.";

// ── Test helpers ────────────────────────────────────────────

struct StreamResult {
    tool_call_names: Vec<String>,
    tool_results: usize,
    final_text: String,
    tool_execution_duration: Duration,
    errors: Vec<String>,
}

async fn run_stream<R: std::fmt::Debug>(
    stream: impl futures::Stream<Item = Result<MultiTurnStreamItem<R>, rig::agent::StreamingError>>
    + Unpin,
) -> StreamResult {
    let mut result = StreamResult {
        tool_call_names: vec![],
        tool_results: 0,
        final_text: String::new(),
        tool_execution_duration: Duration::ZERO,
        errors: vec![],
    };

    futures::pin_mut!(stream);

    let mut first_tool_call_time: Option<Instant> = None;
    let mut last_tool_result_time: Option<Instant> = None;

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(content)) => match content {
                StreamedAssistantContent::ToolCall { ref tool_call, .. } => {
                    if first_tool_call_time.is_none() {
                        first_tool_call_time = Some(Instant::now());
                    }
                    result.tool_call_names.push(tool_call.function.name.clone());
                }
                StreamedAssistantContent::Text(ref text) => {
                    result.final_text.push_str(&text.text);
                }
                _ => {}
            },
            Ok(MultiTurnStreamItem::StreamUserItem(ref user_content)) => match user_content {
                StreamedUserContent::ToolResult { .. } => {
                    result.tool_results += 1;
                    last_tool_result_time = Some(Instant::now());
                }
            },
            Ok(MultiTurnStreamItem::FinalResponse(_)) => {}
            Ok(_) => {}
            Err(ref e) => result.errors.push(e.to_string()),
        }
    }

    if let (Some(start), Some(end)) = (first_tool_call_time, last_tool_result_time) {
        result.tool_execution_duration = end.duration_since(start);
    }

    result
}

// ── Tests ───────────────────────────────────────────────────

/// Parallel execution: both tools should complete in roughly the time of one tool.
#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn test_parallel_streaming_openai() {
    use rig::providers::openai;

    let weather_count = Arc::new(AtomicUsize::new(0));
    let pop_count = Arc::new(AtomicUsize::new(0));

    let client = openai::Client::from_env();
    let agent = client
        .agent(openai::GPT_4O)
        .preamble(PREAMBLE)
        .max_tokens(1024)
        .tool(SlowWeatherTool {
            call_count: weather_count.clone(),
        })
        .tool(SlowPopulationTool {
            call_count: pop_count.clone(),
        })
        .build();

    let stream = agent
        .stream_chat(PROMPT, Vec::<Message>::new())
        .multi_turn(3)
        .with_tool_concurrency(10)
        .await;

    let result = run_stream(stream).await;

    assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
    assert!(
        weather_count.load(Ordering::SeqCst) >= 1,
        "get_weather was not called"
    );
    assert!(
        pop_count.load(Ordering::SeqCst) >= 1,
        "get_population was not called"
    );
    assert!(result.tool_results >= 2, "Expected at least 2 tool results");

    // With parallel execution of 2 tools (500ms each), total should be
    // well under 2x (i.e. < 1000ms). Sequential would be >= 1000ms.
    let max_parallel_time = Duration::from_millis(TOOL_DELAY_MS * 2);
    eprintln!(
        "Tool execution duration: {:?} (threshold: {:?})",
        result.tool_execution_duration, max_parallel_time
    );
    if result.tool_call_names.len() >= 2 {
        assert!(
            result.tool_execution_duration < max_parallel_time,
            "Tools took {:?}, expected < {:?} for parallel execution",
            result.tool_execution_duration,
            max_parallel_time
        );
    }
}

/// Sequential execution (default): tools should take roughly the sum of individual times.
#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn test_sequential_streaming_openai() {
    use rig::providers::openai;

    let weather_count = Arc::new(AtomicUsize::new(0));
    let pop_count = Arc::new(AtomicUsize::new(0));

    let client = openai::Client::from_env();
    let agent = client
        .agent(openai::GPT_4O)
        .preamble(PREAMBLE)
        .max_tokens(1024)
        .tool(SlowWeatherTool {
            call_count: weather_count.clone(),
        })
        .tool(SlowPopulationTool {
            call_count: pop_count.clone(),
        })
        .build();

    // Default concurrency = 1 (sequential)
    let stream = agent
        .stream_chat(PROMPT, Vec::<Message>::new())
        .multi_turn(3)
        .await;

    let result = run_stream(stream).await;

    assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
    assert!(
        weather_count.load(Ordering::SeqCst) >= 1,
        "get_weather was not called"
    );
    assert!(
        pop_count.load(Ordering::SeqCst) >= 1,
        "get_population was not called"
    );

    // With sequential execution of 2 tools (500ms each), total should be >= 1000ms.
    if result.tool_call_names.len() >= 2 {
        let min_sequential_time = Duration::from_millis(TOOL_DELAY_MS * 2 - 100); // small margin
        eprintln!(
            "Sequential tool execution duration: {:?} (expected >= {:?})",
            result.tool_execution_duration, min_sequential_time
        );
        assert!(
            result.tool_execution_duration >= min_sequential_time,
            "Sequential execution took {:?}, expected >= {:?}",
            result.tool_execution_duration,
            min_sequential_time
        );
    }
}

/// Parallel execution with Anthropic provider.
#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn test_parallel_streaming_anthropic() {
    use rig::providers::anthropic;

    let weather_count = Arc::new(AtomicUsize::new(0));
    let pop_count = Arc::new(AtomicUsize::new(0));

    let client = anthropic::Client::from_env();
    let agent = client
        .agent("claude-sonnet-4-5-20250929")
        .preamble(PREAMBLE)
        .max_tokens(1024)
        .tool(SlowWeatherTool {
            call_count: weather_count.clone(),
        })
        .tool(SlowPopulationTool {
            call_count: pop_count.clone(),
        })
        .build();

    let stream = agent
        .stream_chat(PROMPT, Vec::<Message>::new())
        .multi_turn(3)
        .with_tool_concurrency(10)
        .await;

    let result = run_stream(stream).await;

    assert!(result.errors.is_empty(), "Errors: {:?}", result.errors);
    assert!(
        weather_count.load(Ordering::SeqCst) >= 1,
        "get_weather was not called"
    );
    assert!(
        pop_count.load(Ordering::SeqCst) >= 1,
        "get_population was not called"
    );
    assert!(result.tool_results >= 2, "Expected at least 2 tool results");

    let max_parallel_time = Duration::from_millis(TOOL_DELAY_MS * 2);
    eprintln!(
        "Tool execution duration: {:?} (threshold: {:?})",
        result.tool_execution_duration, max_parallel_time
    );
    if result.tool_call_names.len() >= 2 {
        assert!(
            result.tool_execution_duration < max_parallel_time,
            "Tools took {:?}, expected < {:?} for parallel execution",
            result.tool_execution_duration,
            max_parallel_time
        );
    }
}
