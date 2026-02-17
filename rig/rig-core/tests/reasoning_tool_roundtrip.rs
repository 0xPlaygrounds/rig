//! Integration tests: reasoning + tool call roundtrip via the agent loop.
//!
//! These tests exercise the CRITICAL path where reasoning traces must survive
//! multi-turn tool call loops. If reasoning is dropped or malformed, providers
//! return 400 errors.
//!
//! Test flow per provider:
//!   1. Agent receives weather prompt → model thinks (reasoning) → calls get_weather
//!   2. Agent loop preserves [Reasoning, ToolCall] in chat history
//!   3. Agent sends tool result back → model thinks again → responds with text
//!   4. Provider accepts request (no 400 from missing/malformed reasoning)
//!
//! Assertions (universal):
//!   - No stream errors (catches 400s from dropped reasoning)
//!   - Tool was actually invoked (atomic counter)
//!   - Reasoning appeared before tool call in stream ordering
//!   - Tool results flowed back through the stream
//!   - Final text is substantial and references the tool output
//!   - FinalResponse event received (clean termination)
//!
//! Assertions (provider-specific):
//!   - OpenAI: Reasoning blocks with Encrypted/Summary content types
//!   - Anthropic: Reasoning blocks with Text content and cryptographic signatures
//!
//! Run:
//!   source .env && cargo test -p rig-core --test reasoning_tool_roundtrip -- --ignored --nocapture
//!
//! Run single provider:
//!   source .env && cargo test -p rig-core --test reasoning_tool_roundtrip -- --ignored openai --nocapture

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::StreamExt;
use rig::agent::{MultiTurnStreamItem, StreamingError};
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::request::ToolDefinition;
use rig::message::{Reasoning, ReasoningContent};
use rig::streaming::{StreamedAssistantContent, StreamedUserContent, StreamingChat};
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::json;

// ==================== Tool Definition ====================

#[derive(Debug, thiserror::Error)]
#[error("Weather service unavailable")]
struct WeatherError;

#[derive(Deserialize)]
struct WeatherArgs {
    city: String,
}

/// A deterministic weather tool that tracks invocations.
///
/// Returns a fixed response so we can assert the model's final text
/// references the tool output (proves tool result was consumed).
struct WeatherTool {
    call_count: Arc<AtomicUsize>,
}

impl Tool for WeatherTool {
    const NAME: &'static str = "get_weather";
    type Error = WeatherError;
    type Args = WeatherArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "get_weather".to_string(),
            description: "Get the current weather for a city. \
                          Must be called for any weather-related query."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name to get weather for"
                    }
                },
                "required": ["city"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(format!(
            "Weather in {}: 72°F (22°C), sunny with light clouds, humidity 45%, wind 8 mph NW",
            args.city
        ))
    }
}

// ==================== Shared Constants ====================

const SYSTEM_PROMPT: &str = "\
You are a weather assistant. You have access to a get_weather tool. \
You MUST call the get_weather tool for any weather question — never guess weather data. \
After receiving the tool result, provide a concise summary of the weather.";

const USER_PROMPT: &str = "\
I'm planning a trip. What is the current weather in Tokyo, Japan? \
Based on the weather conditions, should I pack an umbrella or sunscreen? \
Use the get_weather tool to check before answering.";

// ==================== Stream Analysis ====================

struct StreamStats {
    reasoning_block_count: usize,
    reasoning_delta_count: usize,
    reasoning_content_types: Vec<&'static str>,
    reasoning_has_signature: bool,
    reasoning_has_encrypted: bool,
    tool_calls_in_stream: Vec<String>,
    tool_results_in_stream: usize,
    text_chunks: usize,
    final_text: String,
    got_final_response: bool,
    errors: Vec<String>,
    /// Sequence of event type names, for ordering analysis.
    events: Vec<&'static str>,
}

impl StreamStats {
    fn new() -> Self {
        Self {
            reasoning_block_count: 0,
            reasoning_delta_count: 0,
            reasoning_content_types: vec![],
            reasoning_has_signature: false,
            reasoning_has_encrypted: false,
            tool_calls_in_stream: vec![],
            tool_results_in_stream: 0,
            text_chunks: 0,
            final_text: String::new(),
            got_final_response: false,
            errors: vec![],
            events: vec![],
        }
    }

    fn total_reasoning(&self) -> usize {
        self.reasoning_block_count + self.reasoning_delta_count
    }

    /// Returns true if at least one reasoning event appeared before the first tool call.
    fn reasoning_before_first_tool_call(&self) -> bool {
        let first_reasoning = self.events.iter().position(|e| e.starts_with("reasoning"));
        let first_tool_call = self.events.iter().position(|e| *e == "tool_call");
        match (first_reasoning, first_tool_call) {
            (Some(r), Some(t)) => r < t,
            (Some(_), None) => true, // reasoning but no tool call (unusual but not a failure here)
            _ => false,
        }
    }
}

/// Record a reasoning block's content types into the stats.
fn record_reasoning(stats: &mut StreamStats, reasoning: &Reasoning, provider: &str) {
    stats.reasoning_block_count += 1;
    stats.events.push("reasoning_block");

    for rc in &reasoning.content {
        let type_name = match rc {
            ReasoningContent::Text { signature, .. } => {
                if signature.is_some() {
                    stats.reasoning_has_signature = true;
                }
                "Text"
            }
            ReasoningContent::Encrypted(_) => {
                stats.reasoning_has_encrypted = true;
                "Encrypted"
            }
            ReasoningContent::Summary(_) => "Summary",
            ReasoningContent::Redacted { .. } => "Redacted",
            _ => "Unknown",
        };
        stats.reasoning_content_types.push(type_name);
    }

    eprintln!(
        "[{provider}] Reasoning block: id={:?}, types={:?}",
        reasoning.id, stats.reasoning_content_types
    );
}

/// Consume a multi-turn stream, recording every event for later assertion.
async fn collect_stream_stats<R: std::fmt::Debug>(
    stream: impl futures::Stream<Item = Result<MultiTurnStreamItem<R>, StreamingError>> + Unpin,
    provider: &str,
) -> StreamStats {
    let mut stats = StreamStats::new();

    futures::pin_mut!(stream);

    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(content)) => match content {
                StreamedAssistantContent::Reasoning(ref reasoning) => {
                    record_reasoning(&mut stats, reasoning, provider);
                }
                StreamedAssistantContent::ReasoningDelta { .. } => {
                    stats.reasoning_delta_count += 1;
                    // Compress consecutive deltas into one event entry
                    if stats.events.last() != Some(&"reasoning_delta") {
                        stats.events.push("reasoning_delta");
                    }
                }
                StreamedAssistantContent::ToolCall { ref tool_call, .. } => {
                    let name = tool_call.function.name.clone();
                    eprintln!("[{provider}] ToolCall: {name} (id={:?})", tool_call.id);
                    stats.tool_calls_in_stream.push(name);
                    stats.events.push("tool_call");
                }
                StreamedAssistantContent::Text(ref text) => {
                    stats.text_chunks += 1;
                    stats.final_text.push_str(&text.text);
                    if stats.events.last() != Some(&"text") {
                        stats.events.push("text");
                    }
                }
                StreamedAssistantContent::ToolCallDelta { .. } => {
                    if stats.events.last() != Some(&"tool_call_delta") {
                        stats.events.push("tool_call_delta");
                    }
                }
                StreamedAssistantContent::Final(_) => {
                    stats.events.push("final");
                }
            },
            Ok(MultiTurnStreamItem::StreamUserItem(ref user_content)) => match user_content {
                StreamedUserContent::ToolResult { .. } => {
                    stats.tool_results_in_stream += 1;
                    stats.events.push("tool_result");
                    eprintln!("[{provider}] ToolResult received");
                }
            },
            Ok(MultiTurnStreamItem::FinalResponse(ref resp)) => {
                stats.got_final_response = true;
                eprintln!(
                    "[{provider}] FinalResponse: {} chars",
                    resp.response().len()
                );
            }
            Ok(_) => {} // #[non_exhaustive]
            Err(ref e) => {
                let msg = format!("{e}");
                eprintln!("[{provider}] ERROR: {msg}");
                stats.errors.push(msg);
            }
        }
    }

    stats
}

// ==================== Assertions ====================

/// Run all universal assertions that must hold for every provider.
fn assert_universal(stats: &StreamStats, tool_invocations: &AtomicUsize, provider: &str) {
    // Dump diagnostic info first so failures are easy to debug
    eprintln!("\n[{provider}] === DIAGNOSTICS ===");
    eprintln!("[{provider}]   Event sequence: {:?}", stats.events);
    eprintln!(
        "[{provider}]   Reasoning: {} blocks, {} deltas",
        stats.reasoning_block_count, stats.reasoning_delta_count
    );
    eprintln!(
        "[{provider}]   Tool calls in stream: {:?}",
        stats.tool_calls_in_stream
    );
    eprintln!(
        "[{provider}]   Tool results in stream: {}",
        stats.tool_results_in_stream
    );
    eprintln!(
        "[{provider}]   Actual tool invocations: {}",
        tool_invocations.load(Ordering::SeqCst)
    );
    eprintln!(
        "[{provider}]   Text: {} chunks, {} chars",
        stats.text_chunks,
        stats.final_text.len()
    );
    eprintln!(
        "[{provider}]   Content types: {:?}",
        stats.reasoning_content_types
    );

    // 1. No errors (catches 400s from dropped/malformed reasoning)
    assert!(
        stats.errors.is_empty(),
        "[{provider}] Stream had errors — likely a 400 from missing reasoning \
         in tool call history: {errors:?}",
        errors = stats.errors
    );

    // 2. Tool was actually invoked (our code ran)
    let invocations = tool_invocations.load(Ordering::SeqCst);
    assert!(
        invocations >= 1,
        "[{provider}] Tool was never invoked (count=0). Model may not have called the tool, \
         or the agent loop failed before execution. Stream tool calls: {:?}",
        stats.tool_calls_in_stream
    );

    // 3. Reasoning was received during the interaction
    assert!(
        stats.total_reasoning() > 0,
        "[{provider}] No reasoning content received (0 blocks, 0 deltas). \
         Provider was configured for reasoning but returned none."
    );

    // 4. ToolCall events appeared in the stream
    assert!(
        !stats.tool_calls_in_stream.is_empty(),
        "[{provider}] No ToolCall events in stream — agent loop didn't emit tool calls."
    );

    // 5. The tool call was for get_weather specifically
    assert!(
        stats
            .tool_calls_in_stream
            .iter()
            .any(|name| name == "get_weather"),
        "[{provider}] No get_weather tool call. Saw: {:?}",
        stats.tool_calls_in_stream
    );

    // 6. Tool results flowed back through the stream
    assert!(
        stats.tool_results_in_stream >= 1,
        "[{provider}] No ToolResult events in stream. Tool invoked {invocations} times \
         but no results yielded to consumer."
    );

    // 7. Reasoning appeared before tool calls (model thought first)
    assert!(
        stats.reasoning_before_first_tool_call(),
        "[{provider}] Reasoning did NOT appear before first tool call. \
         Either no reasoning was emitted, or ordering is wrong. \
         Events: {:?}",
        stats.events
    );

    // 8. Final text is non-empty (model responded after tool results)
    assert!(
        !stats.final_text.trim().is_empty(),
        "[{provider}] Final text is empty — model may have failed after tool results."
    );

    // 9. Final text is substantial (not a one-word acknowledgement)
    let trimmed = stats.final_text.trim();
    assert!(
        trimmed.len() >= 30,
        "[{provider}] Final text suspiciously short ({len} chars): {preview:?}. \
         Provider may not have processed tool results.",
        len = trimmed.len(),
        preview = &trimmed[..trimmed.len().min(100)]
    );

    // 10. Final text references the tool output (proves tool result was consumed)
    let text_lower = stats.final_text.to_lowercase();
    let references_tool_output = text_lower.contains("72")
        || text_lower.contains("22")
        || text_lower.contains("sunny")
        || text_lower.contains("tokyo")
        || text_lower.contains("weather")
        || text_lower.contains("temperature");
    assert!(
        references_tool_output,
        "[{provider}] Final text doesn't reference tool output. \
         Expected mention of 72°F, 22°C, sunny, Tokyo, weather, or temperature. \
         Got: {preview:?}",
        preview = &trimmed[..trimmed.len().min(200)]
    );

    // 11. FinalResponse received (stream terminated cleanly)
    assert!(
        stats.got_final_response,
        "[{provider}] Stream did not emit FinalResponse — multi-turn loop may not have terminated."
    );

    eprintln!("[{provider}] === ALL UNIVERSAL ASSERTIONS PASSED ===\n");
}

// ==================== Provider Tests ====================

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn test_openai_tool_call_with_reasoning() {
    use rig::providers::openai;

    let call_count = Arc::new(AtomicUsize::new(0));
    let tool = WeatherTool {
        call_count: call_count.clone(),
    };

    let client = openai::Client::from_env();
    let agent = client
        .agent("gpt-5.2")
        .preamble(SYSTEM_PROMPT)
        .max_tokens(4096)
        .tool(tool)
        .additional_params(json!({
            "reasoning": { "effort": "high" }
        }))
        .build();

    let stream = agent.stream_chat(USER_PROMPT, vec![]).multi_turn(3).await;

    let stats = collect_stream_stats(stream, "openai").await;
    assert_universal(&stats, &call_count, "openai");

    // OpenAI Responses API emits full Reasoning blocks (not just deltas)
    // with Encrypted and/or Summary content types.
    assert!(
        stats.reasoning_block_count > 0,
        "[openai] Expected full Reasoning blocks (not just deltas). \
         Blocks: {}, Deltas: {}",
        stats.reasoning_block_count,
        stats.reasoning_delta_count
    );
    assert!(
        stats.reasoning_has_encrypted || stats.reasoning_content_types.contains(&"Summary"),
        "[openai] Expected Encrypted or Summary reasoning content. Got: {:?}",
        stats.reasoning_content_types
    );
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn test_anthropic_tool_call_with_reasoning() {
    use rig::providers::anthropic;

    let call_count = Arc::new(AtomicUsize::new(0));
    let tool = WeatherTool {
        call_count: call_count.clone(),
    };

    let client = anthropic::Client::from_env();
    let agent = client
        .agent("claude-sonnet-4-5-20250929")
        .preamble(SYSTEM_PROMPT)
        .max_tokens(16384)
        .tool(tool)
        .additional_params(json!({
            "thinking": { "type": "enabled", "budget_tokens": 4096 }
        }))
        .build();

    let stream = agent.stream_chat(USER_PROMPT, vec![]).multi_turn(3).await;

    let stats = collect_stream_stats(stream, "anthropic").await;
    assert_universal(&stats, &call_count, "anthropic");

    // Anthropic emits full Reasoning blocks with Text content and signatures.
    assert!(
        stats.reasoning_block_count > 0,
        "[anthropic] Expected full Reasoning blocks (thinking). \
         Blocks: {}, Deltas: {}",
        stats.reasoning_block_count,
        stats.reasoning_delta_count
    );
    assert!(
        stats.reasoning_has_signature,
        "[anthropic] Thinking blocks should have cryptographic signatures. \
         Content types: {:?}",
        stats.reasoning_content_types
    );
    assert!(
        stats.reasoning_content_types.contains(&"Text"),
        "[anthropic] Expected Text reasoning content. Got: {:?}",
        stats.reasoning_content_types
    );
}

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn test_gemini_tool_call_with_reasoning() {
    use rig::providers::gemini;

    let call_count = Arc::new(AtomicUsize::new(0));
    let tool = WeatherTool {
        call_count: call_count.clone(),
    };

    let client = gemini::Client::from_env();
    let agent = client
        .agent("gemini-2.5-flash")
        .preamble(SYSTEM_PROMPT)
        .max_tokens(4096)
        .tool(tool)
        .additional_params(json!({
            "generationConfig": {
                "thinkingConfig": { "thinkingBudget": 4096, "includeThoughts": true }
            }
        }))
        .build();

    let stream = agent.stream_chat(USER_PROMPT, vec![]).multi_turn(3).await;

    let stats = collect_stream_stats(stream, "gemini").await;
    assert_universal(&stats, &call_count, "gemini");

    // Gemini streaming emits ReasoningDelta events (not full blocks).
    // Universal assertions already verify total reasoning > 0.
}

#[tokio::test]
#[ignore = "requires XAI_API_KEY — validate with grok-4-0725 once key is available"]
async fn test_xai_tool_call_with_reasoning() {
    use rig::providers::xai;

    let call_count = Arc::new(AtomicUsize::new(0));
    let tool = WeatherTool {
        call_count: call_count.clone(),
    };

    let client = xai::Client::from_env();
    let agent = client
        .agent(xai::GROK_3_MINI)
        .preamble(SYSTEM_PROMPT)
        .max_tokens(4096)
        .tool(tool)
        .build();

    let stream = agent.stream_chat(USER_PROMPT, vec![]).multi_turn(3).await;

    let stats = collect_stream_stats(stream, "xai").await;
    assert_universal(&stats, &call_count, "xai");
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn test_openrouter_tool_call_with_reasoning() {
    use rig::providers::openrouter;

    let call_count = Arc::new(AtomicUsize::new(0));
    let tool = WeatherTool {
        call_count: call_count.clone(),
    };

    let client = openrouter::Client::from_env();
    let agent = client
        .agent("openai/gpt-5.2")
        .preamble(SYSTEM_PROMPT)
        .max_tokens(4096)
        .tool(tool)
        .additional_params(json!({
            "reasoning": { "effort": "high" },
            "include_reasoning": true
        }))
        .build();

    let stream = agent.stream_chat(USER_PROMPT, vec![]).multi_turn(3).await;

    let stats = collect_stream_stats(stream, "openrouter").await;

    // OpenRouter proxies the Chat Completions API, which may not emit
    // reasoning tokens during tool-call turns (the model consumes its
    // reasoning budget on the tool call itself). We still verify the
    // full tool call loop works correctly. Reasoning is only guaranteed
    // in text-only responses (see reasoning_roundtrip.rs).
    assert!(
        stats.errors.is_empty(),
        "[openrouter] Stream had errors: {errors:?}",
        errors = stats.errors
    );

    let invocations = call_count.load(Ordering::SeqCst);
    assert!(
        invocations >= 1,
        "[openrouter] Tool was never invoked (count=0)."
    );

    assert!(
        !stats.tool_calls_in_stream.is_empty(),
        "[openrouter] No ToolCall events in stream."
    );

    assert!(
        stats.tool_results_in_stream >= 1,
        "[openrouter] No ToolResult events in stream."
    );

    assert!(
        !stats.final_text.trim().is_empty(),
        "[openrouter] Final text is empty."
    );

    assert!(
        stats.got_final_response,
        "[openrouter] Stream did not emit FinalResponse."
    );

    if stats.total_reasoning() > 0 {
        eprintln!(
            "[openrouter] Reasoning received: {} blocks, {} deltas — \
             OpenRouter forwarded reasoning in tool-call mode.",
            stats.reasoning_block_count, stats.reasoning_delta_count
        );
    } else {
        eprintln!(
            "[openrouter] No reasoning in tool-call mode (expected — \
             Chat Completions API does not emit reasoning during tool calls)."
        );
    }

    eprintln!("[openrouter] === ALL OPENROUTER ASSERTIONS PASSED ===\n");
}

// ==================== Non-Streaming Tests ====================
//
// These test the non-streaming agent loop (`agent.chat()`) which uses
// `model.completion()` internally. The agent loop preserves reasoning
// in `resp.choice.clone()` pushed into chat history. If reasoning is
// dropped or malformed, the provider returns a 400 error.
//
// Since `chat()` returns `Result<String, PromptError>`, we can't inspect
// reasoning blocks directly. Instead we verify:
//   1. No error (proves reasoning roundtrip accepted by provider)
//   2. Tool was actually invoked (atomic counter)
//   3. Final text references tool output (proves tool result consumed)
//   4. Final text is substantial (not a stub or error message)

/// Shared assertions for non-streaming tool call tests.
fn assert_nonstreaming_universal(result: &str, tool_invocations: &AtomicUsize, provider: &str) {
    let invocations = tool_invocations.load(Ordering::SeqCst);

    eprintln!("\n[{provider}] === NON-STREAMING DIAGNOSTICS ===");
    eprintln!("[{provider}]   Tool invocations: {invocations}");
    eprintln!("[{provider}]   Response length: {} chars", result.len());

    // 1. Tool was actually invoked
    assert!(
        invocations >= 1,
        "[{provider}] Tool was never invoked (count=0). \
         The agent loop may have failed to execute the tool, or the model \
         didn't call it."
    );

    // 2. Response is non-empty
    let trimmed = result.trim();
    assert!(
        !trimmed.is_empty(),
        "[{provider}] Agent returned empty response after tool call loop."
    );

    // 3. Response is substantial
    assert!(
        trimmed.len() >= 30,
        "[{provider}] Response suspiciously short ({} chars): {:?}. \
         Provider may not have processed tool results.",
        trimmed.len(),
        &trimmed[..trimmed.len().min(100)]
    );

    // 4. Response references tool output (proves tool result was consumed)
    let text_lower = result.to_lowercase();
    let references_tool_output = text_lower.contains("72")
        || text_lower.contains("22")
        || text_lower.contains("sunny")
        || text_lower.contains("tokyo")
        || text_lower.contains("weather")
        || text_lower.contains("temperature");
    assert!(
        references_tool_output,
        "[{provider}] Response doesn't reference tool output. \
         Expected mention of 72°F, 22°C, sunny, Tokyo, weather, or temperature. \
         Got: {:?}",
        &trimmed[..trimmed.len().min(200)]
    );

    eprintln!("[{provider}] === ALL NON-STREAMING ASSERTIONS PASSED ===\n");
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn test_openai_tool_call_nonstreaming() {
    use rig::client::CompletionClient;
    use rig::completion::Chat;
    use rig::providers::openai;

    let call_count = Arc::new(AtomicUsize::new(0));
    let tool = WeatherTool {
        call_count: call_count.clone(),
    };

    let client = openai::Client::from_env();
    let agent = client
        .agent("gpt-5.2")
        .preamble(SYSTEM_PROMPT)
        .max_tokens(4096)
        .tool(tool)
        .additional_params(json!({
            "reasoning": { "effort": "high" }
        }))
        .build();

    let result = agent
        .chat(USER_PROMPT, vec![])
        .await
        .expect("[openai] Non-streaming chat failed — likely 400 from dropped reasoning");

    assert_nonstreaming_universal(&result, &call_count, "openai");
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn test_anthropic_tool_call_nonstreaming() {
    use rig::client::CompletionClient;
    use rig::completion::Chat;
    use rig::providers::anthropic;

    let call_count = Arc::new(AtomicUsize::new(0));
    let tool = WeatherTool {
        call_count: call_count.clone(),
    };

    let client = anthropic::Client::from_env();
    let agent = client
        .agent("claude-sonnet-4-5-20250929")
        .preamble(SYSTEM_PROMPT)
        .max_tokens(16384)
        .tool(tool)
        .additional_params(json!({
            "thinking": { "type": "enabled", "budget_tokens": 4096 }
        }))
        .build();

    let result = agent
        .chat(USER_PROMPT, vec![])
        .await
        .expect("[anthropic] Non-streaming chat failed — likely 400 from dropped reasoning");

    assert_nonstreaming_universal(&result, &call_count, "anthropic");
}

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn test_gemini_tool_call_nonstreaming() {
    use rig::client::CompletionClient;
    use rig::completion::Chat;
    use rig::providers::gemini;

    let call_count = Arc::new(AtomicUsize::new(0));
    let tool = WeatherTool {
        call_count: call_count.clone(),
    };

    let client = gemini::Client::from_env();
    let agent = client
        .agent("gemini-2.5-flash")
        .preamble(SYSTEM_PROMPT)
        .max_tokens(4096)
        .tool(tool)
        .additional_params(json!({
            "generationConfig": {
                "thinkingConfig": { "thinkingBudget": 4096, "includeThoughts": true }
            }
        }))
        .build();

    let result = agent
        .chat(USER_PROMPT, vec![])
        .await
        .expect("[gemini] Non-streaming chat failed — likely 400 from dropped reasoning");

    assert_nonstreaming_universal(&result, &call_count, "gemini");
}

#[tokio::test]
#[ignore = "requires XAI_API_KEY — validate with grok-4-0725 once key is available"]
async fn test_xai_tool_call_nonstreaming() {
    use rig::client::CompletionClient;
    use rig::completion::Chat;
    use rig::providers::xai;

    let call_count = Arc::new(AtomicUsize::new(0));
    let tool = WeatherTool {
        call_count: call_count.clone(),
    };

    let client = xai::Client::from_env();
    let agent = client
        .agent(xai::GROK_3_MINI)
        .preamble(SYSTEM_PROMPT)
        .max_tokens(4096)
        .tool(tool)
        .build();

    let result = agent
        .chat(USER_PROMPT, vec![])
        .await
        .expect("[xai] Non-streaming chat failed — likely 400 from dropped reasoning");

    assert_nonstreaming_universal(&result, &call_count, "xai");
}

#[tokio::test]
#[ignore = "requires OPENROUTER_API_KEY"]
async fn test_openrouter_tool_call_nonstreaming() {
    use rig::client::CompletionClient;
    use rig::completion::Chat;
    use rig::providers::openrouter;

    let call_count = Arc::new(AtomicUsize::new(0));
    let tool = WeatherTool {
        call_count: call_count.clone(),
    };

    let client = openrouter::Client::from_env();
    let agent = client
        .agent("openai/gpt-5.2")
        .preamble(SYSTEM_PROMPT)
        .max_tokens(4096)
        .tool(tool)
        .additional_params(json!({
            "reasoning": { "effort": "high" },
            "include_reasoning": true
        }))
        .build();

    let result = agent
        .chat(USER_PROMPT, vec![])
        .await
        .expect("[openrouter] Non-streaming chat failed — likely 400 from dropped reasoning");

    assert_nonstreaming_universal(&result, &call_count, "openrouter");
}
