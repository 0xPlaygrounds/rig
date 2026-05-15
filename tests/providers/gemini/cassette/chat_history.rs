//! Gemini high-level Chat history regression tests.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rig::client::CompletionClient;
use rig::completion::{Chat, Message, ToolDefinition};
use rig::message::{AssistantContent, UserContent};
use rig::providers::gemini;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::json;

use crate::reasoning::{self, WeatherTool};

const STRESS_EXPECTED_FINAL: &str = "NOVA-200-142-LIME";

#[derive(Debug, thiserror::Error)]
#[error("stress calculator error")]
struct StressCalculatorError;

#[derive(Deserialize)]
struct StressMathArgs {
    x: i32,
    y: i32,
}

struct StressAdd {
    call_count: Arc<AtomicUsize>,
}

impl StressAdd {
    fn new(call_count: Arc<AtomicUsize>) -> Self {
        Self { call_count }
    }
}

impl Tool for StressAdd {
    const NAME: &'static str = "stress_add";
    type Error = StressCalculatorError;
    type Args = StressMathArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Add x and y. This tool must be used for stress-test addition turns."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "x": { "type": "number", "description": "Left operand" },
                    "y": { "type": "number", "description": "Right operand" }
                },
                "required": ["x", "y"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(args.x + args.y)
    }
}

struct StressSubtract {
    call_count: Arc<AtomicUsize>,
}

impl StressSubtract {
    fn new(call_count: Arc<AtomicUsize>) -> Self {
        Self { call_count }
    }
}

impl Tool for StressSubtract {
    const NAME: &'static str = "stress_subtract";
    type Error = StressCalculatorError;
    type Args = StressMathArgs;
    type Output = i32;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description:
                "Subtract y from x. This tool must be used for stress-test subtraction turns."
                    .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "x": { "type": "number", "description": "Value to subtract from" },
                    "y": { "type": "number", "description": "Value to subtract" }
                },
                "required": ["x", "y"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(args.x - args.y)
    }
}

#[tokio::test]
async fn chat_appends_reasoning_tool_turns_to_caller_history() {
    let call_count = Arc::new(AtomicUsize::new(0));
    super::super::support::with_gemini_cassette(
        "chat_history/chat_appends_reasoning_tool_turns_to_caller_history",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(reasoning::TOOL_SYSTEM_PROMPT)
                .max_tokens(4096)
                .tool(WeatherTool::new(call_count.clone()))
                .additional_params(serde_json::json!({
                    "generationConfig": {
                        "thinkingConfig": { "thinkingBudget": 4096, "includeThoughts": true }
                    }
                }))
                .build();
            let mut chat_history = Vec::<Message>::new();

            let result = agent
                .chat(reasoning::TOOL_USER_PROMPT, &mut chat_history)
                .await
                .expect("[gemini] Chat failed before it could update caller-owned history");

            reasoning::assert_nonstreaming_universal(&result, &call_count, "gemini");
            reasoning::assert_chat_history_preserves_reasoning_tool_roundtrip(
                &chat_history,
                &result,
                "gemini",
            );
        },
    )
    .await;
}

#[tokio::test]
async fn five_turn_chat_history_stress_preserves_context_and_tools() {
    let add_count = Arc::new(AtomicUsize::new(0));
    let subtract_count = Arc::new(AtomicUsize::new(0));
    super::super::support::with_gemini_cassette("chat_history/five_turn_chat_history_stress_preserves_context_and_tools", |client| async move {
    let agent = client
        .agent(gemini::completion::GEMINI_2_5_FLASH)
        .preamble(
            "You are running a deterministic Rig integration test. Preserve facts across turns. \
             When a prompt says to use a tool, call exactly that tool before answering. \
             Keep replies concise. When asked for the final code, return only the code with no \
             markdown, quotes, or commentary.",
        )
        .temperature(0.0)
        .max_tokens(4096)
        .tool(StressAdd::new(add_count.clone()))
        .tool(StressSubtract::new(subtract_count.clone()))
        .additional_params(serde_json::json!({
            "generationConfig": {
                "thinkingConfig": { "thinkingBudget": 1024, "includeThoughts": true }
            }
        }))
        .build();

    let mut chat_history = Vec::<Message>::new();

    let turn1 = agent
        .chat(
            "Turn 1 of 5. Remember that the alpha token is NOVA. Reply exactly ACK-ALPHA.",
            &mut chat_history,
        )
        .await
        .expect("[gemini] turn 1 should succeed");
    assert_response_contains("turn 1", &turn1, "ACK-ALPHA");

    let turn2 = agent
        .chat(
            "Turn 2 of 5. Use the stress_add tool to compute 123 + 77. \
             Remember the tool result as SUM. Reply exactly SUM=<tool result>.",
            &mut chat_history,
        )
        .await
        .expect("[gemini] turn 2 should succeed");
    assert_response_contains("turn 2", &turn2, "200");

    let turn3 = agent
        .chat(
            "Turn 3 of 5. Use the stress_subtract tool to compute the remembered SUM minus 58. \
             If you need concrete tool arguments, use x=200 and y=58. \
             Remember the tool result as DELTA. Reply exactly DELTA=<tool result>.",
            &mut chat_history,
        )
        .await
        .expect("[gemini] turn 3 should succeed");
    assert_response_contains("turn 3", &turn3, "142");

    let turn4 = agent
        .chat(
            "Turn 4 of 5. Remember that the suffix token is LIME. \
             Also remember that the final code format is <ALPHA>-<SUM>-<DELTA>-<SUFFIX>. \
             Reply exactly ACK-SUFFIX.",
            &mut chat_history,
        )
        .await
        .expect("[gemini] turn 4 should succeed");
    assert_response_contains("turn 4", &turn4, "ACK-SUFFIX");

    let final_result = agent
        .chat(
            "Turn 5 of 5. Using only facts stored in the previous turns, produce the final code. \
             Reply with the code only.",
            &mut chat_history,
        )
        .await
        .expect("[gemini] turn 5 should succeed");

    assert_eq!(
        final_result.trim(),
        STRESS_EXPECTED_FINAL,
        "[gemini] final stress result mismatch. Full chat history: {chat_history:#?}"
    );
    assert_eq!(
        add_count.load(Ordering::SeqCst),
        1,
        "[gemini] stress_add should be called exactly once"
    );
    assert_eq!(
        subtract_count.load(Ordering::SeqCst),
        1,
        "[gemini] stress_subtract should be called exactly once"
    );

    assert_eq!(
        count_text_user_turns(&chat_history),
        5,
        "[gemini] expected five caller chat turns in history: {chat_history:#?}"
    );
    assert!(
        chat_history.len() >= 14,
        "[gemini] expected five chat turns plus two tool roundtrips in history, got {}: {chat_history:#?}",
        chat_history.len()
    );
    assert!(
        count_assistant_tool_calls(&chat_history, StressAdd::NAME) >= 1,
        "[gemini] chat history is missing stress_add tool call: {chat_history:#?}"
    );
    assert!(
        count_assistant_tool_calls(&chat_history, StressSubtract::NAME) >= 1,
        "[gemini] chat history is missing stress_subtract tool call: {chat_history:#?}"
    );
    assert!(
        count_user_tool_results(&chat_history) >= 2,
        "[gemini] chat history should contain both tool results: {chat_history:#?}"
    );

    })
    .await;
}

fn assert_response_contains(turn: &str, response: &str, expected: &str) {
    assert!(
        response.contains(expected),
        "[gemini] {turn} response should contain {expected:?}, got {response:?}"
    );
}

fn count_text_user_turns(chat_history: &[Message]) -> usize {
    chat_history
        .iter()
        .filter(|message| match message {
            Message::User { content } => content.iter().any(|content| match content {
                UserContent::Text(text) => text.text.contains("Turn "),
                _ => false,
            }),
            _ => false,
        })
        .count()
}

fn count_assistant_tool_calls(chat_history: &[Message], tool_name: &str) -> usize {
    chat_history
        .iter()
        .filter_map(|message| match message {
            Message::Assistant { content, .. } => Some(content),
            _ => None,
        })
        .flat_map(|content| content.iter())
        .filter(|content| match content {
            AssistantContent::ToolCall(tool_call) => tool_call.function.name == tool_name,
            _ => false,
        })
        .count()
}

fn count_user_tool_results(chat_history: &[Message]) -> usize {
    chat_history
        .iter()
        .filter_map(|message| match message {
            Message::User { content } => Some(content),
            _ => None,
        })
        .flat_map(|content| content.iter())
        .filter(|content| matches!(content, UserContent::ToolResult(_)))
        .count()
}
