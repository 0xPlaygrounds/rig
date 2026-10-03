//! Live facts behind Rig's handling of Claude thinking bound to the request's
//! tools and system prompt (#2703).
//!
//! Claude Opus 5.5 binds a signed thinking block to the tools it was made
//! under. Rig replays such a turn verbatim and asks Anthropic, under the
//! `thinking-binding-controls-2026-08-01` beta, to drop a block whose
//! binding no longer matches (`block_binding.prefix_mismatch_behavior:
//! "drop_block"`). These cells pin what that rests on:
//!
//! | # | Cell | Fact |
//! |---|------|------|
//! | 1 | `changed_tools_with_drop_block` | After the tools change, the signed thinking goes back verbatim with `drop_block` and the beta, and Anthropic accepts it |
//! | 2 | `changed_tools_without_drop_block` | The same request without the binding is refused with a 400 that names the beta |
//! | 3 | `thinkingless_tool_loop` | A tool loop opened by a turn with no thinking block is accepted under adaptive thinking |
//!
//! Cell 2 sends its second request raw: Rig always adds the binding.

use rig::completion::{CompletionRequest, CompletionResponse, Message, ToolDefinition};
use rig::message::{
    AssistantContent, AssistantMessage, CallId, ToolCall, ToolFunction, ToolName,
    ToolResultContent, UserContent,
};
use rig::providers::anthropic::completion::CLAUDE_OPUS_5_5;
use rig_test_support::cassette_models::AnthropicModels;
use serde_json::{Value, json};

use super::super::support::with_anthropic_cassette;

const BETA: &str = "thinking-binding-controls-2026-08-01";
const WITH_DROP_BLOCK: &str = "context_binding/changed_tools_with_drop_block";
const THINKINGLESS: &str = "thinking/thinkingless_tool_loop";

/// A question the model reasons about before verifying with the tool.
const PROMPT: &str = "A train leaves at 3:17pm going 83 km/h; another leaves 41 minutes later \
     going 112 km/h on the same track. Reason through when the second catches up, then call \
     calc once to verify your final expression.";

fn tool(name: &str) -> ToolDefinition {
    ToolDefinition::new(
        ToolName::new(name).expect("a tool name"),
        "Evaluates an arithmetic expression.",
        json!({"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}),
    )
}

/// Adaptive thinking with its summary shown, at high effort, so the first
/// turn thinks before it calls the tool.
fn thinking() -> Value {
    json!({
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "high"}
    })
}

/// The first turn: Opus 5.5 thinks, then calls `calc`.
async fn thinking_turn(models: &AnthropicModels) -> CompletionResponse {
    let request = CompletionRequest::new(PROMPT)
        .max_tokens(8000)
        .tool(tool("calc"))
        .additional_params(thinking());
    let response = models
        .completion(CLAUDE_OPUS_5_5)
        .call(request)
        .await
        .expect("the first turn succeeds");
    let kinds: Vec<&str> = response.raw["content"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|block| block["type"].as_str())
        .collect();
    assert!(
        kinds.contains(&"thinking") && kinds.contains(&"tool_use"),
        "the premise is a turn that thinks and calls a tool: {kinds:?}"
    );
    response
}

/// The user message answering every call of `turn`.
fn answers(turn: &AssistantMessage) -> Message {
    Message::User {
        content: turn
            .tool_calls()
            .map(|call| UserContent::ToolResult(call.result(vec![ToolResultContent::text("1.07")])))
            .collect(),
    }
}

fn assistant(message: Message) -> AssistantMessage {
    match message {
        Message::Assistant(turn) => turn,
        other => panic!("expected an assistant turn, got {other:?}"),
    }
}

fn request_bodies(scenario: &str) -> Vec<Value> {
    crate::cassettes::recorded_interaction_bodies("anthropic", scenario)
        .into_iter()
        .map(|(request, _)| serde_json::from_str(&request).expect("a JSON request body"))
        .collect()
}

fn tool_names(body: &Value) -> Vec<&str> {
    body["tools"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|tool| tool["name"].as_str())
        .collect()
}

fn signatures(message: &Value) -> Vec<&str> {
    message["content"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|block| block["type"] == "thinking")
        .filter_map(|block| block["signature"].as_str())
        .collect()
}

#[tokio::test]
async fn changed_tools_with_drop_block() {
    with_anthropic_cassette(
        "context_binding/changed_tools_with_drop_block",
        |models| async move {
            let first = thinking_turn(&models).await;
            let turn = assistant(first.message().expect("a turn"));
            let next = answers(&turn);
            let request = CompletionRequest::new(next)
                .messages([Message::user(PROMPT), Message::Assistant(turn)])
                .max_tokens(8000)
                .tools(vec![tool("calc"), tool("plot")])
                .additional_params(thinking());
            let reply = models
                .completion(CLAUDE_OPUS_5_5)
                .call(request)
                .await
                .expect("Anthropic accepts thinking bound to the old tools under drop_block");
            assert!(!reply.choice.is_empty(), "the second turn answers");
        },
    )
    .await;

    let bodies = request_bodies(WITH_DROP_BLOCK);
    assert_eq!(bodies.len(), 2, "two turns are recorded");
    let (first, second) = (&bodies[0], &bodies[1]);
    assert_ne!(tool_names(first), tool_names(second), "the tools change");
    assert_eq!(
        second["thinking"]["block_binding"]["prefix_mismatch_behavior"], "drop_block",
        "the second request asks Anthropic to drop a mismatched block"
    );
    let replayed = signatures(&second["messages"][1]);
    assert!(
        !replayed.is_empty() && replayed.iter().all(|signature| !signature.is_empty()),
        "the signed thinking made under the old tools goes back verbatim: {}",
        second["messages"][1]
    );
    let headers = crate::cassettes::recorded_request_header_pairs("anthropic", WITH_DROP_BLOCK);
    assert!(
        headers[1]
            .iter()
            .any(|(name, value)| name == "anthropic-beta" && value.contains(BETA)),
        "the second request carries the binding beta"
    );
    let statuses = crate::cassettes::recorded_statuses_and_bodies("anthropic", WITH_DROP_BLOCK);
    assert_eq!(statuses[1].0, 200, "Anthropic accepts it");
}

#[tokio::test]
async fn changed_tools_without_drop_block() {
    with_anthropic_cassette("context_binding/changed_tools_without_drop_block", |models| async move {
        let first = thinking_turn(&models).await;
        let content = first.raw["content"].clone();
        let results: Vec<Value> = content
            .as_array()
            .into_iter()
            .flatten()
            .filter(|block| block["type"] == "tool_use")
            .map(|block| json!({"type": "tool_result", "tool_use_id": block["id"], "content": "1.07"}))
            .collect();
        // Rig always adds the binding, so the second request goes raw:
        // the first turn verbatim under changed tools, without the beta.
        let body = json!({
            "model": CLAUDE_OPUS_5_5,
            "max_tokens": 8000,
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": "high"},
            "tools": [
                {"name": "calc", "description": "Evaluates an arithmetic expression.",
                 "input_schema": tool("calc").parameters},
                {"name": "plot", "description": "Evaluates an arithmetic expression.",
                 "input_schema": tool("plot").parameters},
            ],
            "messages": [
                {"role": "user", "content": PROMPT},
                {"role": "assistant", "content": content},
                {"role": "user", "content": results},
            ],
        });
        let http = reqwest::Client::builder()
            .no_proxy()
            .build()
            .expect("an HTTP client");
        let response = http
            .post(format!(
                "{}/v1/messages",
                models.config.base_url.trim_end_matches('/')
            ))
            .header("x-api-key", models.config.api_key.expose())
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .body(body.to_string())
            .send()
            .await
            .expect("the raw request is sent");
        let status = response.status().as_u16();
        let reply: Value = response.json().await.expect("a JSON error body");
        assert_eq!(status, 400, "the mismatched binding is refused: {reply}");
        let message = reply["error"]["message"].as_str().unwrap_or_default();
        assert!(
            message.contains(BETA) && message.contains("drop_block"),
            "the refusal names the beta and drop_block: {message}"
        );
    })
    .await;
}

#[tokio::test]
async fn thinkingless_tool_loop() {
    with_anthropic_cassette("thinking/thinkingless_tool_loop", |models| async move {
        // A turn another model made: text, then a call, and no thinking.
        let call = ToolCall::new(
            CallId::from_wire("toolu_x1"),
            ToolFunction::new(
                ToolName::new("calc").expect("a tool name"),
                json!({"q": "17*23"}),
            ),
        );
        let turn = AssistantMessage::new(vec![
            AssistantContent::text("Let me compute."),
            AssistantContent::ToolCall(call.clone()),
        ]);
        let request = CompletionRequest::new(Message::User {
            content: vec![UserContent::ToolResult(
                call.result(vec![ToolResultContent::text("391")]),
            )],
        })
        .messages([
            Message::user("What is 17*23? Use calc."),
            Message::Assistant(turn),
        ])
        .max_tokens(3000)
        .tool(tool("calc"))
        .additional_params(json!({"thinking": {"type": "adaptive"}}));
        let reply = models
            .completion(CLAUDE_OPUS_5_5)
            .call(request)
            .await
            .expect("a tool loop opened without thinking is accepted under adaptive thinking");
        assert!(!reply.choice.is_empty(), "the model answers");
    })
    .await;

    let bodies = request_bodies(THINKINGLESS);
    let body = &bodies[0];
    assert_eq!(body["thinking"]["type"], "adaptive", "thinking is on");
    assert!(
        signatures(&body["messages"][1]).is_empty()
            && body["messages"][1]["content"][0]["type"] == "text",
        "the open loop's assistant turn starts with text and has no thinking: {}",
        body["messages"][1]
    );
    let statuses = crate::cassettes::recorded_statuses_and_bodies("anthropic", THINKINGLESS);
    assert_eq!(statuses[0].0, 200, "Anthropic accepts it");
}
