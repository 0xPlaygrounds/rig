//! Claude Opus 5.5 and Fable 5.1 bind each replayed thinking block to the
//! `system` prompt, the `tools` and every message before it. The agent loop
//! must therefore only append between requests, within a tool loop and across
//! runs that continue a conversation. These drive the real Anthropic wire with
//! scripted replies shaped like the documented progress-update responses.
//! Source: <https://platform.claude.com/docs/en/build-with-claude/preserved-thinking#keeping-the-prefix-unchanged>

#![allow(clippy::expect_used, clippy::indexing_slicing, clippy::panic)]

use rig_agent::{
    AgentBuilder,
    tool::{Tool, ToolContext, ToolExecutionError},
};
use rig_core::providers::anthropic::{
    completion::{CLAUDE_OPUS_5_5, Thinking, ThinkingDisplay},
    wire::AnthropicConfig,
};
use rig_core::test_utils::{MockHttpResponse, SequencedHttpClient};
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize)]
struct EditArgs {
    path: String,
}

struct EditFile;

impl Tool for EditFile {
    const NAME: &'static str = "edit_file";
    type Args = EditArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        "Replace the contents of a file in the repository.".into()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]
        })
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: EditArgs,
    ) -> Result<String, Self::Error> {
        Ok(format!("saved {}", args.path))
    }
}

fn reply(content: Value, stop_reason: &str) -> MockHttpResponse {
    MockHttpResponse::success(
        json!({
            "type": "message",
            "id": format!("msg_{stop_reason}"),
            "model": CLAUDE_OPUS_5_5,
            "role": "assistant",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": null,
            "usage": {"input_tokens": 100, "output_tokens": 40}
        })
        .to_string(),
    )
}

/// A reasoning block, a progress update and the `tool_use` it introduces.
fn tool_turn() -> Value {
    json!([
        {"type": "thinking", "thinking": "", "signature": "sig-reasoning-1"},
        {"type": "thinking", "thinking": "Found the retry path. Editing auth.py.", "signature": "sig-update-1"},
        {"type": "tool_use", "id": "toolu_01", "name": "edit_file",
         "input": {"path": "auth.py", "content": "refresh()"}}
    ])
}

fn answer_turn(text: &str, signature: &str) -> Value {
    json!([
        {"type": "thinking", "thinking": "", "signature": signature},
        {"type": "text", "text": text}
    ])
}

fn bodies(http: &SequencedHttpClient) -> Vec<Value> {
    http.requests()
        .iter()
        .map(|request| serde_json::from_slice(&request.body).expect("JSON request"))
        .collect()
}

/// `later` keeps `earlier`'s `system` and `tools`, and its messages start
/// with every message `earlier` sent, unchanged.
fn assert_appends(earlier: &Value, later: &Value) -> usize {
    assert_eq!(later["system"], earlier["system"], "system prompt changed");
    assert_eq!(later["tools"], earlier["tools"], "tool set changed");
    let earlier_messages = earlier["messages"].as_array().expect("messages");
    let later_messages = later["messages"].as_array().expect("messages");
    assert!(later_messages.len() > earlier_messages.len());
    assert_eq!(
        &later_messages[..earlier_messages.len()],
        earlier_messages.as_slice(),
        "an earlier message changed"
    );
    earlier_messages.len()
}

#[tokio::test]
async fn the_agent_loop_only_appends_between_requests_on_opus_5_5() {
    let http = SequencedHttpClient::new([
        reply(tool_turn(), "tool_use"),
        reply(
            answer_turn("Fixed the token refresh.", "sig-reasoning-2"),
            "end_turn",
        ),
        reply(
            answer_turn("It retries once.", "sig-reasoning-3"),
            "end_turn",
        ),
    ]);
    let mut model = AnthropicConfig::new("sk-test")
        .connect(http.clone())
        .completion(CLAUDE_OPUS_5_5);
    model.wire = model
        .wire
        .with_thinking(Thinking::adaptive().with_display(ThinkingDisplay::Updates));
    let agent = AgentBuilder::new(model)
        .preamble("You fix bugs.")
        .context("The repository is a Python service.")
        .tool(EditFile)
        .max_tokens(4096)
        .build();

    let first = agent
        .prompt("The login test fails after an hour of uptime.")
        .max_turns(3)
        .run()
        .await
        .expect("the tool loop completes");
    assert_eq!(first.output, "Fixed the token refresh.");

    let history = first
        .messages()
        .expect("the run tracks its messages")
        .to_vec();
    let second = agent
        .prompt("Does it retry?")
        .history(history)
        .max_turns(3)
        .run()
        .await
        .expect("the follow-up completes");
    assert_eq!(second.output, "It retries once.");

    let bodies = bodies(&http);
    assert_eq!(bodies.len(), 3);

    // Within the tool loop: the assistant turn comes back exactly as received,
    // progress update and empty reasoning block included, then its result.
    let sent = assert_appends(&bodies[0], &bodies[1]);
    assert_eq!(bodies[1]["messages"][sent]["role"], "assistant");
    assert_eq!(bodies[1]["messages"][sent]["content"], tool_turn());
    assert_eq!(
        bodies[1]["messages"][sent + 1]["content"][0]["type"],
        "tool_result"
    );

    // Across runs that continue the conversation.
    let sent = assert_appends(&bodies[1], &bodies[2]);
    assert_eq!(
        bodies[2]["messages"][sent]["content"],
        answer_turn("Fixed the token refresh.", "sig-reasoning-2")
    );
}

/// One streamed reply: each content block of `content` as its SSE events.
/// Thinking blocks open empty and stream their text and signature as deltas.
fn streamed_reply(content: &Value, stop_reason: &str) -> MockHttpResponse {
    let mut events = vec![json!({"type": "message_start", "message": {
        "type": "message", "id": format!("msg_{stop_reason}"), "model": CLAUDE_OPUS_5_5,
        "role": "assistant", "content": [], "stop_reason": null, "stop_sequence": null,
        "usage": {"input_tokens": 100, "output_tokens": 1}
    }})];
    for (index, block) in content
        .as_array()
        .expect("content blocks")
        .iter()
        .enumerate()
    {
        match block["type"].as_str() {
            Some("thinking") => {
                events.push(json!({"type": "content_block_start", "index": index,
                    "content_block": {"type": "thinking", "thinking": "", "signature": ""}}));
                events.push(json!({"type": "content_block_delta", "index": index,
                    "delta": {"type": "thinking_delta", "thinking": block["thinking"]}}));
                events.push(json!({"type": "content_block_delta", "index": index,
                    "delta": {"type": "signature_delta", "signature": block["signature"]}}));
            }
            Some("text") => {
                events.push(json!({"type": "content_block_start", "index": index,
                    "content_block": {"type": "text", "text": ""}}));
                events.push(json!({"type": "content_block_delta", "index": index,
                    "delta": {"type": "text_delta", "text": block["text"]}}));
            }
            Some("tool_use") => {
                events.push(json!({"type": "content_block_start", "index": index,
                    "content_block": {"type": "tool_use", "id": block["id"],
                        "name": block["name"], "input": {}}}));
                events.push(json!({"type": "content_block_delta", "index": index,
                    "delta": {"type": "input_json_delta",
                        "partial_json": block["input"].to_string()}}));
            }
            other => panic!("unscripted block {other:?}"),
        }
        events.push(json!({"type": "content_block_stop", "index": index}));
    }
    events.push(json!({"type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": null},
        "usage": {"output_tokens": 40}}));
    events.push(json!({"type": "message_stop"}));
    let body: String = events
        .iter()
        .map(|event| {
            format!(
                "event: {}\ndata: {event}\n\n",
                event["type"].as_str().unwrap_or("")
            )
        })
        .collect();
    MockHttpResponse::success_typed(body, "text/event-stream")
}

#[tokio::test]
async fn a_streamed_tool_loop_replays_adjacent_thinking_blocks_unchanged() {
    use futures::StreamExt;

    let http = SequencedHttpClient::new([
        streamed_reply(&tool_turn(), "tool_use"),
        streamed_reply(
            &answer_turn("Fixed the token refresh.", "sig-reasoning-2"),
            "end_turn",
        ),
    ]);
    let mut model = AnthropicConfig::new("sk-test")
        .connect(http.clone())
        .completion(CLAUDE_OPUS_5_5);
    model.wire = model
        .wire
        .with_thinking(Thinking::adaptive().with_display(ThinkingDisplay::Updates));
    let agent = AgentBuilder::new(model)
        .preamble("You fix bugs.")
        .tool(EditFile)
        .max_tokens(4096)
        .build();

    let mut stream = agent
        .prompt("The login test fails after an hour of uptime.")
        .max_turns(3)
        .stream();
    while let Some(item) = stream.next().await {
        item.expect("the streamed tool loop completes");
    }

    let bodies = bodies(&http);
    assert_eq!(bodies.len(), 2);
    let sent = assert_appends(&bodies[0], &bodies[1]);
    assert_eq!(bodies[1]["messages"][sent]["content"], tool_turn());
}
