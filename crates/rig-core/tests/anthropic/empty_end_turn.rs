//! Anthropic live regression coverage for empty `end_turn` tool follow-ups.
//!
//! Run only these ignored cases with:
//! `cargo test -p rig-core --test anthropic anthropic::empty_end_turn -- --ignored --nocapture --test-threads=1`

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use rig_core::{
    client::{CompletionClient, ProviderClient},
    completion::{CompletionModel, Prompt, ToolDefinition},
    message::{AssistantContent, Message, UserContent},
    providers::anthropic::{self, completion::CLAUDE_SONNET_4_6},
    tool::Tool,
};
use serde::Deserialize;
use serde_json::json;

const TERMINAL_NOTIFY_PREAMBLE: &str = "\
When the user reports their status, call `notify` with a short summary. \
Do not answer with any normal assistant text before or after the tool call. \
Once the tool result is available, the assistant turn is complete and you must end the turn with no content.";

const TERMINAL_NOTIFY_WITH_ACK_PREAMBLE: &str = "\
When the user reports their status, first write one short sentence acknowledging that you are sending the notification. \
Then call `notify` with a short summary. \
After the tool result is available, stop immediately and do not send any more assistant text.";

const TERMINAL_NOTIFY_PROMPT: &str = "I finished the deploy.";

#[derive(Deserialize)]
struct NotifyArgs {
    msg: String,
}

#[derive(Debug, thiserror::Error)]
#[error("notify error")]
struct NotifyError;

fn notify_tool_definition() -> ToolDefinition {
    ToolDefinition {
        name: Notify::NAME.to_string(),
        description: "Send a short notification for a user status update.".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {
                "msg": {
                    "type": "string",
                    "description": "The short notification to send."
                }
            },
            "required": ["msg"]
        }),
    }
}

struct Notify {
    call_count: Arc<AtomicUsize>,
}

impl Notify {
    fn new(call_count: Arc<AtomicUsize>) -> Self {
        Self { call_count }
    }
}

impl Tool for Notify {
    const NAME: &'static str = "notify";
    type Error = NotifyError;
    type Args = NotifyArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        notify_tool_definition()
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(format!("sent: {}", args.msg))
    }
}

fn assistant_message_has_notify_tool_call(message: &Message) -> bool {
    matches!(
        message,
        Message::Assistant { content, .. }
            if content.iter().any(|item| matches!(
                item,
                AssistantContent::ToolCall(tool_call) if tool_call.function.name == Notify::NAME
            ))
    )
}

fn assistant_message_has_nonempty_text_and_notify_tool_call(message: &Message) -> bool {
    matches!(
        message,
        Message::Assistant { content, .. }
            if content.iter().any(|item| matches!(
                item,
                AssistantContent::Text(text) if !text.text.trim().is_empty()
            )) && content.iter().any(|item| matches!(
                item,
                AssistantContent::ToolCall(tool_call) if tool_call.function.name == Notify::NAME
            ))
    )
}

fn message_has_tool_result(message: &Message) -> bool {
    matches!(
        message,
        Message::User { content }
            if content.iter().any(|item| matches!(item, UserContent::ToolResult(_)))
    )
}

fn history_has_empty_assistant_text(messages: &[Message]) -> bool {
    messages.iter().any(|message| {
        matches!(
            message,
            Message::Assistant { content, .. }
                if content.iter().any(|item| matches!(
                    item,
                    AssistantContent::Text(text) if text.text.is_empty()
                ))
        )
    })
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn raw_followup_empty_end_turn_normalizes_to_empty_text_choice() {
    let model = anthropic::Client::from_env()
        .expect("client should build")
        .completion_model(CLAUDE_SONNET_4_6);

    let first_turn = model
        .completion_request(TERMINAL_NOTIFY_PROMPT)
        .preamble(TERMINAL_NOTIFY_PREAMBLE.to_string())
        .max_tokens(1024)
        .tool(notify_tool_definition())
        .send()
        .await
        .expect("first Anthropic turn should succeed");

    let tool_call = first_turn
        .choice
        .iter()
        .find_map(|item| match item {
            AssistantContent::ToolCall(tool_call) => Some(tool_call.clone()),
            _ => None,
        })
        .expect("first Anthropic turn should emit a notify tool call");

    let followup = model
        .completion_request(Message::tool_result_with_call_id(
            tool_call.id.clone(),
            tool_call.call_id.clone(),
            "sent: deploy finished",
        ))
        .preamble(TERMINAL_NOTIFY_PREAMBLE.to_string())
        .max_tokens(1024)
        .message(Message::Assistant {
            id: first_turn.message_id.clone(),
            content: first_turn.choice.clone(),
        })
        .send()
        .await
        .expect("follow-up Anthropic turn should not error on empty end_turn");

    assert_eq!(
        followup.choice.len(),
        1,
        "expected normalized empty follow-up choice, got {:?}",
        followup.choice
    );

    match followup.choice.first() {
        AssistantContent::Text(text) => assert!(
            text.text.is_empty(),
            "expected empty follow-up text sentinel, got {:?}",
            text.text
        ),
        other => panic!("expected empty text sentinel, got {other:?}"),
    }
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn prompt_loop_accepts_empty_terminal_turn_after_tool_result() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let agent = anthropic::Client::from_env()
        .expect("client should build")
        .agent(CLAUDE_SONNET_4_6)
        .preamble(TERMINAL_NOTIFY_PREAMBLE)
        .max_tokens(1024)
        .tool(Notify::new(call_count.clone()))
        .build();

    let response = agent
        .prompt(TERMINAL_NOTIFY_PROMPT)
        .max_turns(5)
        .extended_details()
        .await
        .expect("agent prompt should not fail on an empty terminal Anthropic turn");

    assert!(
        response.output.trim().is_empty(),
        "expected empty final output for the terminal tool prompt, got {:?}",
        response.output
    );
    assert!(
        call_count.load(Ordering::SeqCst) >= 1,
        "notify should be called at least once"
    );

    let messages = response
        .messages
        .expect("extended details should include history");
    assert!(
        messages.iter().any(assistant_message_has_notify_tool_call),
        "expected notify tool call in history, got {:?}",
        messages
    );
    assert!(
        messages.iter().any(message_has_tool_result),
        "expected tool result in history, got {:?}",
        messages
    );
    assert!(
        !history_has_empty_assistant_text(&messages),
        "history should not contain the normalized empty assistant sentinel: {:?}",
        messages
    );
}

#[tokio::test]
#[ignore = "requires ANTHROPIC_API_KEY"]
async fn prompt_loop_preserves_pre_tool_text_when_terminal_followup_is_empty() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let agent = anthropic::Client::from_env()
        .expect("client should build")
        .agent(CLAUDE_SONNET_4_6)
        .preamble(TERMINAL_NOTIFY_WITH_ACK_PREAMBLE)
        .max_tokens(1024)
        .tool(Notify::new(call_count.clone()))
        .build();

    let response = agent
        .prompt(TERMINAL_NOTIFY_PROMPT)
        .max_turns(5)
        .extended_details()
        .await
        .expect("agent prompt should preserve prior-turn text when Anthropic ends empty");

    assert!(
        response.output.trim().is_empty(),
        "expected empty final output for the terminal tool prompt, got {:?}",
        response.output
    );
    assert!(
        call_count.load(Ordering::SeqCst) >= 1,
        "notify should be called at least once"
    );

    let messages = response
        .messages
        .expect("extended details should include history");
    assert!(
        messages
            .iter()
            .any(assistant_message_has_nonempty_text_and_notify_tool_call),
        "expected an assistant message that preserved pre-tool text alongside the notify tool call, got {:?}",
        messages
    );
    assert!(
        messages.iter().any(message_has_tool_result),
        "expected tool result in history, got {:?}",
        messages
    );
    assert!(
        !history_has_empty_assistant_text(&messages),
        "history should not contain the normalized empty assistant sentinel: {:?}",
        messages
    );
}
