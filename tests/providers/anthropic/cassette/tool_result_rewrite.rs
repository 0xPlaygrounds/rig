//! Verifies that a `ToolResultAction::Rewrite` hook redacts a tool's output before the
//! model sees it, end-to-end through a real Anthropic round-trip.
//!
//! The `get_user_record` tool returns a record containing a (fake) SSN. A default
//! hook redacts the SSN on the `ToolResult` event via `ToolResultAction::rewrite`, so
//! the value the tool actually produced never reaches the model. The tool records
//! its real output; the assertions check that the tool DID produce the secret but
//! the model's answer never contains it — and the blocking and streaming tests
//! assert the same behavior, since both drivers share the same tool seam.

use std::sync::{Arc, Mutex};

use rig::agent::AgentHook;
use rig::client::CompletionClient;
use rig::completion::{CompletionModel, Prompt};
use rig::providers::anthropic;
use rig::streaming::StreamingPrompt;
use rig::tool::{Tool, ToolContext, ToolExecutionError};
use serde::Deserialize;
use serde_json::json;

use super::super::support::with_anthropic_cassette;
use crate::support::collect_stream_final_response;

const LOOKUP_PROMPT: &str =
    "Look up user u-42 with the get_user_record tool and tell me their account status.";
/// The fake secret the tool emits and the hook must redact before the model sees it.
const SECRET_SSN: &str = "123-45-6789";
const PREAMBLE: &str = "You are a support agent. Use the get_user_record tool to look up a user, \
                        then report their account status.";

#[derive(Deserialize)]
struct LookupArgs {
    #[allow(dead_code)]
    user_id: String,
}

/// A tool that returns a record containing a sensitive field, recording its real
/// output so the test can assert the tool produced the secret even though the
/// model never sees it.
#[derive(Clone, Default)]
struct GetUserRecord {
    raw_outputs: Arc<Mutex<Vec<String>>>,
}

impl GetUserRecord {
    fn produced_secret(&self) -> bool {
        self.raw_outputs
            .lock()
            .expect("outputs lock")
            .iter()
            .any(|out| out.contains(SECRET_SSN))
    }
}

impl Tool for GetUserRecord {
    const NAME: &'static str = "get_user_record";
    type Args = LookupArgs;
    type Output = String;

    fn description(&self) -> String {
        "Look up a user record by id.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "user_id": { "type": "string", "description": "The user id, e.g. 'u-42'" }
            },
            "required": ["user_id"]
        })
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, ToolExecutionError> {
        // Constant (id-independent) so the round-trip is deterministic for replay.
        let record = format!("name=Alice; ssn={SECRET_SSN}; status=active");
        self.raw_outputs
            .lock()
            .expect("outputs lock")
            .push(record.clone());
        Ok(record)
    }
}

/// Redact the `ssn=...` field of a `"; "`-delimited record, leaving the rest.
fn redact_ssn(record: &str) -> String {
    record
        .split("; ")
        .map(|field| {
            if field.starts_with("ssn=") {
                "ssn=[REDACTED]".to_string()
            } else {
                field.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// A guardrail hook that redacts the SSN from `get_user_record` output on the
/// `ToolResult` event, before the model ever sees it — the post-tool redaction
/// use case `ToolResultAction::Rewrite` exists for.
struct RedactSsnFromResult;

impl<M: CompletionModel> AgentHook<M> for RedactSsnFromResult {
    async fn on_tool_result(
        &self,
        _ctx: &rig::agent::HookContext,
        event: rig::agent::ToolResultEvent<'_>,
    ) -> rig::agent::ToolResultAction {
        if event.tool_name == GetUserRecord::NAME {
            return rig::agent::ToolResultAction::rewrite(redact_ssn(event.result));
        }
        rig::agent::ToolResultAction::keep()
    }
}

fn assert_answer_hides_secret(answer: &str) {
    assert!(!answer.is_empty(), "agent should produce a final answer");
    assert!(
        !answer.contains(SECRET_SSN),
        "the redacted SSN must never reach the model, but the answer contained it: {answer}"
    );
}

#[tokio::test]
async fn tool_result_redacted_by_hook_blocking() {
    let tool = GetUserRecord::default();
    let probe = tool.clone();

    with_anthropic_cassette(
        "tool_result_rewrite/tool_result_redacted_by_hook_blocking",
        move |client| async move {
            let agent = client
                .agent(anthropic::completion::CLAUDE_SONNET_4_6)
                .preamble(PREAMBLE)
                .tool(tool)
                .add_hook(RedactSsnFromResult)
                .build();

            let response = agent
                .prompt(LOOKUP_PROMPT)
                .max_turns(5)
                .await
                .expect("blocking lookup should succeed");

            assert_answer_hides_secret(&response);
        },
    )
    .await;

    assert!(
        probe.produced_secret(),
        "the tool itself should have produced the secret SSN that the hook redacted"
    );
}

#[tokio::test]
async fn tool_result_redacted_by_hook_streaming() {
    let tool = GetUserRecord::default();
    let probe = tool.clone();

    with_anthropic_cassette(
        "tool_result_rewrite/tool_result_redacted_by_hook_streaming",
        move |client| async move {
            let agent = client
                .agent(anthropic::completion::CLAUDE_SONNET_4_6)
                .preamble(PREAMBLE)
                .tool(tool)
                .add_hook(RedactSsnFromResult)
                .build();

            let mut stream = agent.stream_prompt(LOOKUP_PROMPT).max_turns(5).await;
            let response = collect_stream_final_response(&mut stream)
                .await
                .expect("streaming lookup should succeed");

            assert_answer_hides_secret(&response);
        },
    )
    .await;

    assert!(
        probe.produced_secret(),
        "the tool itself should have produced the secret SSN that the hook redacted"
    );
}
