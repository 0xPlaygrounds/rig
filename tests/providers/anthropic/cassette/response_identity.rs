//! Response identity metadata (rig#2265): the native response id and the
//! transport request id (`request-id` response header) reach every observer
//! surface, with blocking/streaming parity.

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use rig::agent::{AgentHook, HookContext, ObservationAction, StreamResponseFinish};
use rig::completion::{CompletionModel, Message};
use rig::prelude::*;
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;
use rig::streaming::StreamedAssistantContent;
use rig::tool::Tool;

use super::super::support::with_anthropic_cassette;
use crate::support::{Adder, TOOLS_PREAMBLE};

fn assert_request_id(id: Option<&str>, context: &str) {
    assert!(
        id.is_some_and(|id| !id.trim().is_empty()),
        "{context}: Anthropic reports a `request-id` response header, so \
         provider_request_id must be populated"
    );
}

#[tokio::test]
async fn nonstreaming_response_carries_identity() {
    with_anthropic_cassette(
        "response_identity/nonstreaming_response_carries_identity",
        |client| async move {
            let model = client.completion_model(CLAUDE_SONNET_4_6);
            let response = model
                .completion_request("Reply with exactly: identity probe")
                .max_tokens(32)
                .send()
                .await
                .expect("completion should succeed");

            assert!(
                response
                    .message_id
                    .as_deref()
                    .is_some_and(|id| id.starts_with("msg")),
                "Anthropic reports message.id, got {:?}",
                response.message_id
            );
            assert_request_id(response.provider_request_id.as_deref(), "blocking response");
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_terminal_carries_identity() {
    with_anthropic_cassette(
        "response_identity/streaming_terminal_carries_identity",
        |client| async move {
            let model = client.completion_model(CLAUDE_SONNET_4_6);
            let mut stream = model
                .completion_request("Reply with exactly: stream identity probe")
                .max_tokens(32)
                .stream()
                .await
                .expect("stream should open");

            let mut terminal = None;
            while let Some(item) = stream.next().await {
                if let StreamedAssistantContent::Final(final_record) =
                    item.expect("stream item should succeed")
                {
                    terminal = Some(final_record);
                }
            }
            let terminal = terminal.expect("stream should yield a terminal record");

            assert!(
                terminal
                    .message_id
                    .as_deref()
                    .is_some_and(|id| id.starts_with("msg")),
                "streaming terminal carries message_id, got {:?}",
                terminal.message_id
            );
            // Blocking/streaming parity: the SSE connection's `request-id`
            // header lands on the terminal record.
            assert_request_id(
                terminal.provider_request_id.as_deref(),
                "streaming terminal",
            );
        },
    )
    .await;
}

type IdentityPair = (Option<String>, Option<String>);

#[derive(Clone, Default)]
struct IdentityCapture {
    blocking: Arc<Mutex<Vec<IdentityPair>>>,
    streaming: Arc<Mutex<Vec<IdentityPair>>>,
}

impl AgentHook for IdentityCapture {
    async fn on_completion_response(
        &self,
        _ctx: &HookContext,
        event: rig::agent::CompletionResponseEvent<'_>,
    ) -> ObservationAction {
        self.blocking.lock().expect("snapshots").push((
            event.message_id.map(str::to_owned),
            event.identity.provider_request_id.clone(),
        ));
        ObservationAction::continue_run()
    }

    async fn on_stream_response_finish(
        &self,
        _ctx: &HookContext,
        event: StreamResponseFinish<'_>,
    ) -> ObservationAction {
        self.streaming.lock().expect("snapshots").push((
            event.message_id.map(str::to_owned),
            event.identity.provider_request_id.clone(),
        ));
        ObservationAction::continue_run()
    }
}

/// A two-call tool run: each `CompletionCall` and each hook observation
/// carries the identity of its *own* attempt — two different request ids, not
/// stale run state.
#[tokio::test]
async fn agent_run_records_per_attempt_identity() {
    with_anthropic_cassette(
        "response_identity/agent_run_records_per_attempt_identity",
        |client| async move {
            let hook = IdentityCapture::default();
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .preamble(TOOLS_PREAMBLE)
                .max_tokens(1024)
                .tool(Adder)
                .add_hook(hook.clone())
                .build();

            let response = agent
                .prompt("What is 2 + 3? Use the tool, then state the result.")
                .max_turns(3)
                .extended_details()
                .await
                .expect("agent run should succeed");

            let calls = &response.completion_calls;
            assert!(
                calls.len() >= 2,
                "a tool run makes at least two completion calls, got {}",
                calls.len()
            );
            for call in calls {
                assert_request_id(
                    call.provider_request_id.as_deref(),
                    "completion_calls entry",
                );
                assert!(call.message_id.is_some(), "per-call message_id");
            }
            let request_ids: Vec<_> = calls
                .iter()
                .map(|call| call.provider_request_id.clone())
                .collect();
            assert_ne!(
                request_ids[0], request_ids[1],
                "each attempt reports its own request id"
            );

            let seen = hook.blocking.lock().expect("snapshots").clone();
            assert_eq!(seen.len(), calls.len(), "one observation per model call");
            for (index, (message_id, request_id)) in seen.iter().enumerate() {
                assert_eq!(
                    message_id.as_deref(),
                    calls[index].message_id.as_deref(),
                    "hook and completion_calls agree on message_id"
                );
                assert_eq!(
                    request_id.as_deref(),
                    calls[index].provider_request_id.as_deref(),
                    "hook and completion_calls agree on request id"
                );
            }
        },
    )
    .await;
}

/// Streaming agent-run parity: the stream response-finish hook observes the
/// same identity metadata the blocking hook does.
#[tokio::test]
async fn streamed_agent_run_hook_observes_identity() {
    with_anthropic_cassette(
        "response_identity/streamed_agent_run_hook_observes_identity",
        |client| async move {
            let hook = IdentityCapture::default();
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .preamble("You are a terse assistant.")
                .max_tokens(64)
                .add_hook(hook.clone())
                .build();

            let mut stream = rig::streaming::StreamingPrompt::stream_prompt(
                &agent,
                Message::user("Reply with exactly: streamed hook identity probe"),
            )
            .await;
            while let Some(item) = stream.next().await {
                item.expect("stream item should succeed");
            }

            let seen = hook.streaming.lock().expect("snapshots").clone();
            assert_eq!(seen.len(), 1, "one stream response-finish observation");
            let (message_id, request_id) = &seen[0];
            assert!(
                message_id
                    .as_deref()
                    .is_some_and(|id| id.starts_with("msg")),
                "streamed hook sees message_id, got {message_id:?}"
            );
            assert_request_id(request_id.as_deref(), "stream response-finish hook");
        },
    )
    .await;
}

// Silence unused-import lint when Tool is only used via `Adder::NAME`-style
// references in future edits.
#[allow(dead_code)]
fn _tool_trait_in_scope() -> &'static str {
    Adder::NAME
}

/// Live proof of the #2313 hook-coverage fix: a *streamed* tool run's
/// tool-only turn fires no `StreamResponseFinish`, yet its
/// `ModelTurnFinished` carries the attempt's full identity — and each of the
/// run's attempts reports its own request id.
#[tokio::test]
async fn streamed_agent_tool_run_reports_per_attempt_identity() {
    use crate::support::IdentityProbe;

    with_anthropic_cassette(
        "response_identity/streamed_agent_tool_run_reports_per_attempt_identity",
        |client| async move {
            let probe = IdentityProbe::default();
            let agent = client
                .agent(CLAUDE_SONNET_4_6)
                .preamble(TOOLS_PREAMBLE)
                .max_tokens(1024)
                .tool(Adder)
                .add_hook(probe.clone())
                .build();

            let mut stream = agent
                .runner(Message::user(
                    "What is 2 + 3? Use the tool, then state the result.",
                ))
                .max_turns(3)
                .stream()
                .await;
            let mut completion_calls = Vec::new();
            while let Some(item) = stream.next().await {
                if let rig::agent::MultiTurnStreamItem::CompletionCall(call) =
                    item.expect("stream item should succeed")
                {
                    completion_calls.push(call);
                }
            }

            let turns = probe.turn_identities();
            assert!(
                turns.len() >= 2,
                "a streamed tool run completes at least two model turns, got {}",
                turns.len()
            );
            for (index, turn) in turns.iter().enumerate() {
                assert_request_id(
                    turn.provider_request_id.as_deref(),
                    &format!("streamed turn {index}"),
                );
            }
            assert_ne!(
                turns[0].provider_request_id, turns[1].provider_request_id,
                "each streamed attempt reports its own request id"
            );

            // The tool-only turn fires no StreamResponseFinish; only the
            // final text turn does, and it agrees with that turn's identity.
            let finishes = probe.stream_finish_identities();
            assert_eq!(finishes.len(), 1, "one text turn, one finish event");
            assert_eq!(
                finishes[0].provider_request_id,
                turns.last().expect("turns").provider_request_id
            );

            // The per-call records the stream emitted agree with the hooks.
            assert_eq!(completion_calls.len(), turns.len());
            for (call, turn) in completion_calls.iter().zip(&turns) {
                assert_eq!(call.provider_request_id, turn.provider_request_id);
                assert_eq!(call.message_id, turn.message_id);
            }
        },
    )
    .await;
}
