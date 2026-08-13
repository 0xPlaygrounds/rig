//! Response identity metadata (rig#2265): OpenAI reports `x-request-id` on
//! both the Responses and Chat Completions APIs; blocking and streaming turns
//! carry it identically.

use futures::StreamExt;
use rig::completion::CompletionModel;
use rig::prelude::*;
use rig::providers::openai;
use rig::streaming::StreamedAssistantContent;

use super::super::support::{with_openai_cassette, with_openai_completions_cassette};

fn assert_request_id(id: Option<&str>, context: &str) {
    assert!(
        id.is_some_and(|id| !id.trim().is_empty()),
        "{context}: OpenAI reports an `x-request-id` response header, so \
         provider_request_id must be populated"
    );
}

#[tokio::test]
async fn responses_nonstreaming_carries_identity() {
    with_openai_cassette(
        "response_identity/responses_nonstreaming_carries_identity",
        |client| async move {
            let model = client.completion_model(openai::GPT_4O);
            let response = model
                .completion_request("Reply with exactly: identity probe")
                .send()
                .await
                .expect("completion should succeed");

            assert!(
                response
                    .response_id
                    .as_deref()
                    .is_some_and(|id| id.starts_with("resp")),
                "Responses API reports resp_ ids, got {:?}",
                response.response_id
            );
            assert_request_id(
                response.provider_request_id.as_deref(),
                "responses blocking",
            );
        },
    )
    .await;
}

#[tokio::test]
async fn responses_streaming_carries_identity() {
    with_openai_cassette(
        "response_identity/responses_streaming_carries_identity",
        |client| async move {
            let model = client.completion_model(openai::GPT_4O);
            let mut stream = model
                .completion_request("Reply with exactly: stream identity probe")
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
            assert_request_id(
                terminal.provider_request_id.as_deref(),
                "responses streaming terminal",
            );
        },
    )
    .await;
}

#[tokio::test]
async fn chat_completions_nonstreaming_carries_identity() {
    with_openai_completions_cassette(
        "response_identity/chat_completions_nonstreaming_carries_identity",
        |client| async move {
            let model = client.completion_model(openai::GPT_4O);
            let response = model
                .completion_request("Reply with exactly: identity probe")
                .send()
                .await
                .expect("completion should succeed");

            assert!(
                response
                    .response_id
                    .as_deref()
                    .is_some_and(|id| id.starts_with("chatcmpl")),
                "Chat Completions reports chatcmpl- ids, got {:?}",
                response.response_id
            );
            assert_request_id(response.provider_request_id.as_deref(), "chat blocking");
        },
    )
    .await;
}

#[tokio::test]
async fn chat_completions_streaming_carries_identity() {
    with_openai_completions_cassette(
        "response_identity/chat_completions_streaming_carries_identity",
        |client| async move {
            let model = client.completion_model(openai::GPT_4O);
            let mut stream = model
                .completion_request("Reply with exactly: stream identity probe")
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
            assert_request_id(
                terminal.provider_request_id.as_deref(),
                "chat streaming terminal",
            );
        },
    )
    .await;
}

/// Agent-run reachability on OpenAI (Responses API): a two-call tool run's
/// hooks and `completion_calls` report distinct per-attempt request ids.
#[tokio::test]
async fn agent_tool_run_reports_per_attempt_identity() {
    use crate::support::{Adder, IdentityProbe, TOOLS_PREAMBLE};
    use rig::completion::Prompt;

    with_openai_cassette(
        "response_identity/agent_tool_run_reports_per_attempt_identity",
        |client| async move {
            let probe = IdentityProbe::default();
            let agent = client
                .agent(openai::GPT_4O)
                .preamble(TOOLS_PREAMBLE)
                .tool(Adder)
                .add_hook(probe.clone())
                .build();

            let response = agent
                .prompt("What is 2 + 3? Use the tool, then state the result.")
                .max_turns(3)
                .extended_details()
                .await
                .expect("agent run should succeed");

            let turns = probe.turn_identities();
            assert!(turns.len() >= 2, "tool run makes at least two calls");
            for turn in &turns {
                assert_request_id(turn.provider_request_id.as_deref(), "turn identity");
            }
            assert_ne!(turns[0].provider_request_id, turns[1].provider_request_id);
            let calls = &response.completion_calls;
            assert_eq!(calls.len(), turns.len());
            for (call, turn) in calls.iter().zip(&turns) {
                assert_eq!(call.provider_request_id, turn.provider_request_id);
                assert_eq!(call.response_id, turn.response_id);
            }
        },
    )
    .await;
}

/// Streamed agent run on OpenAI: the turn event carries the attempt's
/// identity from the SSE connection's headers.
#[tokio::test]
async fn streamed_agent_run_reports_identity() {
    use crate::support::IdentityProbe;
    use futures::StreamExt as _;

    with_openai_cassette(
        "response_identity/streamed_agent_run_reports_identity",
        |client| async move {
            let probe = IdentityProbe::default();
            let agent = client
                .agent(openai::GPT_4O)
                .preamble("You are a terse assistant.")
                .add_hook(probe.clone())
                .build();

            let mut stream = rig::streaming::StreamingPrompt::stream_prompt(
                &agent,
                rig::completion::Message::user("Reply with exactly: streamed identity probe"),
            )
            .await;
            while let Some(item) = stream.next().await {
                item.expect("stream item should succeed");
            }

            let turns = probe.turn_identities();
            assert_eq!(turns.len(), 1);
            assert_request_id(turns[0].provider_request_id.as_deref(), "streamed turn");
            let finishes = probe.stream_finish_identities();
            assert_eq!(finishes.len(), 1);
            assert_eq!(finishes[0], turns[0]);
        },
    )
    .await;
}
