//! Response identity metadata (rig#2265): Gemini is the documented `None`
//! provider — its live responses carry no request-id response header
//! (verified against `generativelanguage.googleapis.com`), so
//! `provider_request_id` is `None` by design, never an error. These fixtures
//! are the recorded proof of that absence, on both surfaces.

use futures::StreamExt;
use rig::completion::CompletionModel;
use rig::prelude::*;
use rig::providers::gemini;
use rig::streaming::StreamedAssistantContent;

use super::super::support::with_gemini_cassette;

#[tokio::test]
async fn nonstreaming_request_id_is_none_by_design() {
    with_gemini_cassette(
        "response_identity/nonstreaming_request_id_is_none_by_design",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let response = model
                .completion_request("Reply with exactly: identity probe")
                .send()
                .await
                .expect("completion should succeed");

            assert_eq!(
                response.provider_request_id, None,
                "Gemini reports no request-id header; None is the documented outcome"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_request_id_is_none_by_design() {
    with_gemini_cassette(
        "response_identity/streaming_request_id_is_none_by_design",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
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
            assert_eq!(
                terminal.provider_request_id, None,
                "blocking/streaming parity for the None provider"
            );
        },
    )
    .await;
}

/// The documented `None` propagates through the agent surfaces too: hooks and
/// `completion_calls` report `provider_request_id: None` for Gemini while the
/// run itself succeeds — absence is data, never an error.
#[tokio::test]
async fn agent_run_reports_none_identity() {
    use crate::support::IdentityProbe;
    use rig::completion::Prompt;

    with_gemini_cassette(
        "response_identity/agent_run_reports_none_identity",
        |client| async move {
            let probe = IdentityProbe::default();
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble("You are a terse assistant.")
                .add_hook(probe.clone())
                .build();

            let response = agent
                .prompt("Reply with exactly: identity probe")
                .extended_details()
                .await
                .expect("agent run should succeed");

            let turns = probe.turn_identities();
            assert_eq!(turns.len(), 1);
            assert_eq!(turns[0].provider_request_id, None);
            assert_eq!(
                response.completion_calls[0].provider_request_id, None,
                "Gemini reports no request-id header; None is the documented outcome"
            );
        },
    )
    .await;
}

/// Streamed parity for the `None` provider through the agent surfaces.
#[tokio::test]
async fn streamed_agent_run_reports_none_identity() {
    use crate::support::IdentityProbe;

    with_gemini_cassette(
        "response_identity/streamed_agent_run_reports_none_identity",
        |client| async move {
            let probe = IdentityProbe::default();
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
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
            assert_eq!(turns[0].provider_request_id, None);
        },
    )
    .await;
}
