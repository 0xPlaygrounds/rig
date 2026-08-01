//! OpenAI-backed regression coverage for retrying a completed model turn.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rig::agent::ModelTurnAction;
use rig::completion::Message;
use rig::hooks::{HookDecision, HookEntry, HookEvent};
use rig::message::{AssistantContent, UserContent};
use rig::prelude::*;
use rig::providers::openai;

use super::super::support::with_openai_cassette;

/// Retries the first model turn that carries the `RETRY:` marker, then stops if
/// the marker survives a second attempt. The attempt counter is host-owned
/// state captured by the closure (formerly the run-scoped scratchpad).
fn retry_once_on_marker() -> HookEntry {
    let attempts = Arc::new(AtomicUsize::new(0));
    HookEntry::new("retry-once-on-marker", move |event| {
        let HookEvent::ModelTurnFinished { content, .. } = event else {
            return Box::pin(async { HookDecision::Continue });
        };
        let rejected = content.iter().any(|content| {
            matches!(content, AssistantContent::Text(text) if text.text.contains("RETRY:"))
        });
        if !rejected {
            return Box::pin(async { HookDecision::ModelTurn(ModelTurnAction::continue_run()) });
        }

        let attempt = attempts.fetch_add(1, Ordering::SeqCst) + 1;
        let action = if attempt == 1 {
            ModelTurnAction::retry_with_feedback(
                "Replace the rejected response. Reply exactly `ACCEPTED`.",
            )
        } else {
            ModelTurnAction::stop("response retry limit exceeded")
        };
        Box::pin(async move { HookDecision::ModelTurn(action) })
    })
}

#[tokio::test]
async fn rejected_response_is_retried_with_feedback() {
    with_openai_cassette(
        "response_retry/rejected_response_is_retried_with_feedback",
        |client| async move {
            let response = client
                .agent(openai::GPT_4O_MINI)
                .preamble(
                    "Follow this protocol exactly. For the initial request, reply exactly \
                 `RETRY: incomplete draft`. If the latest user message asks you to \
                 replace the rejected response, reply exactly `ACCEPTED`.",
                )
                .temperature(0.0)
                .build()
                .runner("Begin the retry-hook demonstration.")
                .max_turns(2)
                .add_hook(retry_once_on_marker())
                .run()
                .await
                .expect("the feedback retry should recover");

            assert_eq!(response.output.trim(), "ACCEPTED");
            assert_eq!(response.completion_calls.len(), 2);
            assert!(response.usage.input_tokens > 0);
            assert!(response.usage.output_tokens > 0);
            let transcript = response
                .messages
                .expect("response history")
                .into_iter()
                .map(|message| match message {
                    Message::System { content } => ("system", content),
                    Message::User { content } => (
                        "user",
                        content
                            .iter()
                            .filter_map(|content| match content {
                                UserContent::Text(text) => Some(text.text.as_str()),
                                _ => None,
                            })
                            .collect::<String>(),
                    ),
                    Message::Assistant { content, .. } => (
                        "assistant",
                        content
                            .iter()
                            .filter_map(|content| match content {
                                AssistantContent::Text(text) => Some(text.text.as_str()),
                                _ => None,
                            })
                            .collect::<String>(),
                    ),
                })
                .collect::<Vec<_>>();
            assert_eq!(
                transcript,
                vec![
                    ("user", "Begin the retry-hook demonstration.".to_string()),
                    ("assistant", "RETRY: incomplete draft".to_string()),
                    (
                        "user",
                        "Replace the rejected response. Reply exactly `ACCEPTED`.".to_string(),
                    ),
                    ("assistant", "ACCEPTED".to_string()),
                ]
            );
        },
    )
    .await;
}
