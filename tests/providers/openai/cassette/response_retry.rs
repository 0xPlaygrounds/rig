//! OpenAI-backed regression coverage for retrying a completed model turn.

use rig::agent::{AgentHook, HookContext, ModelTurnAction, ModelTurnFinished};
use rig::client::AgentClientExt;
use rig::completion::Message;
use rig::message::{AssistantContent, UserContent};
use rig::providers::openai;

use super::super::support::with_openai_cassette;

#[derive(Clone, Default)]
struct RetryAttempts(usize);

struct RetryOnceOnMarker;

impl AgentHook for RetryOnceOnMarker {
    async fn on_model_turn_finished(
        &self,
        ctx: &HookContext,
        event: ModelTurnFinished<'_>,
    ) -> ModelTurnAction {
        let rejected = event.content.iter().any(|content| {
            matches!(content, AssistantContent::Text(text) if text.text.contains("RETRY:"))
        });
        if !rejected {
            return ModelTurnAction::continue_run();
        }

        let attempt = ctx.scratchpad().update(|attempts: &mut RetryAttempts| {
            attempts.0 += 1;
            attempts.0
        });
        if attempt == 1 {
            ModelTurnAction::retry_with_feedback(
                "Replace the rejected response. Reply exactly `ACCEPTED`.",
            )
        } else {
            ModelTurnAction::stop("response retry limit exceeded")
        }
    }
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
                .add_hook(RetryOnceOnMarker)
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
