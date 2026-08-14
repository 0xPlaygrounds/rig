//! Adversarial response-identity coverage on OpenAI (rig#2265 / PR #2313
//! follow-up): structured output, response chaining, live hook retries, error
//! responses, and raw-vs-normalized agreement.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use rig::agent::{AgentHook, HookContext, ModelTurnAction, ModelTurnFinished};
use rig::completion::{CompletionModel, Prompt};
use rig::prelude::*;
use rig::providers::openai;

use super::super::support::with_openai_cassette;
use crate::support::{IdentityProbe, assert_transport_request_id};

/// Family A: `output_schema` reshapes the request (structured output);
/// identity still rides it.
#[tokio::test]
async fn structured_output_and_identity() {
    #[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
    struct Sum {
        value: i64,
    }

    with_openai_cassette(
        "response_identity_edge/structured_output_and_identity",
        |client| async move {
            let model = client.completion_model(openai::GPT_4O);
            let schema = schemars::schema_for!(Sum);
            let response = model
                .completion_request("What is 2 + 3? Respond with the JSON object.")
                .output_schema(schema)
                .send()
                .await
                .expect("structured completion should succeed");
            assert_transport_request_id(
                response.provider_request_id.as_deref(),
                "structured-output response",
            );
        },
    )
    .await;
}

/// Family A: a `previous_response_id` chain — exactly where response-scoped
/// and transport ids are most likely to be crossed. The second call reuses
/// the first's *response id* on the wire, yet reports its own transport id.
#[tokio::test]
async fn previous_response_id_chain_keeps_axes_distinct() {
    with_openai_cassette(
        "response_identity_edge/previous_response_id_chain_keeps_axes_distinct",
        |client| async move {
            let model = client.completion_model(openai::GPT_4O);
            let first = model
                .completion_request(
                    "Remember the code word 'heliotrope'. Reply with exactly: noted",
                )
                .send()
                .await
                .expect("first chained call should succeed");
            let first_response_id = first
                .response_id
                .clone()
                .expect("Responses API reports a response id");
            assert_transport_request_id(first.provider_request_id.as_deref(), "chain call 1");

            let second = model
                .completion_request("What was the code word? Reply with just the word.")
                .additional_params(serde_json::json!({
                    "previous_response_id": first_response_id.clone(),
                }))
                .send()
                .await
                .expect("chained call should succeed");

            assert_transport_request_id(second.provider_request_id.as_deref(), "chain call 2");
            assert_ne!(
                first.provider_request_id, second.provider_request_id,
                "each chained call has its own transport id"
            );
            let second_response_id = second.response_id.expect("second response id");
            assert_ne!(
                first_response_id, second_response_id,
                "chaining reuses the first response id as *input*; the second \
                 response still gets its own"
            );
            assert_ne!(
                Some(second_response_id.as_str()),
                second.provider_request_id.as_deref(),
                "response-scoped and transport ids are never conflated"
            );
        },
    )
    .await;
}

/// Family B: a live hook-driven retry on the blocking surface — the second
/// event's identity is the second attempt's.
#[tokio::test]
async fn blocking_hook_retry_uses_second_attempts_id() {
    #[derive(Clone, Default)]
    struct RetryOnce {
        probe: IdentityProbe,
        retried: Arc<AtomicBool>,
    }

    impl AgentHook for RetryOnce {
        async fn on_model_turn_finished(
            &self,
            ctx: &HookContext,
            event: ModelTurnFinished<'_>,
        ) -> ModelTurnAction {
            let action = if !self.retried.swap(true, Ordering::SeqCst) {
                ModelTurnAction::retry_with_feedback("Answer again with exactly: retried probe")
            } else {
                ModelTurnAction::continue_run()
            };
            let _ = self.probe.on_model_turn_finished(ctx, event).await;
            action
        }
    }

    with_openai_cassette(
        "response_identity_edge/blocking_hook_retry_uses_second_attempts_id",
        |client| async move {
            let hook = RetryOnce::default();
            let agent = client
                .agent(openai::GPT_4O)
                .preamble("You are a terse assistant.")
                .add_hook(hook.clone())
                .build();

            agent
                .prompt("Reply with exactly: first probe")
                .max_turns(3)
                .await
                .expect("retried run should succeed");

            let turns = hook.probe.turn_identities();
            assert_eq!(turns.len(), 2, "rejected attempt plus its retry");
            assert_transport_request_id(turns[0].provider_request_id.as_deref(), "attempt 1");
            assert_transport_request_id(turns[1].provider_request_id.as_deref(), "attempt 2");
            assert_ne!(turns[0].provider_request_id, turns[1].provider_request_id);
        },
    )
    .await;
}

/// A provider 4xx carries the failed call's transport request id (rig#2314):
/// the recorded error response's `x-request-id` header reaches the error's
/// `provider_request_id()` accessor, alongside the preserved status and body.
#[tokio::test]
async fn provider_error_response_carries_request_id() {
    with_openai_cassette(
        "response_identity_edge/provider_error_response_surfaces_cleanly",
        |client| async move {
            let model = client.completion_model("gpt-nonexistent-model-for-identity-edge");
            let error = model
                .completion_request("Never answered")
                .send()
                .await
                .expect_err("a nonexistent model must fail");
            assert_transport_request_id(error.provider_request_id(), "4xx error");
            assert!(error.provider_response_status().is_some());
            assert!(
                error.to_string().contains("request id:"),
                "the id appears in the logged message: {error}"
            );
        },
    )
    .await;
}

/// Family D: one interaction, two views — the raw Responses wire value and
/// its normalized form agree on the transport id.
#[tokio::test]
async fn raw_and_normalized_views_agree_on_identity() {
    use rig::completion::NormalizeCompletionResponse;

    with_openai_cassette(
        "response_identity_edge/raw_and_normalized_views_agree_on_identity",
        |client| async move {
            let model = client.completion_model(openai::GPT_4O);
            let request = model
                .completion_request("Reply with exactly: two views probe")
                .build();
            let raw = model
                .raw_completion(request)
                .await
                .expect("raw completion should succeed");
            let raw_id = raw.provider_request_id.clone();
            assert_transport_request_id(raw_id.as_deref(), "raw view");

            let normalized = raw
                .normalize("openai")
                .expect("raw response should normalize");
            assert_eq!(
                normalized.provider_request_id, raw_id,
                "raw and normalized views describe the same interaction"
            );
        },
    )
    .await;
}
