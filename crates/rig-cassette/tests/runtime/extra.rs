//! The matrix's extra world rows (`tests/common/ecs_matrix/extra.rs`) once:
//! a batch-held call approved either way, a despawn that waits for an
//! in-flight stream, and a rejection's error facts, each on one wire over
//! the bank, with every assertion the driver makes.

use rig_test_support::bank;

use crate::ecs_matrix::{
    cells,
    extra::{Approval, ErrorProbe, batch_hold, despawn_waits_for_the_stream, error_facts},
};
use crate::goldens::capture_world_programs;

#[tokio::test]
async fn batch_held_call_approved_by_removing_held() {
    let replies = bank::script("gemini", "corpus_matrix/serving_concurrent_concurrency_one");
    let wire = crate::wires::gemini(bank::client(&replies));
    capture_world_programs(batch_hold(&wire, Approval::RemoveHeld, |_| {})).await;
}

#[tokio::test]
async fn batch_held_call_approved_by_releasing_the_batch_owner() {
    let replies = bank::script(
        "openai",
        "corpus_matrix_responses/serving_concurrent_concurrency_one",
    );
    let wire = crate::wires::openai_responses(bank::client(&replies));
    capture_world_programs(batch_hold(&wire, Approval::ReleaseBatchOwner, |_| {})).await;
}

#[tokio::test]
async fn despawn_run_waits_for_an_in_flight_stream() {
    let replies = bank::script("doubleword", "corpus_matrix/endings_text_delta_stop");
    let wire = crate::wires::doubleword(bank::client(&replies));
    capture_world_programs(despawn_waits_for_the_stream(
        &wire,
        &cells::ENDINGS_TEXT_DELTA_STOP,
        |_| {},
    ))
    .await;
}

async fn deepseek_error_facts(scenario: &str, streamed: bool) {
    let replies = bank::script("deepseek", scenario);
    let wire = crate::wires::deepseek(bank::client(&replies));
    capture_world_programs(error_facts(
        wire.model,
        ErrorProbe {
            prompt: "hi",
            max_tokens: Some(8),
            additional_params: Some(serde_json::json!({ "thinking": { "type": "disabled" } })),
            options: None,
            streamed,
            status: 400,
            code: Some("invalid_request_error"),
        },
        |_| {},
    ))
    .await;
}

#[tokio::test]
async fn error_facts_unary() {
    deepseek_error_facts(
        "wire_shape_matrix/chat_completion_rejects_an_unknown_model_with_the_provider_body",
        false,
    )
    .await;
}

#[tokio::test]
async fn error_facts_streamed() {
    deepseek_error_facts("corpus_matrix/error_facts_streamed", true).await;
}
