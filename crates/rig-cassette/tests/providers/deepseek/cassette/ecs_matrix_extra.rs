//! The ECS contract matrix's world-only rows on the DeepSeek wire: the batch
//! holds of §8.1 (row 11), a recorded 4xx through `spawn_run` with the
//! witness's facts (row 13) and `despawn_run` on a run cancelled with
//! its stream in flight (row 15). The drivers are in
//! `tests/common/ecs_matrix/extra.rs`; this file holds the scenario
//! literals and the wire's models.

use rig::completion::CompletionModel;

use crate::deepseek::support::{BoundDeepSeek, with_deepseek_cassette};
use crate::ecs_matrix::{
    Wire, cells,
    extra::{Approval, ErrorProbe, batch_hold, despawn_waits_for_the_stream, error_facts},
};

fn wire(client: &BoundDeepSeek) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: crate::ecs_matrix::cells::ThinkingWire::DeepSeek,
        model: client.completion("deepseek-chat"),
        route: Some(client.completion("deepseek-reasoner")),
        temperature: Some(0.0),
        additional_params: None,
    }
}

crate::matrix::case_matrix! {
    wrapper: with_deepseek_cassette, family: wire_matrix_case;
    /// A batch-held call approved by removing `Held` dispatches, lands in call
    /// order, and the scene saved afterwards loads.
    #[tokio::test]
    batch_held_call_approved_by_removing_held: ("corpus_matrix/serving_concurrent_concurrency_one", batch_held_call_approved_by_removing_held_15, "deepseek_matrix_extra_batch_held_call_approved_by_removing_held");
    /// The same, approved by `release_hold("rig-ecs/batch")`.
    #[tokio::test]
    batch_held_call_approved_by_releasing_the_batch_owner: ("corpus_matrix/serving_concurrent_concurrency_one", batch_held_call_approved_by_releasing_the_batch_owner_16, "deepseek_matrix_extra_batch_held_call_approved_by_releasing_the_batch_owner");
    /// A run cancelled while its completion still streams refuses
    /// `despawn_run` until the stream drains.
    #[tokio::test]
    despawn_run_waits_for_an_in_flight_stream: ("corpus_matrix/endings_text_delta_stop", despawn_run_waits_for_an_in_flight_stream_22, "deepseek_matrix_extra_despawn_run_waits_for_an_in_flight_stream");
}

/// The recorded 4xx, unary: the run fails as the provider's response and
/// the record and the witness carry the same facts.
#[tokio::test]
async fn error_facts_unary() {
    crate::goldens::capture_world_programs(async {
        with_deepseek_cassette(
            "wire_shape_matrix/chat_completion_rejects_an_unknown_model_with_the_provider_body",
            |client| async move {
                error_facts(
                    client.completion("deepseek-v9-nonexistent"),
                    ErrorProbe {
                        prompt: "hi",
                        max_tokens: Some(8),
                        additional_params: Some(
                            serde_json::json!({ "thinking": { "type": "disabled" } }),
                        ),
                        streamed: false,
                        status: 400,
                        code: Some("invalid_request_error"),
                    },
                    |log| {
                        crate::goldens::world_golden_effects(
                            "deepseek_matrix_extra_error_facts_unary",
                            log,
                        )
                    },
                )
                .await;
            },
        )
        .await;
    })
    .await
}

/// The recorded 4xx on the streaming surface.
#[tokio::test]
async fn error_facts_streamed() {
    crate::goldens::capture_world_programs(async {
        with_deepseek_cassette("corpus_matrix/error_facts_streamed", |client| async move {
            error_facts(
                client.completion("deepseek-v9-nonexistent"),
                ErrorProbe {
                    prompt: "hi",
                    max_tokens: Some(8),
                    additional_params: Some(
                        serde_json::json!({ "thinking": { "type": "disabled" } }),
                    ),
                    streamed: true,
                    status: 400,
                    code: Some("invalid_request_error"),
                },
                |log| {
                    crate::goldens::world_golden_effects(
                        "deepseek_matrix_extra_error_facts_streamed",
                        log,
                    )
                },
            )
            .await;
        })
        .await;
    })
    .await
}
