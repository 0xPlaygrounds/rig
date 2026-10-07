//! The ECS contract matrix's world-only rows on the OpenAI Chat Completions wire: the batch
//! holds of §8.1 (row 11), a recorded 4xx through `spawn_run` with the
//! witness's facts (row 13) and `despawn_run` on a run cancelled with
//! its stream in flight (row 15). The drivers are in
//! `tests/common/ecs_matrix/extra.rs`; this file holds the scenario
//! literals and the wire's models.

use rig::providers::openai::GPT_4O;

use super::super::support::with_openai_cassette;
use crate::ecs_matrix::extra::{ErrorProbe, error_facts};

/// The recorded 4xx, unary: the run fails as the provider's response and
/// the record and the witness carry the same facts.
#[tokio::test]
async fn error_facts_unary() {
    crate::goldens::capture_world_programs(async {
        with_openai_cassette(
            "error_identity_edge/chat_completions_validation_error_carries_identity",
            |client| async move {
                error_facts(
                    client.openai.chat(GPT_4O),
                    ErrorProbe {
                        prompt: "Never validated",
                        max_tokens: None,
                        additional_params: Some(serde_json::json!({"temperature": 99.0})),
                        options: None,
                        streamed: false,
                        status: 400,
                        code: Some("decimal_above_max_value"),
                    },
                    |log| {
                        crate::goldens::world_golden_effects(
                            "openai_matrix_extra_chat_error_facts_unary",
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
