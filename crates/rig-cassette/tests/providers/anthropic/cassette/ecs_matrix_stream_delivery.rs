//! Existing complete Anthropic programs through the shared public-delivery
//! consumers. Their original producer/native cases remain registered separately.
use super::super::support::with_anthropic_cassette;
use crate::ecs_matrix::{Wire, cells, world::run_world};
use rig::completion::CompletionModel;
use rig::driver::Bound;
use rig::providers::anthropic::wire::Anthropic;

fn wire(client: &Bound<Anthropic>) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: cells::ThinkingWire::Anthropic,
        model: client.completion("claude-sonnet-4-6"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

#[tokio::test]
async fn text() {
    crate::goldens::capture_world_programs(async {
        with_anthropic_cassette(
            "corpus_shaping/extra_context_streamed",
            |client| async move {
                run_world(
                    &wire(&client),
                    &cells::SHAPING_EXTRA_CONTEXT_STREAMED,
                    |log| {
                        crate::goldens::world_golden_effects(
                            "anthropic_matrix_stream_delivery_text",
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

#[tokio::test]
async fn parallel() {
    crate::goldens::capture_world_programs(async {

    with_anthropic_cassette("streaming_tools/streaming_tool_concurrency_emits_results_as_completed_but_persists_call_order", |client| async move {
        run_world(&wire(&client), &cells::SERVING_CONCURRENT_CONCURRENCY_TWO_EVENTS, |log| crate::goldens::world_golden_effects("anthropic_matrix_stream_delivery_parallel", log)).await;
    }).await;

}).await
}
