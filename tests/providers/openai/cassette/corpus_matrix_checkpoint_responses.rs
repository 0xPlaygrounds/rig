//! Focused tool-turn checkpoint matrix on OpenAiResponses: gpt-4.1-mini.
//! One producer recording is reused by every native cut with strict matching.

use super::super::support::{OpenAiCassette, with_openai_cassette};
use crate::ecs_matrix::{Wire, cells, checkpoint};
use rig::completion::CompletionModel;

fn wire(client: &OpenAiCassette) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: cells::ThinkingWire::OpenAiResponses,
        model: client.openai.completion("gpt-4.1-mini"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

crate::matrix::golden_matrix! {
    wrapper: with_openai_cassette, wire: wire, run: checkpoint::run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    multi_turn_unary: ("checkpoint_matrix_responses/multi_turn_unary", checkpoint::MULTI_TURN_UNARY, "openai_responses_checkpoint_multi_turn_unary");
    #[tokio::test]
    multi_turn_streamed: ("checkpoint_matrix_responses/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED, "openai_responses_checkpoint_multi_turn_streamed");
    #[tokio::test]
    parallel_batch: ("checkpoint_matrix_responses/parallel_batch", checkpoint::PARALLEL_BATCH, "openai_responses_checkpoint_parallel_batch");
    #[tokio::test]
    large_result: ("checkpoint_matrix_responses/large_result", checkpoint::LARGE_RESULT, "openai_responses_checkpoint_large_result");
}
