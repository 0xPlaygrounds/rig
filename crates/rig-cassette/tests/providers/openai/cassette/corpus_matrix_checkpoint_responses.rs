//! Focused tool-turn checkpoint matrix on OpenAiResponses: gpt-4.1-mini.

use super::super::support::{OpenAiCassette, with_openai_cassette};
use crate::corpus_matrix::{Wire, cells, checkpoint};

fn wire(client: &OpenAiCassette) -> Wire<rig::Model<rig::providers::openai::wire::OpenAiWire>> {
    Wire {
        thinking: cells::ThinkingWire::OpenAiResponses,
        model: client.openai.completion("gpt-4.1-mini"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
        options: None,
    }
}

crate::matrix::golden_matrix! {
    wrapper: with_openai_cassette, wire: wire, run: checkpoint::run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    multi_turn_streamed: ("checkpoint_matrix_responses/multi_turn_streamed", checkpoint::MULTI_TURN_STREAMED, "openai_responses_checkpoint_multi_turn_streamed");
    #[tokio::test]
    parallel_batch: ("checkpoint_matrix_responses/parallel_batch", checkpoint::PARALLEL_BATCH, "openai_responses_checkpoint_parallel_batch");
}
