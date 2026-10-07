//! Focused tool-turn checkpoint matrix on OpenAiChat: gpt-4.1-mini.
//! One producer recording is reused by every native cut with strict matching.

use super::super::support::{OpenAiCassette, with_openai_cassette};
use crate::ecs_matrix::{Wire, cells, checkpoint};

fn wire(client: &OpenAiCassette) -> Wire<rig::Model<rig::providers::openai::wire::Chat>> {
    Wire {
        thinking: cells::ThinkingWire::OpenAiChat,
        model: client.openai.chat("gpt-4.1-mini"),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
        options: None,
    }
}

crate::matrix::golden_matrix! {
    wrapper: with_openai_cassette, wire: wire, run: checkpoint::run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    parallel_batch: ("checkpoint_matrix_chat/parallel_batch", checkpoint::PARALLEL_BATCH, "openai_chat_checkpoint_parallel_batch");
}
