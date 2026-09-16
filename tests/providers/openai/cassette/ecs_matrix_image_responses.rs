//! The image matrix on the OpenAI Responses wire (`gpt-5-mini`, no temperature: the gpt-5 family takes only its default): the image cells of
//! `tests/common/ecs_matrix/cells.rs` as agent graphs in a Bevy `World`,
//! served by the real adapter over the same recording as their producers in
//! `corpus_matrix_image*.rs` and asserted against those producers' goldens,
//! then by their history, their live-resumed cut and their despawn (the
//! driver is `tests/common/ecs_matrix/world.rs`). This file holds the
//! scenario literals, the wire's model and the wire's `#[ignore]` reasons.

use rig::completion::CompletionModel;
use rig::providers::openai::GPT_5_MINI;

use super::super::support::{OpenAiCassette, with_openai_cassette};
use crate::ecs_matrix::{Wire, cells, world::run_world};

fn wire(client: &OpenAiCassette) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: cells::ThinkingWire::OpenAiResponses,
        model: client.openai.completion(GPT_5_MINI),
        route: None,
        temperature: None,
        additional_params: None,
    }
}

crate::matrix::golden_matrix! {
    wrapper: with_openai_cassette, wire: wire, run: run_world, oracle: crate::ecs_goldens::golden_effects;
    #[tokio::test]
    inline_text_unary: ("image_matrix_responses/inline_text_unary", cells::IMAGE_INLINE_TEXT_UNARY, "openai_responses_image_inline_text_unary");
    #[tokio::test]
    inline_text_streamed: ("image_matrix_responses/inline_text_streamed", cells::IMAGE_INLINE_TEXT_STREAMED, "openai_responses_image_inline_text_streamed");
    #[tokio::test]
    inline_mixed_order: ("image_matrix_responses/inline_mixed_order", cells::IMAGE_INLINE_MIXED_ORDER, "openai_responses_image_inline_mixed_order");
    #[tokio::test]
    inline_tool_unary: ("image_matrix_responses/inline_tool_unary", cells::IMAGE_INLINE_TOOL_UNARY, "openai_responses_image_inline_tool_unary");
    #[tokio::test]
    inline_tool_streamed: ("image_matrix_responses/inline_tool_streamed", cells::IMAGE_INLINE_TOOL_STREAMED, "openai_responses_image_inline_tool_streamed");
    #[tokio::test]
    inline_followup: ("image_matrix_responses/inline_followup", cells::IMAGE_INLINE_FOLLOWUP, "openai_responses_image_inline_followup");
    #[tokio::test]
    url_text_unary: ("image_matrix_responses/url_text_unary", cells::IMAGE_URL_TEXT_UNARY, "openai_responses_image_url_text_unary");
    #[tokio::test]
    url_tool_unary: ("image_matrix_responses/url_tool_unary", cells::IMAGE_URL_TOOL_UNARY, "openai_responses_image_url_tool_unary");
}
