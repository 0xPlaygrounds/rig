//! The image matrix on the OpenAI Responses wire (`gpt-5-mini`, no temperature: the gpt-5 family takes only its default): the rig-agent producers of the image cells of
//! `tests/common/corpus_matrix/cells.rs` over the shared driver
//! (`tests/common/corpus_matrix/agent.rs`), their logs written as the
//! goldens. This file holds
//! the scenario literals, the wire's model and the wire's `#[ignore]` reasons.

use rig::providers::openai::GPT_5_MINI;

use super::super::support::{OpenAiCassette, with_openai_cassette};
use crate::corpus_matrix::{Wire, agent::run_agent, cells};

fn wire(client: &OpenAiCassette) -> Wire<rig::Model<rig::providers::openai::wire::OpenAiWire>> {
    Wire {
        thinking: cells::ThinkingWire::OpenAiResponses,
        model: client.openai.completion(GPT_5_MINI),
        route: None,
        temperature: None,
        additional_params: Some(crate::corpus_matrix::cells::openai_responses_stateless),
        options: None,
    }
}

crate::matrix::golden_matrix! {
    wrapper: with_openai_cassette, wire: wire, run: run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    inline_text_streamed: ("image_matrix_responses/inline_text_streamed", cells::IMAGE_INLINE_TEXT_STREAMED, "openai_responses_image_inline_text_streamed");
    #[tokio::test]
    url_text_unary: ("image_matrix_responses/url_text_unary", cells::IMAGE_URL_TEXT_UNARY, "openai_responses_image_url_text_unary");
}
