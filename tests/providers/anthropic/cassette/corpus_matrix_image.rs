//! The image matrix on the Anthropic Messages wire (`claude-sonnet-4-6`, the existing image test's model): the rig-agent producers of the image cells of
//! `tests/common/ecs_matrix/cells.rs` over the shared driver
//! (`tests/common/ecs_matrix/agent.rs`), their logs written as the goldens
//! the world cells (`ecs_matrix_image*.rs`) are compared to. This file holds
//! the scenario literals, the wire's model and the wire's `#[ignore]` reasons.

use rig::completion::CompletionModel;
use rig::driver::Bound;
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;
use rig::providers::anthropic::wire::Anthropic;

use super::super::support::with_anthropic_cassette;
use crate::ecs_matrix::{Wire, agent::run_agent, cells};

fn wire(client: &Bound<Anthropic>) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: cells::ThinkingWire::Anthropic,
        model: client.completion(CLAUDE_SONNET_4_6),
        route: None,
        temperature: Some(0.0),
        additional_params: None,
    }
}

crate::matrix::golden_matrix! {
    wrapper: with_anthropic_cassette, wire: wire, run: run_agent, oracle: crate::goldens::golden_effects;
    #[tokio::test]
    inline_text_unary: ("image_matrix/inline_text_unary", cells::IMAGE_INLINE_TEXT_UNARY, "anthropic_image_inline_text_unary");
    #[tokio::test]
    inline_text_streamed: ("image_matrix/inline_text_streamed", cells::IMAGE_INLINE_TEXT_STREAMED, "anthropic_image_inline_text_streamed");
    #[tokio::test]
    inline_mixed_order: ("image_matrix/inline_mixed_order", cells::IMAGE_INLINE_MIXED_ORDER, "anthropic_image_inline_mixed_order");
    #[tokio::test]
    inline_tool_unary: ("image_matrix/inline_tool_unary", cells::IMAGE_INLINE_TOOL_UNARY, "anthropic_image_inline_tool_unary");
    #[tokio::test]
    inline_tool_streamed: ("image_matrix/inline_tool_streamed", cells::IMAGE_INLINE_TOOL_STREAMED, "anthropic_image_inline_tool_streamed");
    #[tokio::test]
    inline_followup: ("image_matrix/inline_followup", cells::IMAGE_INLINE_FOLLOWUP, "anthropic_image_inline_followup");
    #[tokio::test]
    url_text_unary: ("image_matrix/url_text_unary", cells::IMAGE_URL_TEXT_UNARY, "anthropic_image_url_text_unary");
    #[tokio::test]
    url_tool_unary: ("image_matrix/url_tool_unary", cells::IMAGE_URL_TOOL_UNARY, "anthropic_image_url_tool_unary");
}
