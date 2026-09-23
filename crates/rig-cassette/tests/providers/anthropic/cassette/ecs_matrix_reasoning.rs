//! The reasoning tool row of the ECS contract matrix on Anthropic
//! (claude-haiku-4-5 with extended thinking): the tool turn's signed thinking
//! is checkpointed with the run after turn one, the world is dropped, and a
//! fresh world restored from the scene sends the continuation live. The
//! strict cassette match proves the restored request carries the signature
//! byte for byte (`tests/common/ecs_matrix/world.rs`).

use rig::completion::CompletionModel;
use rig::driver::Bound;
use rig::providers::anthropic::wire::Anthropic;

use super::super::support::with_anthropic_cassette;
use crate::ecs_matrix::{Wire, cells, world::run_world};

fn reasoning_wire(client: &Bound<Anthropic>) -> Wire<impl CompletionModel + Clone + 'static> {
    Wire {
        thinking: cells::ThinkingWire::Anthropic,
        model: client.completion("claude-haiku-4-5"),
        route: None,
        // Extended thinking requires the default temperature.
        temperature: None,
        additional_params: None,
    }
}

crate::matrix::native_matrix! {
    wrapper: with_anthropic_cassette, wire: reasoning_wire, run: run_world;
    #[tokio::test]
    reasoning_tool_unary: ("reasoning_matrix/tool_unary", cells::REASONING_TOOL_UNARY, "anthropic_reasoning_tool_unary");
}
