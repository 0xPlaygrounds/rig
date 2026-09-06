//! Native ECS provider completions preserving the original cassette assertions.
//!
//! | Cell | Contract |
//! | --- | --- |
//! | `completion_smoke` | Native prompt returns nonempty text from the original default-server recording. |
use crate::ecs_agent::EcsAgent;
use crate::llamacpp::cassette_support::CASSETTE_MODEL;
use crate::llamacpp::cassette_support::with_llamacpp_cassette;
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};
use rig::prelude::*;
use rig_ecs::agent::DefaultMaxTurns;
#[tokio::test]
async fn completion_smoke() {
    with_llamacpp_cassette("agent/completion_smoke", |client| async move {
        let mut ecs = EcsAgent::new(client.completion_model(CASSETTE_MODEL), BASIC_PREAMBLE, 1);
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(DefaultMaxTurns(None));
        let response = ecs.prompt(BASIC_PROMPT, false).await;
        assert_nonempty_response(&response);
    })
    .await;
}
