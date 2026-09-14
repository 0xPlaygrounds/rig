//! Native ECS provider completions preserving the original cassette assertions.
use crate::ecs_agent::EcsAgent;
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};
use crate::xai::support::with_xai_cassette;
use rig::prelude::*;
use rig::providers::xai;
use rig_ecs::agent::DefaultMaxTurns;
#[tokio::test]
async fn completion_smoke() {
    with_xai_cassette("agent/completion_smoke", |client| async move {
        let mut ecs = EcsAgent::new(
            client.completion_model(xai::completion::GROK_3_MINI),
            BASIC_PREAMBLE,
            1,
        );
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(DefaultMaxTurns(None));
        let response = ecs.prompt(BASIC_PROMPT, false).await;
        assert_nonempty_response(&response);
    })
    .await;
}
