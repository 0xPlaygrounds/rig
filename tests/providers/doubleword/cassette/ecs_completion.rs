//! Native ECS provider completions preserving the original cassette assertions.
use crate::doubleword::DEFAULT_MODEL;
use crate::doubleword::support::with_doubleword_cassette;
use crate::ecs_agent::EcsAgent;
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};
use rig::prelude::*;
use rig_ecs::agent::DefaultMaxTurns;
#[tokio::test]
async fn completion_smoke() {
    with_doubleword_cassette("agent/completion_smoke", |client| async move {
        let mut ecs = EcsAgent::new(client.completion_model(DEFAULT_MODEL), BASIC_PREAMBLE, 1);
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert(DefaultMaxTurns(None));
        let response = ecs.prompt(BASIC_PROMPT, false).await;
        assert_nonempty_response(&response);
    })
    .await;
}
