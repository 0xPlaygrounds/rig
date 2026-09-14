//! Reasoning usage from the actual native completion outcome.
use super::reasoning_usage_matrix::{
    Observed, THINKING_PROMPT, budget_thinking, recorded_blocking_thinking_tokens,
};
use crate::ecs_agent::EcsAgent;
use bevy_ecs::prelude::*;
use rig::{effect::Outcome, prelude::*, providers::anthropic};
use rig_ecs::{
    agent::{AdditionalParams, DefaultMaxTurns, MaxTokens, Order, Turn},
    bus::EffectOutcome,
    systems::spawn_run,
};

#[tokio::test]
async fn agent_blocking_thinking() {
    let observed = Observed::new();
    let slot = observed.clone();
    super::super::support::with_anthropic_reasoning_usage_cassette(
        "reasoning_usage_matrix/agent_blocking_thinking",
        |client| async move {
            let mut ecs = EcsAgent::new(
                client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6),
                "You are a meticulous arithmetic assistant.",
                1,
            );
            ecs.app.world_mut().entity_mut(ecs.agent).insert((
                DefaultMaxTurns(None),
                MaxTokens(Some(2048)),
                AdditionalParams(Some(budget_thinking(1024))),
            ));
            let run = spawn_run(
                ecs.app.world_mut(),
                ecs.agent,
                &[],
                THINKING_PROMPT,
                false,
                None,
            );
            ecs.wait_for_success(run).await;
            let world = ecs.app.world_mut();
            let usage = world
                .query::<(&ChildOf, &EffectOutcome)>()
                .iter(world)
                .filter_map(|(parent, outcome)| {
                    let turn = parent.parent();
                    if world.get::<Turn>(turn).is_none()
                        || world.get::<ChildOf>(turn)?.parent() != run
                    {
                        return None;
                    }
                    let Ok(Outcome::Completion(response)) = &outcome.0 else {
                        return None;
                    };
                    Some((world.get::<Order>(turn)?.0, response.usage))
                })
                .min_by_key(|(order, _)| *order)
                .expect("the run makes at least one completion call")
                .1;
            slot.record(usage);
        },
    )
    .await;
    let scenario = "reasoning_usage_matrix/agent_blocking_thinking";
    observed.assert_matches(scenario, recorded_blocking_thinking_tokens(scenario));
}
