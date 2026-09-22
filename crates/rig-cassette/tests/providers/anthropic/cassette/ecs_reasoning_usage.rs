//! Reasoning usage from the actual native completion outcome.
use super::reasoning_usage_matrix::{
    Observed, THINKING_PROMPT, budget_thinking, recorded_blocking_thinking_tokens,
};
use crate::ecs_agent::EcsAgent;
use bevy_ecs::prelude::*;
use rig::{effect::Outcome, providers::anthropic};
use rig_ecs::{
    agent::{AdditionalParams, DefaultMaxTurns, MaxTokens, Turn},
    bus::EffectOutcome,
    systems::RunCommands,
};

#[tokio::test]
async fn agent_blocking_thinking() {
    rig_test_support::goldens::world_golden_test(
        async {
            let observed = Observed::new();
            let slot = observed.clone();
            super::super::support::with_anthropic_reasoning_usage_cassette(
                "reasoning_usage_matrix/agent_blocking_thinking",
                |client| async move {
                    let mut ecs = EcsAgent::new(
                        client.completion(anthropic::completion::CLAUDE_SONNET_4_6),
                        "You are a meticulous arithmetic assistant.",
                        1,
                    );
                    ecs.app.world_mut().entity_mut(ecs.agent).insert((
                        DefaultMaxTurns(None),
                        MaxTokens(Some(2048)),
                        AdditionalParams(Some(budget_thinking(1024))),
                    ));
                    let run =
                        ecs.app
                            .world_mut()
                            .spawn_run(ecs.agent, &[], THINKING_PROMPT, false, None);
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
                            Some((
                                crate::ecs_agent::sibling_index(world, turn)?,
                                response.usage,
                            ))
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
        },
        |log| {
            rig_test_support::goldens::world_golden_effects(
                "anthropic_reasoning_usage_agent_blocking_thinking",
                log,
            )
        },
    )
    .await
}
