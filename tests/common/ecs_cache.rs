//! Cache-growth observations from real native completion outcomes in turn order.

use crate::{
    cache_conformance::{
        AGENT_CACHE_PROMPT, CacheObservation, CacheProbeLookupTool, CacheSupport,
        assert_agent_growth_still_hits,
    },
    ecs_agent::EcsAgent,
};
use bevy_ecs::prelude::*;
use rig::effect::{EffectKind, Outcome};
use rig_ecs::{
    agent::{DefaultMaxTurns, Order, Temperature, Turn},
    bus::{EffectOutcome, PendingEffect},
    systems::spawn_run,
};

/// Run the original cache-growth workload and preserve its full neutral validator.
pub async fn assert_cache_growth(mut ecs: EcsAgent, support: &CacheSupport, context: &str) {
    ecs.app
        .world_mut()
        .entity_mut(ecs.agent)
        .insert((DefaultMaxTurns(None), Temperature(Some(0.0))));
    ecs.tool(CacheProbeLookupTool);
    let run = spawn_run(
        ecs.app.world_mut(),
        ecs.agent,
        &[],
        AGENT_CACHE_PROMPT,
        false,
        Some(6),
    );
    ecs.wait_for_success(run).await;
    let world = ecs.app.world_mut();
    let mut usages: Vec<_> = world
        .query::<(&ChildOf, &PendingEffect, &EffectOutcome)>()
        .iter(world)
        .filter_map(|(parent, pending, outcome)| {
            let turn = parent.parent();
            if world.get::<Turn>(turn).is_none() || world.get::<ChildOf>(turn)?.parent() != run {
                return None;
            }
            if !matches!(pending.kind, EffectKind::Completion { .. }) {
                return None;
            }
            let Ok(Outcome::Completion(response)) = &outcome.0 else {
                panic!("cache loop completion must succeed")
            };
            Some((
                world.get::<Order>(turn).expect("turn order").0,
                response.usage,
            ))
        })
        .collect();
    usages.sort_by_key(|(order, _)| *order);
    let observation = CacheObservation {
        turns: usages.into_iter().map(|(_, usage)| usage).collect(),
    };
    assert_agent_growth_still_hits(&observation, support, context);
}
