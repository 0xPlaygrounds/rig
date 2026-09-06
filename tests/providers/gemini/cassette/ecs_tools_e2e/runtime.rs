//! Actual native history, completion count and aggregate run usage.

use bevy_ecs::prelude::*;
use rig::{completion::CompletionModel, effect::Outcome, message::Message};
use rig_ecs::{
    agent::{DefaultMaxTurns, Order, Parts, Run, Temperature, ToolPolicy, Usage, Utterance},
    bus::EffectOutcome,
    systems::spawn_run,
};

use crate::ecs_agent::EcsAgent;

pub(super) struct NativeResponse {
    pub output: String,
    pub messages: Option<Vec<Message>>,
    pub usage: rig::completion::Usage,
    requests: usize,
}

impl NativeResponse {
    pub(super) fn requests(&self) -> usize {
        self.requests
    }
}

pub(super) fn configured(
    model: impl CompletionModel + 'static,
    preamble: &str,
    default_max_turns: Option<usize>,
) -> EcsAgent {
    let mut ecs = EcsAgent::new(model, preamble, default_max_turns.unwrap_or(1));
    ecs.app
        .world_mut()
        .entity_mut(ecs.agent)
        .insert((DefaultMaxTurns(default_max_turns), Temperature(Some(0.0))));
    ecs
}

pub(super) async fn execute(
    ecs: &mut EcsAgent,
    prompt: &str,
    streamed: bool,
    max_turns: Option<usize>,
    tool_concurrency: Option<usize>,
) -> NativeResponse {
    let run = spawn_run(
        ecs.app.world_mut(),
        ecs.agent,
        &[],
        prompt,
        streamed,
        max_turns,
    );
    if let Some(concurrency) = tool_concurrency {
        ecs.app
            .world_mut()
            .entity_mut(run)
            .insert(ToolPolicy { concurrency });
    }
    let output = ecs.wait_for_success(run).await;
    assert_eq!(
        ecs.app
            .world_mut()
            .query_filtered::<Entity, With<Run>>()
            .iter(ecs.app.world())
            .count(),
        1,
        "completion observations below are scoped to a fresh one-run app"
    );
    let requests = ecs
        .app
        .world_mut()
        .query::<&EffectOutcome>()
        .iter(ecs.app.world())
        .filter(|outcome| matches!(outcome.0, Ok(Outcome::Completion(_))))
        .count();
    let usage = ecs
        .app
        .world()
        .get::<Usage>(run)
        .expect("run tracks aggregate usage")
        .0;
    let mut messages: Vec<_> = ecs
        .app
        .world_mut()
        .query_filtered::<(&ChildOf, &Order, &Parts), With<Utterance>>()
        .iter(ecs.app.world())
        .filter(|(parent, _, _)| parent.parent() == run)
        .map(|(_, order, parts)| (order.0, parts.0.to_message()))
        .collect();
    messages.sort_by_key(|(order, _)| *order);
    NativeResponse {
        output,
        messages: Some(messages.into_iter().map(|(_, message)| message).collect()),
        usage,
        requests,
    }
}
