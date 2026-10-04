//! Ordered native dispatch and outcome policies for the tool stress family.
use crate::ecs_agent::EcsAgent;
use rig_ecs::{
    agent::{DefaultMaxTurns, Owner, Temperature},
    systems::RunCommands,
};

pub(super) fn agent<
    W: rig_core::wire::Wire<Op = rig_core::operation::Completion>,
    T: rig_core::driver::Transport<W>,
>(
    model: rig_core::driver::Model<W, T>,
    preamble: &str,
    name: &str,
    temperature: f64,
) -> EcsAgent {
    let mut ecs = EcsAgent::new(model, preamble, 1);
    ecs.app.world_mut().entity_mut(ecs.agent).insert((
        DefaultMaxTurns(None),
        Owner(name.into()),
        Temperature(Some(temperature)),
    ));
    ecs
}
pub(super) async fn prompt(ecs: &mut EcsAgent, prompt: &str, max_turns: usize) -> String {
    let run = ecs
        .app
        .world_mut()
        .spawn_run(ecs.agent, &[], prompt, false, Some(max_turns));
    ecs.wait_for_success(run).await
}
