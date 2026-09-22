use bevy_ecs::prelude::*;
use rig_core::effect::EffectKind;
use rig_ecs::bus::{BusSet, Candidate, PendingEffect, RigSchedule};
use std::collections::HashSet;

#[derive(Resource)]
struct ToolBudget {
    admitted: usize,
    seen: HashSet<Entity>,
}

impl ToolBudget {
    fn admit(&mut self, pending: Vec<Entity>) {
        let additional = pending
            .iter()
            .filter(|entity| !self.seen.contains(entity))
            .count();
        assert!(
            self.admitted + additional <= 60,
            "whole-task tool dispatch budget exhausted"
        );
        self.admitted += additional;
        self.seen.extend(pending);
    }
}

pub(super) fn install(app: &mut bevy_app::App, cell: &super::Cell) {
    let admitted = super::tool(cell)
        .0
        .lock()
        .expect("task state")
        .invocations
        .len();
    app.insert_resource(ToolBudget {
        admitted,
        seen: HashSet::new(),
    });
    app.add_systems(
        RigSchedule,
        enforce.after(BusSet::Gate).before(BusSet::Dispatch),
    );
}

pub(super) fn assert_dispatched(world: &World, tools: usize) {
    assert_eq!(
        world.resource::<ToolBudget>().admitted,
        tools,
        "every recorded tool passed the dispatch budget"
    );
}

fn enforce(world: &mut World) {
    let pending = world
        .query_filtered::<(Entity, &PendingEffect), Candidate>()
        .iter(world)
        .filter(|(_, effect)| matches!(effect.kind, EffectKind::ToolCall { .. }))
        .map(|(entity, _)| entity)
        .collect();
    world.resource_mut::<ToolBudget>().admit(pending);
}

#[cfg(test)]
#[path = "budget/tests.rs"]
mod tests;
