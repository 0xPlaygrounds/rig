//! Native observations shared by streamed tool-ordering assertions.
//! Only fields consumed by those assertions are collected, in a fresh one-run app.

#[path = "ecs_observation/tests.rs"]
mod tests;

use crate::{ecs_agent::EcsAgent, support::StreamObservation};
use bevy_ecs::prelude::*;
use rig::streaming::{Delta, StreamEvent};
use rig_ecs::{
    agent::{RunResult, Settled, ToolCallSlot},
    bus::{BusSet, EffectOutcome, RigSchedule, Seq, Streamed},
    systems::RigSet,
};
use std::collections::HashMap;

#[derive(Resource)]
struct Seen {
    observation: StreamObservation,
    offsets: HashMap<Entity, usize>,
}

impl Default for Seen {
    fn default() -> Self {
        Self {
            observation: StreamObservation {
                all_streamed_text: String::new(),
                final_turn_text: String::new(),
                final_response_text: None,
                tool_calls: vec![],
                tool_call_records: vec![],
                tool_results: 0,
                errors: vec![],
                got_final_response: false,
                events: vec![],
            },
            offsets: HashMap::new(),
        }
    }
}

fn observe_streams(streams: Query<(Entity, &Seq, &Streamed)>, mut seen: ResMut<Seen>) {
    let mut streams: Vec<_> = streams.iter().collect();
    streams.sort_by_key(|(_, seq, _)| seq.0);
    for (entity, _, stream) in streams {
        let offset = seen.offsets.get(&entity).copied().unwrap_or(0);
        for event in stream.events.iter().skip(offset) {
            if let StreamEvent::BlockDelta {
                delta: Delta::Text { text },
                ..
            } = event
            {
                seen.observation.all_streamed_text.push_str(text);
                seen.observation.final_turn_text.push_str(text);
                seen.observation.events.push("text");
            }
        }
        seen.offsets.insert(entity, stream.events.len());
    }
}

fn observe_calls(
    calls: Query<(&ChildOf, &ToolCallSlot), Added<ToolCallSlot>>,
    mut seen: ResMut<Seen>,
) {
    let mut calls: Vec<_> = calls.iter().collect();
    if let Some((parent, _)) = calls.first() {
        assert!(
            calls
                .iter()
                .all(|(other, _)| other.parent() == parent.parent()),
            "ordering observer supports one newly materialised turn per pass"
        );
    }
    calls.sort_by_key(|(_, call)| call.index);
    for (_, call) in calls {
        seen.observation.tool_calls.push(call.name.clone());
        // The reused assertion consumes names and order, not signature or
        // additional-params records. Leave those unobserved fields empty.
        seen.observation.events.push("tool_call");
    }
}

type ToolResults<'w, 's> =
    Query<'w, 's, &'static EffectOutcome, (With<ToolCallSlot>, Added<EffectOutcome>)>;

fn observe_results(results: ToolResults, mut seen: ResMut<Seen>) {
    for result in &results {
        if let Err(error) = &result.0 {
            seen.observation.errors.push(error.to_string());
        }
        seen.observation.tool_results += 1;
        seen.observation.final_turn_text.clear();
        seen.observation.events.push("tool_result");
    }
}

fn observe_final(results: Query<&RunResult, Added<Settled>>, mut seen: ResMut<Seen>) {
    for result in &results {
        seen.observation.final_response_text = Some(result.0.clone());
        seen.observation.got_final_response = true;
        seen.observation.events.push("final_response");
    }
}

pub(crate) fn install_observers(ecs: &mut EcsAgent) {
    ecs.app.init_resource::<Seen>().add_systems(
        RigSchedule,
        (
            (observe_streams, observe_results)
                .chain()
                .after(BusSet::Collect)
                .before(RigSet::Fold),
            observe_calls
                .after(RigSet::Materialise)
                .before(RigSet::Settle),
            observe_final.after(RigSet::Settle),
        ),
    );
}

pub(crate) fn observation(ecs: &EcsAgent) -> &StreamObservation {
    &ecs.app.world().resource::<Seen>().observation
}
