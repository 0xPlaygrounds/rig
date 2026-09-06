//! Native run-owned entry storage and lifecycle observers for provider parity.
use crate::ecs_agent::EcsAgent;
use bevy_ecs::prelude::*;
use rig::{
    effect::EffectKind,
    streaming::{StreamEvent, StreamFinal},
};
use rig_ecs::{
    agent::{Cursor, MessageParts, Parts, Run, RunResult, Settled, Turn},
    bus::{PendingEffect, RigSchedule, Seq, Streamed},
    systems::RigSet,
};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct Entry {
    kind: String,
    turn: usize,
    value: serde_json::Value,
}
#[derive(Component, Default, serde::Serialize, serde::Deserialize)]
struct Entries(Vec<Entry>);
#[derive(Resource, Clone, Default)]
pub(crate) struct LifecycleProbe {
    rewrite_to: Option<String>,
    phases: bool,
    pub starts: Arc<AtomicUsize>,
    settles: Arc<Mutex<Vec<String>>>,
    exported: Arc<Mutex<Vec<Entry>>>,
}
impl LifecycleProbe {
    pub fn rewriting_to(prompt: &str) -> Self {
        Self {
            rewrite_to: Some(prompt.into()),
            ..Default::default()
        }
    }
    pub fn entry_log() -> Self {
        Self {
            phases: true,
            ..Default::default()
        }
    }
    pub fn settle_outcomes(&self) -> Vec<String> {
        self.settles.lock().expect("settles").clone()
    }
    pub fn exported_completion_calls(&self) -> Option<u64> {
        self.exported
            .lock()
            .expect("entries")
            .iter()
            .rev()
            .find(|e| e.kind == "completion_calls")
            .and_then(|e| e.value.as_u64())
    }
    pub fn assert_phases(&self, min_calls: usize) {
        // Project only observed data into the original assertion helper. Its
        // AgentHook implementation never executes in the native producer.
        let assertion = crate::support::EntryLogProbe::default();
        *assertion.settled.lock().expect("assertion entries") = self
            .exported
            .lock()
            .expect("entries")
            .iter()
            .map(|e| rig::agent::RunEntry {
                kind: e.kind.clone(),
                turn: e.turn,
                value: e.value.clone(),
            })
            .collect();
        assertion.assert_phases(min_calls);
    }
}
fn start(
    runs: Query<(Entity, &Cursor), Added<Run>>,
    mut utterances: Query<(&ChildOf, &mut Parts)>,
    probe: Res<LifecycleProbe>,
    mut commands: Commands,
) {
    for (run, cursor) in &runs {
        assert_eq!(cursor.turn, 0, "start precedes the first model turn");
        probe.starts.fetch_add(1, Ordering::SeqCst);
        let mut entries = Entries::default();
        if probe.phases {
            entries.0.push(Entry {
                kind: "phase".into(),
                turn: cursor.turn,
                value: "run_start".into(),
            });
        }
        commands.entity(run).insert(entries);
        if let Some(prompt) = &probe.rewrite_to {
            let mut count = 0;
            for (parent, mut parts) in &mut utterances {
                if parent.parent() == run {
                    parts.0 = MessageParts::from_message(&rig::message::Message::user(prompt))
                        .expect("a user prompt has message parts");
                    count += 1;
                }
            }
            assert_eq!(
                count, 1,
                "this startup rewrite has exactly one prompt and no history"
            );
        }
    }
}
fn completion(
    calls: Query<(&PendingEffect, &ChildOf), Added<PendingEffect>>,
    turns: Query<&ChildOf, With<Turn>>,
    mut runs: Query<(&Cursor, &mut Entries)>,
    probe: Res<LifecycleProbe>,
) {
    for (call, parent) in &calls {
        if !matches!(call.kind, EffectKind::Completion { .. }) {
            continue;
        }
        let run = turns
            .get(parent.parent())
            .expect("model effect belongs to turn")
            .parent();
        let (cursor, mut entries) = runs.get_mut(run).expect("run owns durable entries");
        let (kind, value) = if probe.phases {
            ("phase", serde_json::json!("completion_call"))
        } else {
            let count = entries
                .0
                .iter()
                .rev()
                .find(|e| e.kind == "completion_calls")
                .and_then(|e| e.value.as_u64())
                .unwrap_or(0)
                + 1;
            ("completion_calls", serde_json::json!(count))
        };
        entries.0.push(Entry {
            kind: kind.into(),
            turn: cursor.turn,
            value,
        });
    }
}
fn settle(runs: Query<(&RunResult, &Entries), Added<Settled>>, probe: Res<LifecycleProbe>) {
    for (_answer, entries) in &runs {
        // Export from actual run storage at settlement. A serialization round
        // trip checks this application state is data, independent of the World.
        let bytes = serde_json::to_vec(entries).expect("entry log serializes");
        let restored: Entries = serde_json::from_slice(&bytes).expect("entry log restores");
        *probe.exported.lock().expect("export") = restored.0;
        probe
            .settles
            .lock()
            .expect("settles")
            .push("response".into());
    }
}
pub(crate) fn install(ecs: &mut EcsAgent, probe: LifecycleProbe) {
    ecs.app.insert_resource(probe).add_systems(
        RigSchedule,
        (
            start.before(RigSet::Advance),
            completion.after(RigSet::Assemble).before(RigSet::Patch),
            settle.after(RigSet::Settle),
        ),
    );
}
pub(crate) fn provider_final(ecs: &mut EcsAgent) -> StreamFinal {
    let mut streams = ecs.app.world_mut().query::<(&Seq, &Streamed)>();
    streams
        .iter(ecs.app.world())
        .filter_map(|(seq, stream)| {
            stream.events.iter().rev().find_map(|event| {
                if let StreamEvent::Final(value) = event {
                    Some((seq.0, value.clone()))
                } else {
                    None
                }
            })
        })
        .max_by_key(|(seq, _)| *seq)
        .expect("stream yields a typed provider final")
        .1
}

/// Preserve the original undeclared one-turn default; overrides stay run-local.
pub(crate) fn agent(
    model: impl rig::completion::CompletionModel + 'static,
    preamble: &str,
) -> EcsAgent {
    let mut ecs = EcsAgent::new(model, preamble, 1);
    ecs.app
        .world_mut()
        .entity_mut(ecs.agent)
        .insert(rig_ecs::agent::DefaultMaxTurns(None));
    ecs
}
