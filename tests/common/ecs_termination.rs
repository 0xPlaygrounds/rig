//! Native turn-boundary observation and cap escalation for the termination family.
//! Assertion data is shared across surfaces; execution uses only native systems.

use crate::{ecs_agent::EcsAgent, support::ObservedTermination};
use bevy_ecs::prelude::*;
use rig_core::{
    completion::FinishReason,
    effect::{EffectKind, Outcome},
    message::AssistantContent,
};
use rig_ecs::{
    agent::{Outputs, RequestPatch, Retry, Turn},
    bus::{EffectOutcome, PendingEffect, RigSchedule},
    systems::RigSet,
};
use std::sync::{Arc, Mutex};

#[derive(Resource, Clone, Default)]
pub(crate) struct NativeProbe(Arc<Mutex<Vec<ObservedTermination>>>);

impl NativeProbe {
    pub(crate) fn observations(&self) -> Vec<ObservedTermination> {
        self.0.lock().expect("observations").clone()
    }
    pub(crate) fn first_reason(&self) -> Option<FinishReason> {
        self.observations()
            .first()
            .and_then(|(reason, _)| reason.clone())
    }
    pub(crate) fn first_max_tokens(&self) -> Option<u64> {
        self.observations().first().and_then(|(_, cap)| *cap)
    }
}

#[derive(Default)]
struct EscalationState {
    cap: u64,
    grown: u64,
    escalations: Vec<u64>,
}

#[derive(Resource, Clone)]
pub(crate) struct NativeEscalation(Arc<Mutex<EscalationState>>);

impl NativeEscalation {
    pub(crate) fn new(start: u64, grown: u64) -> Self {
        Self(Arc::new(Mutex::new(EscalationState {
            cap: start,
            grown,
            escalations: vec![],
        })))
    }
    pub(crate) fn escalations(&self) -> Vec<u64> {
        self.0.lock().expect("escalations").escalations.clone()
    }
    pub(crate) fn retries(&self) -> u32 {
        self.0.lock().expect("escalations").escalations.len() as u32
    }
}

fn patch_cap(
    turns: Query<Entity, (With<Turn>, Added<Turn>)>,
    escalation: Option<Res<NativeEscalation>>,
    mut commands: Commands,
) {
    if let Some(escalation) = escalation {
        let cap = escalation.0.lock().expect("cap").cap;
        for turn in &turns {
            commands.entity(turn).insert(RequestPatch {
                max_tokens: Some(cap),
                ..Default::default()
            });
        }
    }
}

#[derive(Component)]
struct CompletionObserved;

type Completed<'w, 's> =
    Query<'w, 's, (Entity, &'static Outputs), (With<Turn>, Without<CompletionObserved>)>;

fn observe_turn(
    turns: Completed,
    effects: Query<(&ChildOf, &PendingEffect, &EffectOutcome)>,
    probe: Res<NativeProbe>,
    escalation: Option<Res<NativeEscalation>>,
    mut commands: Commands,
) {
    for (turn, outputs) in &turns {
        if !outputs.done {
            continue;
        }
        let matching: Vec<_> = effects
            .iter()
            .filter(|(parent, _, _)| parent.parent() == turn)
            .filter_map(|(_, pending, outcome)| match (&pending.kind, &outcome.0) {
                (EffectKind::Completion { request, .. }, Ok(Outcome::Completion(response))) => {
                    Some((request, response))
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            matching.len(),
            1,
            "one model outcome for the completed turn"
        );
        let (request, response) = matching[0];
        let has_tool = outputs
            .content
            .iter()
            .any(|part| matches!(part, AssistantContent::ToolCall(_)));
        let reason = response
            .finish_reason()
            .map(|reason| reason.reconcile_with_output(has_tool));
        probe
            .0
            .lock()
            .expect("observations")
            .push((reason.clone(), request.max_tokens));
        // Later accounting changes are not another completed attempt.
        commands.entity(turn).insert(CompletionObserved);
        if let Some(escalation) = &escalation
            && reason.as_ref().is_some_and(FinishReason::truncated_output)
            && !has_tool
        {
            let mut state = escalation.0.lock().expect("cap");
            if state.escalations.is_empty() {
                let grown = state.grown;
                state.cap = grown;
                state.escalations.push(grown);
                commands.entity(turn).insert(Retry { feedback: None });
            }
        }
    }
}

pub(crate) fn install(
    ecs: &mut EcsAgent,
    probe: NativeProbe,
    escalation: Option<NativeEscalation>,
) {
    ecs.app.insert_resource(probe);
    if let Some(escalation) = escalation {
        ecs.app.insert_resource(escalation);
    }
    ecs.app.add_systems(
        RigSchedule,
        (
            patch_cap.after(RigSet::Advance).before(RigSet::Assemble),
            observe_turn.after(RigSet::Fold).before(RigSet::Judge),
        ),
    );
}
