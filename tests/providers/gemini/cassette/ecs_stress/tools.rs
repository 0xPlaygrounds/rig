//! Ordered native dispatch and outcome policies for the tool stress family.
use super::super::{hook_stress_support::ResultRewrite, tools_support::ToolEventRecorder};
use crate::ecs_agent::EcsAgent;
use bevy_ecs::prelude::*;
use rig::{
    completion::{CompletionModel, PromptError},
    effect::{EffectKind, Outcome},
    tool::ToolOutput,
};
use rig_ecs::{
    agent::{
        Cancelled, DefaultMaxTurns, Failure, Order, Owner, Parts, Temperature, ToolCallSlot, Turn,
        Utterance,
    },
    bus::{BusSet, EffectOutcome, Issued, PendingEffect, RigSchedule},
    systems::spawn_run,
};

#[derive(Resource, Default)]
struct PolicyCount(usize);
#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
struct DispatchSlot(usize);
#[derive(SystemSet, Debug, Hash, PartialEq, Eq, Clone)]
struct OutcomeSlot(usize);

pub(super) fn agent(
    model: impl CompletionModel + 'static,
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
    ecs.app.init_resource::<PolicyCount>();
    ecs
}
fn slot(ecs: &mut EcsAgent) -> usize {
    let slot = ecs.app.world().resource::<PolicyCount>().0;
    ecs.app.world_mut().resource_mut::<PolicyCount>().0 += 1;
    ecs.app.configure_sets(
        RigSchedule,
        (
            DispatchSlot(slot).in_set(BusSet::Gate),
            OutcomeSlot(slot).in_set(BusSet::Judge),
        ),
    );
    if slot > 0 {
        ecs.app.configure_sets(
            RigSchedule,
            (
                DispatchSlot(slot).after(DispatchSlot(slot - 1)),
                OutcomeSlot(slot).after(OutcomeSlot(slot - 1)),
            ),
        );
    }
    slot
}
type NewTools<'w, 's> = Query<
    'w,
    's,
    &'static mut PendingEffect,
    (
        With<ToolCallSlot>,
        Added<PendingEffect>,
        Without<Issued>,
        Without<EffectOutcome>,
    ),
>;
pub(super) fn set_arg(
    ecs: &mut EcsAgent,
    tool: &'static str,
    key: &'static str,
    value: serde_json::Value,
) {
    let slot = slot(ecs);
    ecs.app.add_systems(
        RigSchedule,
        (move |mut tools: NewTools| {
            for mut pending in &mut tools {
                if let EffectKind::ToolCall { name, args } = &mut pending.kind
                    && name == tool
                {
                    let mut parsed: serde_json::Value =
                        serde_json::from_str(args).unwrap_or_else(|_| serde_json::json!({}));
                    parsed[key] = value.clone();
                    *args = parsed.to_string();
                }
            }
        })
        .in_set(DispatchSlot(slot)),
    );
}
type Results<'w, 's> = Query<
    'w,
    's,
    (&'static PendingEffect, &'static mut EffectOutcome),
    (With<ToolCallSlot>, Added<EffectOutcome>),
>;
pub(super) fn rewrite_result(ecs: &mut EcsAgent, tool: &'static str, rewrite: ResultRewrite) {
    let slot = slot(ecs);
    ecs.app.add_systems(
        RigSchedule,
        (move |mut tools: Results| {
            for (pending, mut outcome) in &mut tools {
                if matches!(&pending.kind, EffectKind::ToolCall{name,..} if name == tool)
                    && let Ok(Outcome::ToolResult { result }) = &mut outcome.0
                {
                    let text = match &rewrite {
                        ResultRewrite::Replace(marker) => (*marker).to_owned(),
                        ResultRewrite::Wrap { prefix, suffix } => {
                            format!("{prefix}{}{suffix}", result.output().render())
                        }
                        ResultRewrite::Truncate(n) => {
                            result.output().render().chars().take(*n).collect()
                        }
                    };
                    *result = result.clone().with_output(ToolOutput::text(text));
                }
            }
        })
        .in_set(OutcomeSlot(slot)),
    );
}
pub(super) fn record(ecs: &mut EcsAgent, recorder: ToolEventRecorder) {
    let slot = slot(ecs);
    let call_recorder = recorder.clone();
    ecs.app.add_systems(
        RigSchedule,
        (move |tools: NewTools| {
            for pending in &tools {
                if let EffectKind::ToolCall { name, args } = &pending.kind {
                    call_recorder
                        .calls
                        .lock()
                        .expect("calls lock")
                        .push((name.clone(), args.clone()));
                }
            }
        })
        .in_set(DispatchSlot(slot)),
    );
    ecs.app.add_systems(
        RigSchedule,
        (move |tools: Results| {
            for (pending, outcome) in &tools {
                if let EffectKind::ToolCall { name, args } = &pending.kind
                    && let Ok(Outcome::ToolResult { result }) = &outcome.0
                {
                    recorder.results.lock().expect("results lock").push((
                        name.clone(),
                        args.clone(),
                        result.output().render(),
                    ));
                }
            }
        })
        .in_set(OutcomeSlot(slot)),
    );
}
type Landed<'w, 's> = Query<
    'w,
    's,
    (&'static ChildOf, &'static PendingEffect),
    (With<ToolCallSlot>, Added<EffectOutcome>),
>;
pub(super) fn terminate_result(ecs: &mut EcsAgent, tool: &'static str, reason: &'static str) {
    let slot = slot(ecs);
    ecs.app.add_systems(
        RigSchedule,
        (move |tools: Landed, parents: Query<&ChildOf, With<Turn>>, mut commands: Commands| {
            for (parent, pending) in &tools {
                if matches!(&pending.kind,EffectKind::ToolCall{name,..} if name == tool) {
                    commands
                        .entity(parents.get(parent.parent()).expect("tool turn").parent())
                        .insert(Cancelled(reason.into()));
                }
            }
        })
        .in_set(OutcomeSlot(slot)),
    );
}
pub(super) async fn prompt(ecs: &mut EcsAgent, prompt: &str, max_turns: usize) -> String {
    let run = spawn_run(
        ecs.app.world_mut(),
        ecs.agent,
        &[],
        prompt,
        false,
        Some(max_turns),
    );
    ecs.wait_for_success(run).await
}
// Projection only: the original validator consumes PromptError, while native
// cancellation stores Failure on the run and retains separate utterance entities.
// Neither reason nor history is supplied from scenario expectations.
pub(super) async fn cancelled(ecs: &mut EcsAgent, prompt: &str, max_turns: usize) -> PromptError {
    let run = spawn_run(
        ecs.app.world_mut(),
        ecs.agent,
        &[],
        prompt,
        false,
        Some(max_turns),
    );
    let failure = ecs
        .wait_for_outcome(run)
        .await
        .expect_err("ToolResult cancellation must fail");
    let Failure::Cancelled(report) = failure else {
        panic!("expected actual native cancellation, got {failure:?}")
    };
    let world = ecs.app.world_mut();
    let mut history: Vec<_> = world
        .query_filtered::<(&ChildOf, &Order, &Parts), With<Utterance>>()
        .iter(world)
        .filter(|(parent, _, _)| parent.parent() == run)
        .map(|(_, order, parts)| (order.0, parts.0.to_message()))
        .collect();
    history.sort_by_key(|(order, _)| *order);
    PromptError::PromptCancelled {
        reason: report.message,
        chat_history: history.into_iter().map(|(_, message)| message).collect(),
    }
}
