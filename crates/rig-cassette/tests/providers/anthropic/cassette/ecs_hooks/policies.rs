//! Concrete application policies at native dispatch, outcome and turn boundaries.
use crate::goldens::{
    DENY_REASON, DONE_FEEDBACK, LOOKUP_ARGS, LOOKUP_KEY, PATCHED_ARGS, PIRATE_PREAMBLE,
    REPLACED_ANSWER, REPLACED_RESULT,
};
use bevy_ecs::prelude::*;
use rig_core::{
    effect::{EffectFamily, EffectKind, Outcome},
    error::{ErrorKind, ErrorReport},
    message::AssistantContent,
    tool::ToolOutput,
};
use rig_ecs::{
    agent::{Outputs, RequestPatch, Retry, Run, ToolCallSlot, Turn},
    bus::{EffectOutcome, Issued, PendingEffect},
    systems::{Fresh, Materialised},
};

type UnissuedTools<'w, 's> = Query<
    'w,
    's,
    (Entity, &'static mut PendingEffect),
    (With<ToolCallSlot>, Without<Issued>, Without<EffectOutcome>),
>;
pub(super) fn patch_args(mut tools: UnissuedTools) {
    for (_, mut pending) in &mut tools {
        if let EffectKind::ToolCall { name, args } = &mut pending.kind
            && name == "add"
        {
            *args = PATCHED_ARGS.into();
        }
    }
}
pub(super) fn deny_tools(tools: UnissuedTools, mut commands: Commands) {
    for (entity, pending) in &tools {
        if matches!(&pending.kind,EffectKind::ToolCall{name,..} if name=="add") {
            commands
                .entity(entity)
                .insert(EffectOutcome(Err(ErrorReport::new(
                    ErrorKind::Denied,
                    DENY_REASON,
                ))));
        }
    }
}
type LandedTools<'w, 's> = Query<
    'w,
    's,
    (&'static PendingEffect, &'static mut EffectOutcome),
    (With<ToolCallSlot>, Added<EffectOutcome>),
>;
pub(super) fn replace_results(mut tools: LandedTools) {
    for (pending, mut outcome) in &mut tools {
        if matches!(&pending.kind,EffectKind::ToolCall{name,..} if name=="add")
            && let Ok(Outcome::ToolResult { result }) = &mut outcome.0
        {
            *result = result
                .clone()
                .with_output(ToolOutput::text(REPLACED_RESULT));
        }
    }
}
pub(super) fn replace_answer(mut effects: Query<&mut EffectOutcome, Added<EffectOutcome>>) {
    for mut outcome in &mut effects {
        if let Ok(Outcome::Completion(response)) = &mut outcome.0
            && !response
                .choice
                .iter()
                .any(|c| matches!(c, AssistantContent::ToolCall(_)))
        {
            response.choice = vec![AssistantContent::text(REPLACED_ANSWER)];
        }
    }
}
pub(super) fn preamble(fresh: Query<Entity, Added<Fresh>>, mut commands: Commands) {
    for turn in &fresh {
        commands.entity(turn).insert(RequestPatch {
            preamble: Some(PIRATE_PREAMBLE.into()),
            ..Default::default()
        });
    }
}
type UnreadTurns<'w, 's> = Query<
    'w,
    's,
    (Entity, &'static Outputs),
    (With<Turn>, Without<Materialised>, Changed<Outputs>),
>;
pub(super) fn demand_done(turns: UnreadTurns, mut commands: Commands) {
    for (turn, output) in &turns {
        if output.done {
            let text: String = output
                .content
                .iter()
                .filter_map(|c| match c {
                    AssistantContent::Text(t) => Some(t.text.as_str()),
                    _ => None,
                })
                .collect();
            if !text.contains("DONE") {
                commands.entity(turn).insert(Retry {
                    feedback: Some(DONE_FEEDBACK.into()),
                });
            }
        }
    }
}

#[derive(Resource, Default)]
pub(super) struct Observed(pub Vec<EffectFamily>);
pub(super) fn observe_all(
    effects: Query<&PendingEffect, Added<Issued>>,
    mut observed: ResMut<Observed>,
) {
    for effect in &effects {
        observed.0.push(effect.kind.family());
    }
}

#[derive(Component)]
pub(super) struct StartupLookup;
pub(super) fn lookup_at_start(added: On<Add, Run>, mut commands: Commands) {
    commands.spawn((
        StartupLookup,
        PendingEffect::new(
            LOOKUP_KEY,
            EffectKind::ToolCall {
                name: "add".into(),
                args: LOOKUP_ARGS.into(),
            },
        ),
        ChildOf(added.event().entity),
    ));
}
// This app has one run. Ordinary schedule gating preserves the original awaited
// startup call: no model turn is advanced until its actual result is checked.
pub(super) fn lookup_finished(lookups: Query<Option<&EffectOutcome>, With<StartupLookup>>) -> bool {
    let mut any = false;
    for outcome in &lookups {
        any = true;
        let Some(outcome) = outcome else { return false };
        match &outcome.0 {
            Ok(Outcome::ToolResult { result }) => assert_eq!(result.output().render(), "3"),
            other => panic!("startup add answers: {other:?}"),
        }
    }
    any
}
