//! Every hook of the corpus as a user system or observer against
//! `rig_ecs`'s public sets and components (stage 4, ruling 4; the claim of
//! `how-the-ecs-dissolves-rig-agent.md` §12, tested): no hook trait, no
//! library change beyond the sets. Each system reads the program's hook
//! list (`Hooks`) and makes the hook's decision at the hook's moment —
//! CONTRACT §9 names the moment per hook.

#![allow(
    clippy::type_complexity,
    clippy::too_many_arguments,
    reason = "test support: the queries are the point, one system per moment"
)]

use bevy_ecs::prelude::*;
use rig_core::{
    effect::{EffectKind, HandlerKey, MemoryOp, Outcome},
    error::{ErrorKind, ErrorReport},
    id::ConversationId,
    message::AssistantContent,
    streaming::{Delta, StreamEvent},
    tool::ToolOutput,
};
use rig_ecs::{
    agent::{
        Cancelled, Cursor, InvalidCall, MessageParts, Outputs, RequestPatch, Resolution, Retry,
        Run, RunOf, Settled, ToolCallSlot, Turn, UsesModel,
    },
    bus::{
        Bound, BusSet, EffectOutcome, Issued, PendingEffect, RigSchedule, Streamed as BusStreamed,
    },
    systems::{Fresh, Materialised, RigSet},
};

use super::{
    CANCEL_ADD_DISPATCH, CANCEL_ADD_OUTCOME, CANCEL_ANSWER, DENY_REASON, DONE_FEEDBACK, EMBED_KEY,
    Hook, LOOKUP_ARGS, LOOKUP_KEY, NOTE_KEY, Note, PATCHED_ARGS, Program, REPLACED_ANSWER,
    REPLACED_RESULT, RERANK_KEY, SKIP_REASON, STOP_AFTER_TURN, STOP_AT_ANSWER,
    STOP_AT_COMPLETION_CALL, STOP_AT_MODEL_SELECT, STOP_AT_START, STOP_ON_REASONING_DELTA,
    STOP_ON_TEXT_DELTA, STOP_ON_TOOL_ARGUMENTS_DELTA, STOP_ON_TOOL_CALL_DELTA,
    STOP_ON_TOOL_NAME_DELTA, Unserializable, hook_patch, rerank_request, retry_feedback,
    stop_after_turn_reason,
};

/// The program's hooks, in registration order, and what the systems need
/// of the program besides.
#[derive(Resource, Clone)]
struct Hooks {
    hooks: &'static [Hook],
    owner: &'static str,
    prompt: &'static str,
    route: Option<&'static str>,
    late_route: Option<&'static str>,
}

impl Hooks {
    fn has(&self, hook: Hook) -> bool {
        self.hooks.contains(&hook)
    }
}

/// A turn a judging system has decided about.
#[derive(Component)]
struct Judged;

/// Install the program's hooks as systems and observers.
pub fn install(world: &mut World, program: &Program) {
    world.insert_resource(Hooks {
        hooks: program.hooks,
        owner: program.owner,
        prompt: program.prompt,
        route: program.route,
        late_route: program.late_route,
    });
    world.add_observer(at_run_start);
    world.add_observer(at_settled);
    world.add_observer(at_tool_outcome);
    world.add_observer(at_memory_loaded);
    world.resource_mut::<Schedules>().add_systems(
        RigSchedule,
        (
            before_select.after(RigSet::Advance).before(RigSet::Select),
            before_assemble
                .after(RigSet::Select)
                .before(RigSet::Assemble),
            gate.in_set(BusSet::Gate),
            judge_outcomes.in_set(BusSet::Judge),
            after_fold.after(RigSet::Fold).before(RigSet::Judge),
            judge_turn.in_set(RigSet::Judge),
            resolve_invalid
                .before(RigSet::Materialise)
                .after(RigSet::Judge),
            after_settled.after(RigSet::Settle),
        ),
    );
}

fn note(commands: &mut Commands, bound: &Query<&Bound>, run: Entity, at: &str) {
    // A key nothing serves: the bind is refused, nothing is dispatched.
    if !bound.iter().any(|b| b.key.as_str() == NOTE_KEY) {
        return;
    }
    commands.spawn((
        PendingEffect::custom(NOTE_KEY, &Note { at: at.to_owned() }).expect("serializes"),
        ChildOf(run),
    ));
}

/// `on_run_start`: at the run's spawn, before its first turn.
fn at_run_start(
    added: On<Add, Run>,
    hooks: Res<Hooks>,
    bound: Query<&Bound>,
    mut commands: Commands,
) {
    let run = added.event().entity;
    for hook in hooks.hooks {
        match hook {
            Hook::StopAtStart => {
                commands
                    .entity(run)
                    .insert(Cancelled(STOP_AT_START.to_owned()));
            }
            Hook::NoteAtStart | Hook::NoteDeniedAtStart => {
                note(&mut commands, &bound, run, "start")
            }
            Hook::NoteTwice => {
                note(&mut commands, &bound, run, "first");
                note(&mut commands, &bound, run, "second");
            }
            Hook::NotesAtStart(n) => {
                for i in 0..*n {
                    note(&mut commands, &bound, run, &format!("start-{i}"));
                }
            }
            Hook::NoteUnserved => note(&mut commands, &bound, run, "start"),
            Hook::NoteUnserializableAtStart => {
                // No wire form: refused at the spawn, nothing reaches the bus.
                assert!(PendingEffect::custom(super::UNSERIALIZABLE_KEY, &Unserializable).is_err());
            }
            Hook::LookupBeforeRun => {
                commands.spawn((
                    PendingEffect::new(
                        LOOKUP_KEY,
                        EffectKind::ToolCall {
                            name: "add".to_owned(),
                            args: LOOKUP_ARGS.to_owned(),
                        },
                    ),
                    ChildOf(run),
                ));
            }
            Hook::EmbedPrompt => {
                commands.spawn((
                    PendingEffect::new(
                        EMBED_KEY,
                        EffectKind::Embed {
                            inputs: rig_core::effect::EmbedInputs::Texts(vec![
                                hooks.prompt.to_owned(),
                            ]),
                        },
                    ),
                    ChildOf(run),
                ));
            }
            Hook::RerankDocs => {
                commands.spawn((
                    PendingEffect::new(
                        RERANK_KEY,
                        EffectKind::Rerank {
                            request: rerank_request(hooks.prompt),
                        },
                    ),
                    ChildOf(run),
                ));
            }
            _ => {}
        }
    }
}

/// `on_run_settled`: at the answer.
fn at_settled(
    added: On<Add, Settled>,
    hooks: Res<Hooks>,
    bound: Query<&Bound>,
    mut commands: Commands,
) {
    let run = added.event().entity;
    for hook in hooks.hooks {
        if let Hook::NoteAtSettled = hook {
            note(&mut commands, &bound, run, "settled");
        }
    }
}

/// `on_run_start` for `ClearAtStart`: the producer's hook ran after the
/// conversation was loaded and before the first turn — here, when the
/// run's `Load` lands, a `Clear` is dispatched `ChildOf` the run.
fn at_memory_loaded(
    added: On<Add, EffectOutcome>,
    hooks: Res<Hooks>,
    effects: Query<(&PendingEffect, &ChildOf)>,
    runs: Query<(), With<Run>>,
    mut commands: Commands,
) {
    if !hooks.has(Hook::ClearAtStart) {
        return;
    }
    let entity = added.event().entity;
    let Ok((effect, child_of)) = effects.get(entity) else {
        return;
    };
    let EffectKind::Memory {
        op: MemoryOp::Load { conversation },
    } = &effect.kind
    else {
        return;
    };
    let run = child_of.parent();
    if runs.get(run).is_err() {
        return;
    }
    clear(&mut commands, &effect.key, conversation.clone(), run);
}

/// `on_run_settled` for `ClearAtSettled`: after the run's `Append` went
/// out (the producer's hook ran after the memory was appended), a `Clear`.
fn after_settled(
    appended: Query<(&PendingEffect, &ChildOf), Added<PendingEffect>>,
    runs: Query<(), (With<Run>, With<Settled>)>,
    hooks: Res<Hooks>,
    mut commands: Commands,
) {
    if !hooks.has(Hook::ClearAtSettled) {
        return;
    }
    for (effect, child_of) in &appended {
        let EffectKind::Memory {
            op: MemoryOp::Append { conversation, .. },
        } = &effect.kind
        else {
            continue;
        };
        let run = child_of.parent();
        if runs.get(run).is_err() {
            continue;
        }
        clear(&mut commands, &effect.key, conversation.clone(), run);
    }
}

fn clear(commands: &mut Commands, key: &HandlerKey, conversation: ConversationId, run: Entity) {
    commands.spawn((
        PendingEffect::new(
            key.clone(),
            EffectKind::Memory {
                op: MemoryOp::Clear { conversation },
            },
        ),
        ChildOf(run),
    ));
}

fn run_of_effect(
    effect: Entity,
    parents: &Query<&ChildOf>,
    turns: &Query<(), With<Turn>>,
    runs: &Query<(), With<Run>>,
) -> Option<Entity> {
    let mut current = effect;
    while let Ok(parent) = parents.get(current) {
        current = parent.parent();
        if runs.get(current).is_ok() {
            return Some(current);
        }
        if turns.get(current).is_err() {
            return None;
        }
    }
    None
}

/// `on_outcome` after a tool answered: a note, or a stop.
fn at_tool_outcome(
    added: On<Add, EffectOutcome>,
    hooks: Res<Hooks>,
    effects: Query<(&PendingEffect, &EffectOutcome), With<ToolCallSlot>>,
    parents: Query<&ChildOf>,
    turns: Query<(), With<Turn>>,
    runs: Query<(), With<Run>>,
    bound: Query<&Bound>,
    mut commands: Commands,
) {
    let entity = added.event().entity;
    let Ok((effect, outcome)) = effects.get(entity) else {
        return;
    };
    let EffectKind::ToolCall { name, .. } = &effect.kind else {
        return;
    };
    let Some(run) = run_of_effect(entity, &parents, &turns, &runs) else {
        return;
    };
    for hook in hooks.hooks {
        match hook {
            Hook::NoteAtOutcome => note(&mut commands, &bound, run, "outcome"),
            Hook::CancelAddOutcome if name == "add" && outcome.0.is_ok() => {
                commands
                    .entity(run)
                    .insert(Cancelled(CANCEL_ADD_OUTCOME.to_owned()));
            }
            _ => {}
        }
    }
}

/// `on_model_select`: between `Advance` and `Select`, on the fresh turn.
fn before_select(
    fresh: Query<&ChildOf, Added<Fresh>>,
    runs: Query<(&RunOf, &Cursor)>,
    agents: Query<&UsesModel>,
    bound: Query<(Entity, &Bound)>,
    hooks: Res<Hooks>,
    mut commands: Commands,
) {
    let model = |label: &str| -> Entity {
        let key = HandlerKey::from(format!("{}/model:{label}", hooks.owner));
        bound
            .iter()
            .find(|(_, b)| b.key == key)
            .map(|(entity, _)| entity)
            .expect("the route is bound")
    };
    for turn_of in &fresh {
        let run = turn_of.parent();
        let Ok((RunOf(agent), cursor)) = runs.get(run) else {
            continue;
        };
        let default = agents.get(*agent).expect("the agent's model").0;
        for hook in hooks.hooks {
            match hook {
                Hook::StopAtModelSelect => {
                    commands
                        .entity(run)
                        .insert(Cancelled(STOP_AT_MODEL_SELECT.to_owned()));
                }
                Hook::RouteAfterFirstTurn => {
                    let chosen = if cursor.turn > 1 {
                        model(hooks.route.expect("a route"))
                    } else {
                        default
                    };
                    commands.entity(run).insert(UsesModel(chosen));
                }
                Hook::RouteOnFirstTurn => {
                    let chosen = if cursor.turn == 1 {
                        model(hooks.route.expect("a route"))
                    } else {
                        default
                    };
                    commands.entity(run).insert(UsesModel(chosen));
                }
                Hook::SelectLate => {
                    commands
                        .entity(run)
                        .insert(UsesModel(model(hooks.late_route.expect("a late route"))));
                }
                _ => {}
            }
        }
    }
}

/// The corpus hook's patch, as `rig_ecs`'s data.
fn patch_of(hook: Hook, turn: usize) -> Option<RequestPatch> {
    hook_patch(hook, turn).map(|patch| RequestPatch {
        preamble: patch.preamble,
        temperature: patch.temperature,
        max_tokens: patch.max_tokens,
        tool_choice: patch.tool_choice,
        active_tools: patch.active_tools,
        additional_params: patch.additional_params,
        extra_context: patch.extra_context,
        history: patch.history.map(|messages| {
            messages
                .iter()
                .filter_map(MessageParts::from_message)
                .collect()
        }),
    })
}

/// `on_completion_call`: on the fresh turn, before `Assemble` folds it —
/// a stop, a note (dispatched before the completion), a request patch.
fn before_assemble(
    fresh: Query<(Entity, &ChildOf), Added<Fresh>>,
    runs: Query<&Cursor>,
    hooks: Res<Hooks>,
    bound: Query<&Bound>,
    mut commands: Commands,
) {
    for (turn, turn_of) in &fresh {
        let run = turn_of.parent();
        let Ok(cursor) = runs.get(run) else {
            continue;
        };
        let mut patch: Option<RequestPatch> = None;
        for hook in hooks.hooks {
            match hook {
                Hook::StopAtCompletionCall => {
                    commands
                        .entity(run)
                        .insert(Cancelled(STOP_AT_COMPLETION_CALL.to_owned()));
                }
                Hook::NoteAtCompletionCall => note(&mut commands, &bound, run, "completion_call"),
                hook => {
                    if let Some(next) = patch_of(*hook, cursor.turn) {
                        patch = Some(match patch.take() {
                            Some(earlier) => earlier.merge(next),
                            None => next,
                        });
                    }
                }
            }
        }
        if let Some(patch) = patch {
            commands.entity(turn).insert(patch);
        }
    }
}

/// `on_dispatch` on a tool call: patch its arguments, deny it, or stop.
fn gate(
    mut pending: Query<
        (Entity, &mut PendingEffect),
        (With<ToolCallSlot>, Without<Issued>, Without<EffectOutcome>),
    >,
    parents: Query<&ChildOf>,
    turns: Query<(), With<Turn>>,
    runs: Query<(), With<Run>>,
    hooks: Res<Hooks>,
    mut commands: Commands,
) {
    for (entity, mut effect) in &mut pending {
        let EffectKind::ToolCall { name, args } = &mut effect.kind else {
            continue;
        };
        if name != "add" {
            continue;
        }
        for hook in hooks.hooks {
            match hook {
                Hook::PatchAddArgs => {
                    *args = PATCHED_ARGS.to_owned();
                }
                Hook::DenyAdd => {
                    commands
                        .entity(entity)
                        .insert(EffectOutcome(Err(ErrorReport::new(
                            ErrorKind::Denied,
                            DENY_REASON,
                        ))));
                }
                Hook::CancelAddDispatch => {
                    if let Some(run) = run_of_effect(entity, &parents, &turns, &runs) {
                        commands
                            .entity(run)
                            .insert(Cancelled(CANCEL_ADD_DISPATCH.to_owned()));
                    }
                }
                _ => {}
            }
        }
    }
}

/// `on_outcome` → replace a tool result: the bus's `Judge`.
fn judge_outcomes(
    mut landed: Query<
        (&PendingEffect, &mut EffectOutcome),
        (Added<EffectOutcome>, With<ToolCallSlot>),
    >,
    hooks: Res<Hooks>,
) {
    if !hooks.has(Hook::ReplaceAddResult) {
        return;
    }
    for (effect, mut outcome) in &mut landed {
        let EffectKind::ToolCall { name, .. } = &effect.kind else {
            continue;
        };
        if name != "add" {
            continue;
        }
        if let Ok(Outcome::ToolResult { result }) = &outcome.0 {
            let replaced = result
                .clone()
                .with_output(ToolOutput::text(REPLACED_RESULT));
            outcome.0 = Ok(Outcome::ToolResult { result: replaced });
        }
    }
}

/// The delta hooks: after `Fold`, on what changed this pass.
fn after_fold(
    changed_outputs: Query<&ChildOf, (With<Turn>, Changed<Outputs>, Without<Materialised>)>,
    outputs: Query<&Outputs>,
    changed_streams: Query<(Entity, &BusStreamed), Changed<BusStreamed>>,
    parents: Query<&ChildOf>,
    turns: Query<(), With<Turn>>,
    runs: Query<(), With<Run>>,
    hooks: Res<Hooks>,
    mut commands: Commands,
) {
    if hooks.has(Hook::StopOnTextDelta) {
        for turn_of in &changed_outputs {
            let Ok(outs) = outputs.get(turn_of.parent()) else {
                continue;
            };
            let has_text = outs
                .content
                .iter()
                .any(|part| matches!(part, AssistantContent::Text(text) if !text.text.is_empty()));
            if has_text {
                commands
                    .entity(turn_of.parent())
                    .insert(Cancelled(STOP_ON_TEXT_DELTA.to_owned()));
            }
        }
    }
    for (effect, streamed) in &changed_streams {
        let Some(run) = run_of_effect(effect, &parents, &turns, &runs) else {
            continue;
        };
        let mut stop: Option<&str> = None;
        for event in &streamed.events {
            let StreamEvent::BlockDelta { delta, .. } = event else {
                continue;
            };
            stop = match delta {
                Delta::ToolName { .. } if hooks.has(Hook::StopOnToolCallDelta) => {
                    Some(STOP_ON_TOOL_CALL_DELTA)
                }
                Delta::ToolArguments { .. } if hooks.has(Hook::StopOnToolCallDelta) => {
                    Some(STOP_ON_TOOL_CALL_DELTA)
                }
                Delta::ToolName { .. } if hooks.has(Hook::StopOnToolNameDelta) => {
                    Some(STOP_ON_TOOL_NAME_DELTA)
                }
                Delta::ToolArguments { arguments }
                    if hooks.has(Hook::StopOnToolArgumentsDelta) && !arguments.is_empty() =>
                {
                    Some(STOP_ON_TOOL_ARGUMENTS_DELTA)
                }
                Delta::Reasoning { .. } if hooks.has(Hook::StopOnReasoningDelta) => {
                    Some(STOP_ON_REASONING_DELTA)
                }
                Delta::Text { .. } if hooks.has(Hook::StopOnTextDelta) => Some(STOP_ON_TEXT_DELTA),
                Delta::Text { .. }
                | Delta::TextMeta { .. }
                | Delta::Reasoning { .. }
                | Delta::ToolName { .. }
                | Delta::ToolArguments { .. } => None,
            };
            if stop.is_some() {
                break;
            }
        }
        if let Some(reason) = stop {
            commands.entity(run).insert(Cancelled(reason.to_owned()));
        }
    }
}

/// `on_model_turn_finished` and `on_outcome` on a completion: the turn is
/// complete and unread — stop, replace the answer, or retry.
fn judge_turn(
    mut turns: Query<
        (Entity, &ChildOf, &mut Outputs),
        (With<Turn>, Without<Materialised>, Without<Judged>),
    >,
    runs: Query<&Cursor>,
    hooks: Res<Hooks>,
    mut commands: Commands,
) {
    for (turn, turn_of, mut outs) in &mut turns {
        if !outs.done {
            continue;
        }
        commands.entity(turn).insert(Judged);
        let run = turn_of.parent();
        let Ok(cursor) = runs.get(run) else {
            continue;
        };
        let has_tool_calls = outs
            .content
            .iter()
            .any(|part| matches!(part, AssistantContent::ToolCall(_)));
        let text: String = outs
            .content
            .iter()
            .filter_map(|part| match part {
                AssistantContent::Text(text) => Some(text.text.as_str()),
                AssistantContent::ToolCall(_)
                | AssistantContent::Reasoning(_)
                | AssistantContent::Image(_) => None,
            })
            .collect();
        for hook in hooks.hooks {
            match hook {
                Hook::CancelAnswer if !has_tool_calls => {
                    commands
                        .entity(run)
                        .insert(Cancelled(CANCEL_ANSWER.to_owned()));
                }
                Hook::StopAfterTurn => {
                    commands
                        .entity(run)
                        .insert(Cancelled(STOP_AFTER_TURN.to_owned()));
                }
                Hook::StopAfterTurnN(n) if cursor.turn == *n => {
                    commands
                        .entity(run)
                        .insert(Cancelled(stop_after_turn_reason(*n)));
                }
                Hook::StopAtAnswer if !has_tool_calls => {
                    commands
                        .entity(run)
                        .insert(Cancelled(STOP_AT_ANSWER.to_owned()));
                }
                Hook::ReplaceAnswer if !has_tool_calls => {
                    outs.content = vec![AssistantContent::text(REPLACED_ANSWER)];
                }
                Hook::DemandDone if !has_tool_calls && !text.contains("DONE") => {
                    commands.entity(turn).insert(Retry {
                        feedback: Some(DONE_FEEDBACK.to_owned()),
                    });
                }
                _ => {}
            }
        }
    }
}

/// `on_invalid_tool_call`: before `Materialise` reads the resolution.
fn resolve_invalid(
    invalid: Query<(Entity, &InvalidCall), Without<Resolution>>,
    hooks: Res<Hooks>,
    mut commands: Commands,
) {
    for (entity, call) in &invalid {
        for hook in hooks.hooks {
            let resolution = match hook {
                Hook::RetryUnknownTool => match retry_feedback(&call.name) {
                    rig_agent::run::InvalidToolCallAction::Retry { feedback } => {
                        Resolution::Retry { feedback }
                    }
                    _ => continue,
                },
                Hook::RepairToAdd => Resolution::Repair {
                    to: "add".to_owned(),
                },
                Hook::SkipUnknown => Resolution::Skip {
                    reason: SKIP_REASON.to_owned(),
                },
                _ => continue,
            };
            commands.entity(entity).insert(resolution);
            break;
        }
    }
}
