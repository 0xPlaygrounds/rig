//! Matrix Q in the world: the `lookup` tool, the host's relay and its
//! never-answering handler are keys the world serves (`register_open`),
//! answered by systems that nest what they need as effects `ChildOf` the
//! call — so the child's record names the call as its parent, exactly as
//! rig-agent's bus `Lookup`'s sink dispatcher does. Program, not record: the
//! replayers answer only the leaves (the model, the host's note).

#![allow(
    clippy::type_complexity,
    reason = "test support: the queries are the point"
)]

use bevy_ecs::prelude::*;
use rig_core::{
    completion::CompletionRequestBuilder,
    effect::{EffectKind, HandlerKey, Outcome},
    message::AssistantContent,
    tool::{ToolOutput, ToolResult},
};
use rig_ecs::bus::{BusSet, EffectOutcome, InFlight, PendingEffect, RigSchedule};

use super::{
    Hold, LookupArgs, NESTED_PREAMBLE, NESTING_TOOL_KEY, NEVER_KEY, NOTE_KEY, NestedChild, Nesting,
    Note, NoteAck, RELAY_KEY, RelayNote,
};

/// Whether `key` is one the world serves itself in a nesting program.
pub fn is_served_by_the_world(key: &HandlerKey) -> bool {
    matches!(key.as_str(), NESTING_TOOL_KEY | RELAY_KEY | NEVER_KEY)
}

/// The program's nesting axis, as a resource the systems read.
#[derive(Resource, Clone, Copy)]
struct NestingSpec {
    nesting: Nesting,
    model_key: &'static str,
}

/// Install the nesting systems for `nesting` under `owner`'s model.
pub fn install(world: &mut World, nesting: Nesting, owner: &str) {
    let model_key: &'static str = Box::leak(format!("{owner}/model:default").into_boxed_str());
    world.insert_resource(NestingSpec { nesting, model_key });
    world.resource_mut::<Schedules>().add_systems(
        RigSchedule,
        // Between `Dispatch` and `Collect`: an answer a system gives is
        // recorded by `settle` in the same pass, so a scene saved when the
        // run wants its next turn has no open record (`world_resume`).
        (serve_lookup, serve_relay, finish_relay, finish_lookup)
            .chain()
            .after(BusSet::Dispatch)
            .before(BusSet::Collect),
    );
}

/// Whether the host's never-answering handler has been reached: a
/// dispatch to it is in flight.
pub fn reached(world: &mut World) -> bool {
    world
        .query_filtered::<&PendingEffect, With<InFlight>>()
        .iter(world)
        .any(|effect| effect.key.as_str() == NEVER_KEY)
}

/// An in-flight call to the `lookup` key with no child yet: nest what the
/// program says under it (a leaf answers at once).
fn serve_lookup(
    calls: Query<
        (Entity, &PendingEffect),
        (With<InFlight>, Without<EffectOutcome>, Without<Children>),
    >,
    spec: Res<NestingSpec>,
    mut commands: Commands,
) {
    for (entity, effect) in &calls {
        if effect.key.as_str() != NESTING_TOOL_KEY {
            continue;
        }
        let EffectKind::ToolCall { args, .. } = &effect.kind else {
            continue;
        };
        let args: LookupArgs = serde_json::from_str(args).unwrap_or_default();
        if args.leaf {
            commands.entity(entity).insert(tool_text("leaf".to_owned()));
            continue;
        }
        let mut spawn = |pending: PendingEffect| {
            commands.spawn((pending, ChildOf(entity)));
        };
        match spec.nesting.child {
            NestedChild::Completion => {
                let request = CompletionRequestBuilder::unbound(args.q.as_str())
                    .preamble(NESTED_PREAMBLE.to_owned())
                    .temperature(0.0)
                    .build();
                spawn(PendingEffect::new(
                    spec.model_key,
                    EffectKind::Completion {
                        request,
                        stream: false,
                    },
                ));
            }
            NestedChild::Note => spawn(
                PendingEffect::custom(
                    NOTE_KEY,
                    &Note {
                        at: "lookup".to_owned(),
                    },
                )
                .expect("serializes"),
            ),
            NestedChild::Same => spawn(PendingEffect::new(
                NESTING_TOOL_KEY,
                EffectKind::ToolCall {
                    name: "lookup".to_owned(),
                    args: r#"{"q":"","leaf":true}"#.to_owned(),
                },
            )),
            NestedChild::Relay => spawn(
                PendingEffect::custom(
                    RELAY_KEY,
                    &RelayNote {
                        at: "lookup".to_owned(),
                    },
                )
                .expect("serializes"),
            ),
            NestedChild::Never => {
                spawn(PendingEffect::custom(NEVER_KEY, &Hold).expect("serializes"))
            }
            NestedChild::NeverTwice => {
                spawn(PendingEffect::custom(NEVER_KEY, &Hold).expect("serializes"));
                spawn(PendingEffect::custom(NEVER_KEY, &Hold).expect("serializes"));
            }
        }
    }
}

/// A `lookup` call whose children have all landed answers with their text.
fn finish_lookup(
    calls: Query<(Entity, &PendingEffect, &Children), (With<InFlight>, Without<EffectOutcome>)>,
    outcomes: Query<&EffectOutcome>,
    spec: Res<NestingSpec>,
    mut commands: Commands,
) {
    for (entity, effect, children) in &calls {
        if effect.key.as_str() != NESTING_TOOL_KEY {
            continue;
        }
        let landed: Vec<&EffectOutcome> = children
            .iter()
            .filter_map(|child| outcomes.get(child).ok())
            .collect();
        if landed.len() < children.len() {
            continue;
        }
        let text = match spec.nesting.child {
            NestedChild::Completion => match &landed[0].0 {
                Ok(Outcome::Completion(response)) => response
                    .choice
                    .iter()
                    .filter_map(|content| match content {
                        AssistantContent::Text(text) => Some(text.text.trim().to_owned()),
                        AssistantContent::ToolCall(_)
                        | AssistantContent::Reasoning(_)
                        | AssistantContent::Image(_) => None,
                    })
                    .collect::<Vec<_>>()
                    .join(" "),
                other => panic!("the nested completion: {other:?}"),
            },
            NestedChild::Note => format!("noted:{}", ack(&landed[0].0).at),
            NestedChild::Relay => format!("relayed:{}", ack(&landed[0].0).at),
            NestedChild::Same => match &landed[0].0 {
                Ok(Outcome::ToolResult { result }) => {
                    format!("served:{}", result.output().render())
                }
                Ok(other) => panic!("a tool result: {other:?}"),
                Err(report) => format!("refused:{:?}", report.kind),
            },
            NestedChild::Never => match &landed[0].0 {
                Ok(_) => format!("answered:{}", ack(&landed[0].0).at),
                Err(report) => format!("failed:{:?}", report.kind),
            },
            NestedChild::NeverTwice => match (&landed[0].0, &landed[1].0) {
                (Ok(_), Ok(_)) => {
                    format!("answered:{}:{}", ack(&landed[0].0).at, ack(&landed[1].0).at)
                }
                (Err(report), _) | (_, Err(report)) => format!("failed:{:?}", report.kind),
            },
        };
        commands.entity(entity).insert(tool_text(text));
    }
}

/// The host's relay: takes a note under itself and answers with the ack.
fn serve_relay(
    relays: Query<
        (Entity, &PendingEffect),
        (With<InFlight>, Without<EffectOutcome>, Without<Children>),
    >,
    mut commands: Commands,
) {
    for (entity, effect) in &relays {
        if effect.key.as_str() != RELAY_KEY {
            continue;
        }
        let EffectKind::Custom { payload, .. } = &effect.kind else {
            continue;
        };
        let note: RelayNote = serde_json::from_value(payload.clone()).expect("a relay note");
        commands.spawn((
            PendingEffect::custom(
                NOTE_KEY,
                &Note {
                    at: format!("relay<{}", note.at),
                },
            )
            .expect("serializes"),
            ChildOf(entity),
        ));
    }
}

fn finish_relay(
    relays: Query<(Entity, &PendingEffect, &Children), (With<InFlight>, Without<EffectOutcome>)>,
    outcomes: Query<&EffectOutcome>,
    mut commands: Commands,
) {
    for (entity, effect, children) in &relays {
        if effect.key.as_str() != RELAY_KEY {
            continue;
        }
        let Some(note) = children.iter().find_map(|child| outcomes.get(child).ok()) else {
            continue;
        };
        let ack = ack(&note.0);
        commands
            .entity(entity)
            .insert(EffectOutcome(Ok(Outcome::Custom(
                serde_json::to_value(NoteAck {
                    accepted: ack.accepted,
                    at: ack.at,
                })
                .expect("an ack serializes"),
            ))));
    }
}

fn ack(outcome: &Result<Outcome, rig_core::error::ErrorReport>) -> NoteAck {
    match outcome {
        Ok(Outcome::Custom(value)) => serde_json::from_value(value.clone()).expect("an ack"),
        other => panic!("an acknowledgement: {other:?}"),
    }
}

fn tool_text(text: String) -> EffectOutcome {
    EffectOutcome(Ok(Outcome::ToolResult {
        result: ToolResult::success(ToolOutput::text(text)),
    }))
}
