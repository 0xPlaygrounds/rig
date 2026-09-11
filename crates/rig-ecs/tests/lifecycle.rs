//! Checked lifecycle and inspection against observable run behavior.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test assertions"
)]

mod run_support;

use bevy_app::{App, Update};
use bevy_ecs::{prelude::*, system::RunSystemOnce};
use rig_core::{
    effect::{EffectKind, FamilyDescriptor, Outcome},
    message::AssistantContent,
};
use rig_ecs::{
    agent::{self, Assembling, RunCounter, RunResult},
    bus::{EffectOutcome, Handlers, Held, PendingEffect, RigSchedule, run_to_quiescence},
    commands::{Agent, OperationError, Prompt, install},
    inspect::{RunStatus, RunView, inspect},
    lifecycle::{CancelOutcome, cancel, fork},
    systems::RigSet,
};

fn app() -> App {
    let mut app = App::new();
    install(app.world_mut(), Default::default()).expect("install");
    app.add_systems(Update, run_to_quiescence);
    app
}

fn start(app: &mut App) -> Entity {
    let (model, _) = run_support::Capturing::new("model", "answer");
    let model = Handlers::register_in(app.world_mut(), "model", model).expect("model");
    let agent = Agent::new(model).spawn(app.world_mut()).expect("agent");
    Prompt::new(agent, "hello")
        .spawn(app.world_mut())
        .expect("prompt")
}

#[test]
fn cancellation_preserves_the_ending_and_reports_stale_targets() {
    let mut app = app();
    let run = start(&mut app);
    assert_eq!(
        cancel(app.world_mut(), run, "stop"),
        Ok(CancelOutcome::Cancelled)
    );
    assert_eq!(
        cancel(app.world_mut(), run, "late"),
        Ok(CancelOutcome::AlreadyFinished)
    );
    let view = inspect(app.world(), run).expect("inspect");
    assert_eq!(view.status(), RunStatus::Failed);
    assert!(
        matches!(view.failure(), Some(agent::Failure::Cancelled(report)) if report.message == "stop")
    );
    app.world_mut().despawn(run);
    assert_eq!(
        cancel(app.world_mut(), run, "stale"),
        Err(OperationError::MissingEntity(run))
    );
}

#[test]
fn inspection_is_derived_and_scoped_to_the_selected_run() {
    let mut app = app();
    let run = start(&mut app);
    let unrelated = app
        .world_mut()
        .spawn(PendingEffect::new(
            "unrelated",
            EffectKind::Custom {
                kind: "other".into(),
                payload: serde_json::Value::Null,
            },
        ))
        .id();
    let nested = app.world_mut().spawn(ChildOf(run)).id();
    let effect = app
        .world_mut()
        .spawn((
            PendingEffect::new(
                "held",
                EffectKind::Custom {
                    kind: "question".into(),
                    payload: serde_json::Value::Null,
                },
            ),
            Held,
            ChildOf(nested),
        ))
        .id();
    let info = inspect(app.world(), run).expect("inspect");
    assert_eq!(info.status(), RunStatus::Ready);
    let effects: Vec<_> = info.effects().collect();
    assert_eq!(effects.len(), 1);
    assert_eq!(effects[0].entity, effect);
    assert_ne!(effects[0].entity, unrelated);
    assert!(effects[0].held);
    assert!(!effects[0].in_flight);
    app.world_mut().entity_mut(run).remove::<Assembling>();
    assert_eq!(
        inspect(app.world(), run).expect("inspect").status(),
        RunStatus::Unknown
    );
    app.world_mut()
        .run_system_once(move |runs: Query<RunView>| {
            let run = runs.get(run).expect("query view");
            assert_eq!(run.status(), RunStatus::Unknown);
            assert!(!run.is_finished());
            assert_eq!(run.answer(), None);
        })
        .expect("system with inferred lifetimes");
}

#[test]
fn unsafe_forks_do_not_allocate_entities_or_advance_sequence() {
    let mut app = app();
    let run = start(&mut app);
    for marker in ["remembering", "awaiting", "fresh", "effect"] {
        let child = app.world_mut().spawn_empty().id();
        app.world_mut().entity_mut(child).insert(ChildOf(run));
        match marker {
            "remembering" => {
                app.world_mut().entity_mut(run).insert(agent::Remembering);
            }
            "awaiting" => {
                app.world_mut().entity_mut(run).insert(agent::AwaitingModel);
            }
            "fresh" => {
                app.world_mut()
                    .entity_mut(child)
                    .insert((agent::Turn, rig_ecs::systems::Fresh));
            }
            _ => {
                app.world_mut().entity_mut(child).insert((
                    PendingEffect::new(
                        "pending",
                        EffectKind::Custom {
                            kind: "held".into(),
                            payload: serde_json::Value::Null,
                        },
                    ),
                    Held,
                ));
            }
        }
        let before = app.world().entities().count_spawned();
        let sequence = app.world().resource::<RunCounter>().0;
        assert_eq!(
            fork(app.world_mut(), run),
            Err(OperationError::UnsafeFork(run)),
            "{marker}"
        );
        assert_eq!(app.world().entities().count_spawned(), before);
        assert_eq!(app.world().resource::<RunCounter>().0, sequence);
        app.world_mut()
            .entity_mut(run)
            .remove::<(agent::Remembering, agent::AwaitingModel)>();
        app.world_mut().despawn(child);
    }
    let wrong = app.world_mut().spawn_empty().id();
    assert_eq!(
        fork(app.world_mut(), wrong),
        Err(OperationError::NotRun(wrong))
    );
    app.world_mut().despawn(wrong);
    assert_eq!(
        fork(app.world_mut(), wrong),
        Err(OperationError::MissingEntity(wrong))
    );
}

#[derive(Resource)]
struct Branch {
    source: Entity,
    clone: Option<Entity>,
    effects_at_fork: usize,
}

fn branch_between_turns(world: &mut World) {
    let branch = world.resource::<Branch>();
    let source = branch.source;
    if branch.clone.is_some()
        || world
            .get::<agent::Cursor>(source)
            .is_none_or(|cursor| cursor.turn != 1)
        || world.get::<Assembling>(source).is_none()
    {
        return;
    }
    let clone = fork(world, source).expect("a completed tool turn is a safe boundary");
    let count = inspect(world, clone)
        .expect("clone inspection")
        .effects()
        .count();
    let mut branch = world.resource_mut::<Branch>();
    branch.clone = Some(clone);
    branch.effects_at_fork = count;
}

#[test]
fn between_turn_forks_copy_history_without_replaying_prior_tool_effects() {
    let mut app = app();
    let (model, requests) = run_support::Scripted::new(
        "model",
        vec![
            vec![run_support::call(
                "add-1",
                "add",
                serde_json::json!({"x": 2, "y": 3}),
            )],
            vec![AssistantContent::text("five")],
            vec![AssistantContent::text("also five")],
        ],
    );
    let model = Handlers::register_in(app.world_mut(), "model", model).expect("model");
    let tool = Handlers::register_in(app.world_mut(), "add", run_support::Adder::new("add"))
        .expect("tool");
    let agent = Agent::new(model)
        .tools([tool])
        .max_turns(2)
        .spawn(app.world_mut())
        .expect("agent");
    let run = Prompt::new(agent, "add 2 and 3")
        .spawn(app.world_mut())
        .expect("prompt");
    app.insert_resource(Branch {
        source: run,
        clone: None,
        effects_at_fork: usize::MAX,
    });
    app.world_mut()
        .resource_mut::<Schedules>()
        .get_mut(RigSchedule)
        .expect("schedule")
        .add_systems(
            branch_between_turns
                .after(RigSet::Materialise)
                .before(RigSet::Settle),
        );
    run_support::tick_until(&mut app, "both branches complete", |world| {
        let clone = world.resource::<Branch>().clone;
        world.get::<RunResult>(run).is_some()
            && clone.is_some_and(|clone| world.get::<RunResult>(clone).is_some())
    });
    assert_eq!(app.world().resource::<Branch>().effects_at_fork, 0);
    let requests = requests.lock().expect("requests");
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[1].chat_history, requests[2].chat_history);
    let world = app.world_mut();
    let tools = world
        .query::<&PendingEffect>()
        .iter(world)
        .filter(|effect| matches!(effect.kind, EffectKind::ToolCall { .. }))
        .count();
    assert_eq!(
        tools, 1,
        "the earlier tool was dispatched once across both branches"
    );
}

#[test]
fn explicit_history_can_fork_without_duplicating_memory_appends() {
    let mut app = app();
    let (model, _) = run_support::Capturing::new("model", "answer");
    let model = Handlers::register_in(app.world_mut(), "model", model).expect("model");
    let memory = Handlers::with(app.world_mut(), |handlers| {
        handlers.register_open("memory", FamilyDescriptor::Memory {})
    })
    .expect("bus")
    .expect("memory");
    let agent = Agent::new(model)
        .memory(memory, "chat")
        .spawn(app.world_mut())
        .expect("agent");
    let history = vec![agent::MessageParts::User {
        content: vec![rig_core::message::UserContent::text("earlier")],
    }];
    let run = Prompt::new(agent, "hello")
        .history(history)
        .spawn(app.world_mut())
        .expect("prompt");
    assert!(app.world().get::<agent::Remembering>(run).is_none());
    assert!(fork(app.world_mut(), run).is_ok());
}

#[test]
fn an_outcome_does_not_make_an_uncollected_worker_safe_to_fork() {
    let mut app = app();
    let run = start(&mut app);
    let child = app
        .world_mut()
        .spawn((
            PendingEffect::new(
                "worker",
                EffectKind::Custom {
                    kind: "work".into(),
                    payload: serde_json::Value::Null,
                },
            ),
            EffectOutcome(Ok(Outcome::Custom {
                payload: serde_json::Value::Null,
            })),
            rig_ecs::bus::Serving,
            ChildOf(run),
        ))
        .id();
    assert_eq!(
        fork(app.world_mut(), run),
        Err(OperationError::UnsafeFork(run))
    );
    app.world_mut()
        .entity_mut(child)
        .remove::<rig_ecs::bus::Serving>();
    assert!(fork(app.world_mut(), run).is_ok());
}

#[test]
fn clone_observers_can_remove_a_branch_without_leaving_orphan_allocations() {
    let mut app = app();
    let source = start(&mut app);
    app.world_mut()
        .add_observer(move |added: On<Add, agent::Run>, mut commands: Commands| {
            if added.event().entity != source {
                commands.entity(added.event().entity).despawn();
            }
        });
    let before = app.world().entities().count_spawned();
    assert!(matches!(
        fork(app.world_mut(), source),
        Err(OperationError::MissingEntity(_))
    ));
    assert_eq!(app.world().entities().count_spawned(), before);
    assert!(app.world().get::<agent::Run>(source).is_some());
}

#[test]
fn contradictory_active_markers_report_unknown_and_exhausted_forks_do_not_mutate() {
    let mut app = app();
    let run = start(&mut app);
    app.world_mut().entity_mut(run).insert(agent::AwaitingModel);
    assert_eq!(
        inspect(app.world(), run).expect("view").status(),
        RunStatus::Unknown
    );
    app.world_mut()
        .entity_mut(run)
        .remove::<agent::AwaitingModel>();
    app.world_mut().resource_mut::<RunCounter>().0 = u64::MAX;
    let before = app.world().entities().count_spawned();
    assert_eq!(
        fork(app.world_mut(), run),
        Err(OperationError::SequenceExhausted)
    );
    assert_eq!(app.world().entities().count_spawned(), before);
}

#[test]
fn retry_rejects_incomplete_tool_and_finished_turns_and_preserves_pending_feedback() {
    use rig_ecs::{
        agent::{Outputs, Retry, Turn},
        lifecycle::retry_turn,
        systems::{Folded, Materialised},
    };
    let mut app = app();
    let run = start(&mut app);
    app.world_mut()
        .entity_mut(run)
        .remove::<Assembling>()
        .insert(agent::AwaitingModel);
    let turn = app
        .world_mut()
        .spawn((
            Turn,
            ChildOf(run),
            Folded(agent::OutputKind::Native),
            Outputs::default(),
        ))
        .id();
    let invalid = Err(OperationError::InvalidPhase {
        entity: turn,
        operation: "retry",
    });
    assert_eq!(retry_turn(app.world_mut(), turn, Retry::default()), invalid);
    app.world_mut().get_mut::<Outputs>(turn).unwrap().done = true;
    app.world_mut()
        .get_mut::<Outputs>(turn)
        .unwrap()
        .content
        .push(run_support::call("call", "tool", serde_json::json!({})));
    assert_eq!(retry_turn(app.world_mut(), turn, Retry::default()), invalid);
    app.world_mut().get_mut::<Outputs>(turn).unwrap().content =
        vec![AssistantContent::text("answer")];
    let retry = Retry::default().feedback("try again");
    assert_eq!(retry_turn(app.world_mut(), turn, retry.clone()), Ok(()));
    assert_eq!(retry_turn(app.world_mut(), turn, retry.clone()), Ok(()));
    assert_eq!(
        retry_turn(app.world_mut(), turn, Retry::default()),
        Err(OperationError::ConflictingRetry(turn))
    );
    assert_eq!(app.world().get::<Retry>(turn), Some(&retry));
    app.world_mut().entity_mut(turn).insert(Materialised);
    assert_eq!(retry_turn(app.world_mut(), turn, retry.clone()), invalid);
    app.world_mut().entity_mut(turn).remove::<Materialised>();
    cancel(app.world_mut(), run, "stop").unwrap();
    assert_eq!(retry_turn(app.world_mut(), turn, retry), invalid);
    app.world_mut().despawn(turn);
    assert_eq!(
        retry_turn(app.world_mut(), turn, Retry::default()),
        Err(OperationError::MissingEntity(turn))
    );
}

#[test]
fn patch_composes_while_fresh_and_rejects_late_or_cancelled_targets() {
    use rig_ecs::{
        agent::{RequestPatch, Turn},
        lifecycle::patch_turn,
        systems::Fresh,
    };
    let mut app = app();
    let run = start(&mut app);
    let turn = app.world_mut().spawn((Turn, ChildOf(run), Fresh)).id();
    patch_turn(
        app.world_mut(),
        turn,
        RequestPatch {
            preamble: Some("first".into()),
            active_tools: Some(vec!["a".into(), "b".into()]),
            additional_params: Some(serde_json::json!({"keep": 1, "replace": 1})),
            ..Default::default()
        },
    )
    .unwrap();
    patch_turn(
        app.world_mut(),
        turn,
        RequestPatch {
            preamble: Some("second".into()),
            active_tools: Some(vec!["b".into(), "c".into()]),
            additional_params: Some(serde_json::json!({"replace": 2})),
            ..Default::default()
        },
    )
    .unwrap();
    let merged = app.world().get::<RequestPatch>(turn).unwrap().clone();
    assert_eq!(merged.preamble.as_deref(), Some("second"));
    assert_eq!(merged.active_tools, Some(vec!["b".into()]));
    assert_eq!(
        merged.additional_params,
        Some(serde_json::json!({"keep": 1, "replace": 2}))
    );
    let invalid = Err(OperationError::InvalidPhase {
        entity: turn,
        operation: "patch",
    });
    app.world_mut().entity_mut(turn).remove::<Fresh>();
    assert_eq!(
        patch_turn(app.world_mut(), turn, RequestPatch::default()),
        invalid
    );
    assert_eq!(app.world().get::<RequestPatch>(turn), Some(&merged));
    app.world_mut().entity_mut(turn).insert(Fresh);
    cancel(app.world_mut(), run, "stop").unwrap();
    let after_cancel = app.world().get::<RequestPatch>(turn).cloned();
    assert_eq!(
        patch_turn(app.world_mut(), turn, RequestPatch::default()),
        invalid
    );
    assert_eq!(app.world().get::<RequestPatch>(turn), after_cancel.as_ref());
    app.world_mut().despawn(turn);
    assert_eq!(
        patch_turn(app.world_mut(), turn, RequestPatch::default()),
        Err(OperationError::MissingEntity(turn))
    );
}
