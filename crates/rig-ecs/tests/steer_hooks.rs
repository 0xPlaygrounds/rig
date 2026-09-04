//! Steering as systems (CONTRACT §9): the components a user writes and
//! what the library does with them, in this crate's own words — the
//! corpus's hook cells are the full proof (`corpus/world_hooks.rs`).
//!
//! | claim | test |
//! |---|---|
//! | `Cancelled` on a run before its first turn: no record, the reason is the failure's | `cancelled_at_start_dispatches_nothing_and_names_the_reason` |
//! | `Cancelled` in `Patch`: the folded completion is despawned before the bus | `cancelled_in_patch_leaves_no_record` |
//! | `RequestPatch` on the fresh turn is folded in: the preamble, a document, the history with the prompt kept | `a_request_patch_is_folded_into_the_turn` |
//! | `Retry { feedback }` on a text turn makes the turn and the feedback history, then another turn | `a_retry_with_feedback_asks_again` |
//! | `UsesModel` written on the run before `Select` routes the turn; `Route` is in the required row | `uses_model_written_before_select_routes_the_turn` |
//! | a decision written and not yet read survives a scene (§10 of the dissolves doc) | `a_retry_written_before_a_save_is_read_after_the_load` |

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::type_complexity
)]

mod run_support;

use bevy_ecs::prelude::*;
use rig_core::{
    effect::{EffectFamily, HandlerKey},
    error::ErrorKind,
    message::{AssistantContent, Message},
};
use rig_ecs::{
    agent::{
        Cancelled, Cursor, DocumentId, DocumentText, Failed, Failure, Order, RequestPatch, Retry,
        Route, RunResult, Settled, UsesModel,
        scene::{load_world, save_world},
    },
    bus::{EffectLogResource, Handlers, PendingEffect, RigSchedule},
    replay::required_row,
    systems::{Fresh, RigSet, spawn_run},
};
use rig_effect_log::EffectLogRecorder;
use run_support::*;

const MODEL: &str = "t/model:default";
const FAST: &str = "t/model:fast";

fn add_system<M>(
    app: &mut bevy_app::App,
    system: impl IntoScheduleConfigs<bevy_ecs::system::ScheduleSystem, M>,
) {
    app.world_mut()
        .resource_mut::<Schedules>()
        .add_systems(RigSchedule, system);
}

fn ended(app: &mut bevy_app::App, run: Entity, what: &str) {
    tick_until(app, what, |world| {
        world.get::<Settled>(run).is_some() || world.get::<Failed>(run).is_some()
    });
}

fn records(app: &bevy_app::App) -> Vec<String> {
    app.world()
        .resource::<EffectLogResource>()
        .log()
        .records
        .iter()
        .map(|record| record.key.to_string())
        .collect()
}

#[test]
fn cancelled_at_start_dispatches_nothing_and_names_the_reason() {
    let mut app = app();
    EffectLogResource::install(app.world_mut(), EffectLogRecorder::new());
    let (model, _) = Capturing::new(MODEL, "never");
    let model = register(&mut app, MODEL, model);
    let agent = spawn_agent(app.world_mut(), "t", model);
    let run = spawn_run(app.world_mut(), agent, &[], "go", false, None);
    app.world_mut()
        .entity_mut(run)
        .insert(Cancelled("stopped at run start".to_owned()));
    ended(&mut app, run, "cancelled");
    let Some(Failed(Failure::Cancelled(report))) = app.world().get::<Failed>(run) else {
        panic!("cancelled");
    };
    assert_eq!(report.kind, ErrorKind::Cancelled);
    assert_eq!(report.message, "stopped at run start");
    app.update();
    assert!(records(&app).is_empty(), "nothing was dispatched");
}

fn stop_in_patch(
    folded: Query<&ChildOf, Added<PendingEffect>>,
    turns: Query<&ChildOf>,
    mut commands: Commands,
) {
    for turn_of in &folded {
        if let Ok(run_of) = turns.get(turn_of.parent()) {
            commands
                .entity(run_of.parent())
                .insert(Cancelled("stopped before the completion call".to_owned()));
        }
    }
}

#[test]
fn cancelled_in_patch_leaves_no_record() {
    let mut app = app();
    EffectLogResource::install(app.world_mut(), EffectLogRecorder::new());
    let (model, requests) = Capturing::new(MODEL, "never");
    let model = register(&mut app, MODEL, model);
    let agent = spawn_agent(app.world_mut(), "t", model);
    add_system(&mut app, stop_in_patch.in_set(RigSet::Patch));
    let run = spawn_run(app.world_mut(), agent, &[], "go", false, None);
    ended(&mut app, run, "cancelled");
    app.update();
    assert!(
        records(&app).is_empty(),
        "the folded effect never reached the bus"
    );
    assert!(requests.lock().unwrap().is_empty());
    assert!(matches!(
        app.world().get::<Failed>(run),
        Some(Failed(Failure::Cancelled(report))) if report.message == "stopped before the completion call"
    ));
}

fn patch_the_turn(fresh: Query<Entity, Added<Fresh>>, mut commands: Commands) {
    for turn in &fresh {
        commands.entity(turn).insert(RequestPatch {
            preamble: Some("You are a pirate.".to_owned()),
            extra_context: vec![rig_core::completion::Document {
                id: "extra".to_owned(),
                text: "a glarb-glarb".to_owned(),
                additional_props: Default::default(),
            }],
            history: Some(
                [
                    Message::user("My name is Ada."),
                    Message::assistant("Hello, Ada."),
                ]
                .iter()
                .filter_map(rig_ecs::agent::MessageParts::from_message)
                .collect(),
            ),
            ..RequestPatch::default()
        });
    }
}

#[test]
fn a_request_patch_is_folded_into_the_turn() {
    let mut app = app();
    let (model, requests) = Capturing::new(MODEL, "Ada");
    let model = register(&mut app, MODEL, model);
    let agent = spawn_agent(app.world_mut(), "t", model);
    let document = app
        .world_mut()
        .spawn((
            DocumentId("static".to_owned()),
            DocumentText("always".to_owned()),
        ))
        .id();
    app.world_mut()
        .spawn((rig_ecs::agent::Context(document), Order(0), ChildOf(agent)));
    add_system(
        &mut app,
        patch_the_turn
            .after(RigSet::Advance)
            .before(RigSet::Assemble),
    );
    let run = spawn_run(app.world_mut(), agent, &[], "What is my name?", false, None);
    ended(&mut app, run, "answered");
    let requests = requests.lock().unwrap();
    assert_eq!(
        texts(&requests[0]),
        vec![
            "system:You are a pirate.",
            "user:My name is Ada.",
            "assistant:Hello, Ada.",
            "user:What is my name?"
        ],
        "the preamble replaced, the history replaced, the prompt kept"
    );
    let documents: Vec<&str> = requests[0]
        .documents
        .iter()
        .map(|d| d.id.as_str())
        .collect();
    assert_eq!(
        documents,
        ["static", "extra"],
        "the patch's document after the turn's"
    );
}

fn demand_done(
    turns: Query<
        (Entity, &rig_ecs::agent::Outputs),
        (Without<rig_ecs::systems::Materialised>, Without<Retry>),
    >,
    mut commands: Commands,
) {
    for (turn, outs) in &turns {
        if outs.done && !rig_ecs::policy::answer_text(&outs.content).contains("DONE") {
            commands.entity(turn).insert(Retry {
                feedback: Some("End with DONE.".to_owned()),
            });
        }
    }
}

#[test]
fn a_retry_with_feedback_asks_again() {
    let mut app = app();
    let (model, requests) = Scripted::new(
        MODEL,
        vec![
            vec![AssistantContent::text("first")],
            vec![AssistantContent::text("second DONE")],
        ],
    );
    let model = register(&mut app, MODEL, model);
    let agent = spawn_agent(app.world_mut(), "t", model);
    app.world_mut()
        .entity_mut(agent)
        .insert(rig_ecs::agent::MaxTurns(3));
    add_system(&mut app, demand_done.in_set(RigSet::Judge));
    let run = spawn_run(app.world_mut(), agent, &[], "say it", false, None);
    ended(&mut app, run, "answered");
    assert_eq!(
        app.world().get::<RunResult>(run).map(|r| r.0.as_str()),
        Some("second DONE")
    );
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(
        texts(&requests[1]),
        vec![
            "system:You are terse.",
            "user:say it",
            "assistant:first",
            "user:End with DONE."
        ]
    );
}

#[derive(Resource)]
struct Fast(Entity);

fn route_first_turn(
    fresh: Query<&ChildOf, Added<Fresh>>,
    runs: Query<&Cursor>,
    fast: Res<Fast>,
    mut commands: Commands,
) {
    for turn_of in &fresh {
        let run = turn_of.parent();
        if runs.get(run).is_ok_and(|cursor| cursor.turn == 1) {
            commands.entity(run).insert(UsesModel(fast.0));
        }
    }
}

#[test]
fn uses_model_written_before_select_routes_the_turn() {
    let mut app = app();
    EffectLogResource::install(app.world_mut(), EffectLogRecorder::new());
    let (model, _) = Capturing::new(MODEL, "slow");
    let model = register(&mut app, MODEL, model);
    let (fast, _) = Capturing::new(FAST, "fast");
    let fast = register(&mut app, FAST, fast);
    let agent = spawn_agent(app.world_mut(), "t", model);
    app.world_mut()
        .spawn((Route(fast), Order(0), ChildOf(agent)));
    app.insert_resource(Fast(fast));
    add_system(
        &mut app,
        route_first_turn
            .after(RigSet::Advance)
            .before(RigSet::Select),
    );
    let row = required_row(app.world_mut(), agent);
    assert_eq!(
        row.get(&HandlerKey::from(FAST)),
        Some(&EffectFamily::Completion)
    );
    let run = spawn_run(app.world_mut(), agent, &[], "go", false, None);
    ended(&mut app, run, "answered");
    assert_eq!(
        app.world().get::<RunResult>(run).map(|r| r.0.as_str()),
        Some("fast")
    );
    assert_eq!(records(&app), [FAST]);
}

#[test]
fn a_retry_written_before_a_save_is_read_after_the_load() {
    // The first world: the turn answered, a `Retry` written on it, the
    // world saved before `Materialise` read it.
    let mut first = app();
    let (model, _) = Scripted::new(MODEL, vec![vec![AssistantContent::text("first")]]);
    let model = register(&mut first, MODEL, model);
    let agent = spawn_agent(first.world_mut(), "t", model);
    first
        .world_mut()
        .entity_mut(agent)
        .insert(rig_ecs::agent::MaxTurns(3));
    let _run = spawn_run(first.world_mut(), agent, &[], "say it", false, None);
    tick_until(&mut first, "the turn answered", |world| {
        world
            .query_filtered::<&rig_ecs::agent::Outputs, With<rig_ecs::agent::Turn>>()
            .iter(world)
            .any(|outs| outs.done)
    });
    // Whether Materialise has read the turn is not the point — the point
    // is that the decision survives: written on the turn, saved, restored.
    let turn = first
        .world_mut()
        .query_filtered::<Entity, With<rig_ecs::agent::Turn>>()
        .iter(first.world())
        .next()
        .expect("a turn");
    first.world_mut().entity_mut(turn).insert(Retry {
        feedback: Some("End with DONE.".to_owned()),
    });
    let scene = save_world(first.world_mut()).expect("serializes");
    let json = serde_json::to_string(&scene).expect("serde");
    assert!(
        json.contains("End with DONE."),
        "the decision is scene state"
    );
    drop(first);

    let mut app = app();
    let (model, _) = Capturing::new(MODEL, "again");
    register(&mut app, MODEL, model);
    let loaded = load_world(
        &serde_json::from_str(&json).expect("serde"),
        app.world_mut(),
    )
    .expect("loads");
    let turn = loaded
        .graph
        .iter()
        .copied()
        .find(|entity| app.world().get::<rig_ecs::agent::Turn>(*entity).is_some())
        .expect("the turn");
    assert_eq!(
        app.world().get::<Retry>(turn),
        Some(&Retry {
            feedback: Some("End with DONE.".to_owned())
        }),
        "the decision restored intact"
    );
    let _ = Handlers::with(app.world_mut(), |handlers| handlers.keys());
}
