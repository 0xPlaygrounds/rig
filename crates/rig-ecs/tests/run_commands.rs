//! Runs as bundles and commands: `RunCommands` on `Commands` and `World`,
//! `RunBundle` for a run assembled by hand, and `Ready` as the start.
//!
//! | claim | test |
//! |---|---|
//! | a run queued on `Commands` is reserved at once and exists — opened, `Ready`, its prompt an utterance — after the flush; the first pass advances it | `a_run_queued_on_commands_exists_after_the_flush_and_advances_on_the_next_pass` |
//! | a cancel queued behind the spawn lands on the run before any pass: it ends `Failed(Cancelled)` and no request is ever made | `a_cancel_queued_behind_the_spawn_ends_the_run_before_it_starts` |
//! | a queued despawn of a live run is refused by `RunDespawnRefused` on the run, and takes the run once it ended | `a_queued_despawn_is_refused_by_event_while_the_run_lives` |
//! | a run despawned while the queued spawn populates it (a host `Add<Run>` observer refusing it) is simply gone: no panic, no orphan utterance, no request | `a_reserved_run_despawned_before_it_was_populated_is_gone` |
//! | a second queued despawn of an id an earlier one already took is refused `NotARun` on the dead id, without panicking | `a_second_queued_despawn_of_a_gone_run_is_refused_as_not_a_run` |
//! | a scene with `ready` or `prompt` on a non-run is refused before the destination changes | `a_scene_refuses_ready_and_prompt_off_a_run_before_it_loads` |
//! | a run assembled by hand from `RunBundle` and `Prompt` is not read until `Ready`: no utterance, no phase, no request; `Ready` opens it, history first | `a_run_assembled_by_hand_starts_on_ready_with_its_history_first` |
//! | a `Ready` run saved before it opened loads with its `Prompt` and `Ready`, opens in the second world and answers | `a_ready_run_saved_before_it_opened_starts_after_the_load` |
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing
)]

use crate::run_support;

use bevy_ecs::prelude::*;
use rig_core::message::UserContent;
use rig_ecs::{
    agent::{
        Assembling, Failed, Failure, MessageParts, Prompt, Ready, Run, RunOf, Settled, Utterance,
        scene::{RunScene, SceneKind, load_world, save_world},
    },
    systems::{Fresh, RunBundle, RunBusy, RunCommands, RunDespawnRefused, spawn_utterance},
};
use run_support::*;

/// The utterances `ChildOf` `run`.
fn utterances(world: &mut World, run: Entity) -> usize {
    world
        .query_filtered::<&ChildOf, With<Utterance>>()
        .iter(world)
        .filter(|child_of| child_of.parent() == run)
        .count()
}

#[test]
fn a_run_queued_on_commands_exists_after_the_flush_and_advances_on_the_next_pass() {
    let mut app = app();
    let (model, requests) = Capturing::new("m", "hello");
    let model = register(&mut app, "t/model:m", model);
    let agent = spawn_agent(app.world_mut(), "t", model);

    let world = app.world_mut();
    let run = world.commands().spawn_run(agent, &[], "hi", false, None);
    assert!(
        world.get::<Run>(run).is_none(),
        "the id is reserved; the run is queued, not spawned"
    );
    world.flush();
    assert!(world.get::<Run>(run).is_some(), "spawned by the flush");
    assert!(world.get::<Ready>(run).is_some(), "ready by the flush");
    assert!(
        world.get::<Assembling>(run).is_some(),
        "opened by the flush: the command path opens the run it made"
    );
    assert!(
        world.get::<Prompt>(run).is_none(),
        "the prompt was consumed into an utterance"
    );
    assert_eq!(utterances(world, run), 1, "the prompt utterance");
    assert!(
        world
            .query_filtered::<(), With<Fresh>>()
            .iter(world)
            .next()
            .is_none(),
        "no pass has run: no turn yet"
    );

    tick_until(&mut app, "the run settles", |world| {
        world.get::<Settled>(run).is_some()
    });
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        texts(&requests[0]),
        vec!["system:You are terse.".to_owned(), "user:hi".to_owned()]
    );
}

#[test]
fn a_cancel_queued_behind_the_spawn_ends_the_run_before_it_starts() {
    let mut app = app();
    let (model, requests) = Capturing::new("m", "hello");
    let model = register(&mut app, "t/model:m", model);
    let agent = spawn_agent(app.world_mut(), "t", model);

    let world = app.world_mut();
    let mut commands = world.commands();
    let run = commands.spawn_run(agent, &[], "hi", false, None);
    commands.cancel_run(run, "changed my mind");
    world.flush();
    assert!(
        matches!(
            world.get::<Failed>(run).map(|failed| &failed.0),
            Some(Failure::Cancelled(report)) if report.message == "changed my mind"
        ),
        "cancelled by the flush: {:?}",
        world.get::<Failed>(run)
    );
    assert!(world.get::<Assembling>(run).is_none(), "the phase went");

    for _ in 0..3 {
        app.update();
    }
    assert!(
        requests.lock().unwrap().is_empty(),
        "a run cancelled before its first pass never asks the model"
    );
    assert!(
        app.world()
            .get::<Failed>(run)
            .is_some_and(|failed| matches!(failed.0, Failure::Cancelled(_))),
        "the ending stays"
    );
    app.world_mut().commands().despawn_run(run);
    app.world_mut().flush();
    assert!(
        app.world().get::<Run>(run).is_none(),
        "an ended run despawns through commands"
    );
}

#[derive(Resource, Default)]
struct Refusals(Vec<(Entity, RunBusy)>);

fn note_refusal(refused: On<RunDespawnRefused>, mut refusals: ResMut<Refusals>) {
    refusals
        .0
        .push((refused.event().entity, refused.event().reason));
}

#[test]
fn a_queued_despawn_is_refused_by_event_while_the_run_lives() {
    let mut app = app();
    app.world_mut().init_resource::<Refusals>();
    app.world_mut().add_observer(note_refusal);
    let (model, _) = Capturing::new("m", "hello");
    let model = register(&mut app, "t/model:m", model);
    let agent = spawn_agent(app.world_mut(), "t", model);

    let world = app.world_mut();
    let mut commands = world.commands();
    let run = commands.spawn_run(agent, &[], "hi", false, None);
    commands.despawn_run(run);
    world.flush();
    assert!(world.get::<Run>(run).is_some(), "the live run stays");
    assert_eq!(
        world.resource::<Refusals>().0,
        vec![(run, RunBusy::Unsettled)],
        "the refusal is an event on the run"
    );

    tick_until(&mut app, "the run settles", |world| {
        world.get::<Settled>(run).is_some()
    });
    app.world_mut().commands().despawn_run(run);
    app.world_mut().flush();
    assert!(app.world().get::<Run>(run).is_none(), "the ended run went");
    assert_eq!(
        app.world().resource::<Refusals>().0.len(),
        1,
        "no second refusal"
    );
    assert_eq!(
        app.world_mut().despawn_run(run),
        Err(RunBusy::NotARun),
        "the exclusive form reports by value"
    );
}

#[test]
fn a_reserved_run_despawned_before_it_was_populated_is_gone() {
    let mut app = app();
    let (model, requests) = Capturing::new("m", "hello");
    let model = register(&mut app, "t/model:m", model);
    let agent = spawn_agent(app.world_mut(), "t", model);
    let world = app.world_mut();
    let before = world.spawn_run(agent, &[], "first", false, None);

    // A host observer that refuses every run of `agent` as it appears:
    // the run is despawned while the queued spawn is populating it.
    world.add_observer(
        move |added: On<Add, Run>, runs: Query<&RunOf>, mut commands: Commands| {
            if runs.get(added.entity).is_ok_and(|run_of| run_of.0 == agent) {
                commands.entity(added.entity).despawn();
            }
        },
    );
    let history = [MessageParts::User {
        content: vec![UserContent::text("earlier")],
    }];
    let run = world
        .commands()
        .spawn_run(agent, &history, "hi", false, Some(3));
    world.flush();
    assert!(
        world.get_entity(run).is_err(),
        "the refused run is gone, and the spawn did not panic on it"
    );
    assert_eq!(
        world
            .query_filtered::<&ChildOf, With<Utterance>>()
            .iter(world)
            .filter(|child_of| world.get_entity(child_of.parent()).is_err())
            .count(),
        0,
        "no utterance was spawned under the gone run"
    );
    assert_eq!(
        world.query_filtered::<(), With<Run>>().iter(world).count(),
        1,
        "only the run spawned before the observer"
    );
    tick_until(&mut app, "the earlier run settles", |world| {
        world.get::<Settled>(before).is_some()
    });
    assert_eq!(
        requests.lock().unwrap().len(),
        1,
        "no request for the gone run"
    );
}

#[test]
fn a_second_queued_despawn_of_a_gone_run_is_refused_as_not_a_run() {
    let mut app = app();
    app.world_mut().init_resource::<Refusals>();
    app.world_mut().add_observer(note_refusal);
    let (model, _) = Capturing::new("m", "hello");
    let model = register(&mut app, "t/model:m", model);
    let agent = spawn_agent(app.world_mut(), "t", model);

    let run = app.world_mut().spawn_run(agent, &[], "hi", false, None);
    tick_until(&mut app, "the run settles", |world| {
        world.get::<Settled>(run).is_some()
    });
    let world = app.world_mut();
    let mut commands = world.commands();
    commands.despawn_run(run);
    commands.despawn_run(run);
    world.flush();
    assert!(
        world.get_entity(run).is_err(),
        "the first despawn took the run"
    );
    assert_eq!(
        world.resource::<Refusals>().0,
        vec![(run, RunBusy::NotARun)],
        "the second is refused on the dead id, and only the second"
    );
}

#[test]
fn a_scene_refuses_ready_and_prompt_off_a_run_before_it_loads() {
    let mut app = app();
    let (model, _) = Capturing::new("t/model:m", "hello");
    let model = register(&mut app, "t/model:m", model);
    let agent = spawn_agent(app.world_mut(), "t", model);
    let world = app.world_mut();
    let bundle = RunBundle::new(world, agent, false);
    world.spawn((bundle, Prompt::from("saved"), Ready));
    let scene = RunScene::save(world).expect("every component serializes");
    let run = scene
        .entities
        .iter()
        .find(|entity| entity.kind == SceneKind::Run)
        .expect("the run");
    let prompt = run
        .components
        .get("prompt")
        .expect("the unread prompt")
        .clone();
    let ready = run.components.get("ready").expect("ready").clone();

    for (name, value) in [("ready", ready), ("prompt", prompt)] {
        let mut misplaced = scene.clone();
        let agent = misplaced
            .entities
            .iter_mut()
            .find(|entity| entity.kind == SceneKind::Agent)
            .expect("the agent");
        agent.components.insert(name.into(), value);
        let count = app.world().entities().len();
        let error = misplaced.load(app.world_mut()).unwrap_err();
        assert!(
            error.message.contains(&format!("{name} is not on a run")),
            "{name}: {}",
            error.message
        );
        assert_eq!(
            app.world().entities().len(),
            count,
            "{name}: refused before the destination changed"
        );
    }
}

#[test]
fn a_run_assembled_by_hand_starts_on_ready_with_its_history_first() {
    let mut app = app();
    let (model, requests) = Capturing::new("m", "hello");
    let model = register(&mut app, "t/model:m", model);
    let agent = spawn_agent(app.world_mut(), "t", model);

    let world = app.world_mut();
    let bundle = RunBundle::new(world, agent, false);
    let run = world.spawn((bundle, Prompt::from("by hand"))).id();
    for _ in 0..3 {
        app.update();
    }
    let world = app.world_mut();
    assert!(
        world.get::<Prompt>(run).is_some(),
        "the prompt is not read before Ready"
    );
    assert_eq!(utterances(world, run), 0, "no utterance before Ready");
    assert!(
        world.get::<Assembling>(run).is_none(),
        "no phase before Ready"
    );
    assert!(
        requests.lock().unwrap().is_empty(),
        "nothing is assembled before Ready"
    );

    // The history goes in by hand, then the host's word.
    spawn_utterance(
        world,
        run,
        MessageParts::User {
            content: vec![UserContent::text("earlier")],
        },
    )
    .expect("a user utterance");
    world.entity_mut(run).insert(Ready);
    tick_until(&mut app, "the run settles", |world| {
        world.get::<Settled>(run).is_some()
    });
    let world = app.world_mut();
    assert!(
        world.get::<Prompt>(run).is_none(),
        "the prompt was consumed"
    );
    assert!(world.get::<Ready>(run).is_some(), "Ready stays on the run");
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        texts(&requests[0]),
        vec![
            "system:You are terse.".to_owned(),
            "user:earlier".to_owned(),
            "user:by hand".to_owned()
        ],
        "the history first, the prompt last"
    );
}

#[test]
fn a_ready_run_saved_before_it_opened_starts_after_the_load() {
    let mut app = app();
    let (model, _) = Capturing::new("t/model:m", "hello");
    let model = register(&mut app, "t/model:m", model);
    let agent = spawn_agent(app.world_mut(), "t", model);
    let world = app.world_mut();
    let bundle = RunBundle::new(world, agent, false);
    // Ready written by hand, saved before any pass opened the run.
    world.spawn((bundle, Prompt::from("saved"), Ready));
    let saved = save_world(world).expect("every component serializes");
    let json = serde_json::to_string(&saved).expect("serde");
    drop(app);

    let mut app = run_support::app();
    let (model, requests) = Capturing::new("t/model:m", "hello");
    register(&mut app, "t/model:m", model);
    let saved = serde_json::from_str(&json).expect("serde");
    let loaded = load_world(&saved, app.world_mut()).expect("the model is bound");
    let run = loaded
        .graph
        .iter()
        .copied()
        .find(|entity| app.world().get::<Run>(*entity).is_some())
        .expect("the run");
    let world = app.world_mut();
    assert!(
        world.get::<Ready>(run).is_some(),
        "Ready survives the scene"
    );
    assert!(world.get::<Prompt>(run).is_some(), "the unread prompt too");
    assert!(
        world.get::<Assembling>(run).is_none(),
        "loaded as saved: unopened"
    );
    tick_until(&mut app, "the loaded run settles", |world| {
        world.get::<Settled>(run).is_some()
    });
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        texts(&requests[0]),
        vec!["system:You are terse.".to_owned(), "user:saved".to_owned()]
    );
}
