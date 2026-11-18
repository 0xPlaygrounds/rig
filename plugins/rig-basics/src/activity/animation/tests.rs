//! Needs the terminal view, the only front it animates.
#![cfg(feature = "tui")]

use std::time::Duration;

use rig_harness::prelude::*;

use super::*;
use crate::activity::ActivityPlugin;

/// An app with the terminal view's activity, `settings` as its
/// settings.toml, and a thinking shown agent; with the agent and its turn.
fn working(settings: &str) -> (App, Entity, Entity) {
    let mut app = App::new();
    app.insert_resource(rig_harness::settings::Settings::from_toml(settings))
        .add_plugins((
            rig_tui::TuiViewPlugin,
            ActivityPlugin,
            WorkingAnimationPlugin,
        ))
        .init_resource::<rig_harness::main_agent::ShownAgent>();
    animate(&mut app);
    let world = app.world_mut();
    let agent = spawn_root(world, ()).id();
    world
        .resource_mut::<rig_harness::main_agent::ShownAgent>()
        .0 = Some(agent);
    let turn = world.spawn(TurnOf(agent)).id();
    world.spawn((Partial::default(), CallOf(turn)));
    (app, agent, turn)
}

/// Steps Bevy's real clock by `after`, runs a frame, and returns `agent`'s
/// status item and how many [`KeepAwake`]s live.
fn step(app: &mut App, agent: Entity, after: Duration) -> (Option<String>, usize) {
    let world = app.world_mut();
    world
        .resource_mut::<Time<Real>>()
        .update_with_duration(after);
    app.update();
    let items = app.world().get::<rig_tui::StatusItems>(agent);
    let item = items.and_then(|items| items.iter().find(|item| item.name == "status"));
    let text = item.map(|item| item.text.clone());
    let world = app.world_mut();
    let awake = world.query::<&KeepAwake>().iter(world).count();
    (text, awake)
}

#[test]
fn a_working_agents_status_waves_while_the_loop_stays_awake() {
    let (mut app, agent, turn) = working("");
    let (first, awake) = step(&mut app, agent, Duration::ZERO);
    assert_eq!(awake, 1);
    let (second, awake) = step(&mut app, agent, STEP);
    assert_eq!(awake, 1, "one KeepAwake, not one a step");
    assert_ne!(first, second, "the wave moves a step");
    let label = |text: &Option<String>| text.as_deref().is_some_and(|t| t.ends_with(" thinking…"));
    assert!(label(&first) && label(&second), "{first:?} {second:?}");

    // A question stops it; work after the answer starts it again.
    let ask = Ask::yes_no("Keep it?");
    let ask = app.world_mut().spawn((ask, ForAgent(agent))).id();
    let (asking, awake) = step(&mut app, agent, STEP);
    assert_eq!((asking.as_deref(), awake), (Some("asking: Keep it?"), 0));
    app.world_mut().entity_mut(ask).despawn();
    let (_, awake) = step(&mut app, agent, STEP);
    assert_eq!(awake, 1);

    app.world_mut().entity_mut(turn).despawn();
    let (idle, awake) = step(&mut app, agent, STEP);
    assert_eq!((idle.as_deref(), awake), (Some("idle"), 0));
}

#[test]
fn with_animate_off_the_wave_is_still_and_the_loop_sleeps() {
    let (mut app, agent, _) = working("[activity]\nanimate = false\n");
    let (first, awake) = step(&mut app, agent, Duration::ZERO);
    let (second, _) = step(&mut app, agent, STEP);
    assert_eq!(first, second);
    assert_eq!(awake, 0);
    // Still a wave, before the label.
    assert!(first.is_some_and(|text| text.ends_with(" thinking…") && text != "thinking…"));
}

/// A run whose front is not the terminal view, such as `rig -p`, keeps
/// no animation awake.
#[test]
fn another_front_animates_nothing() {
    let mut app = App::new();
    app.insert_resource(rig_harness::settings::Settings::default())
        .add_plugins((
            rig_tui::TuiViewPlugin,
            ActivityPlugin,
            WorkingAnimationPlugin,
        ))
        .init_resource::<rig_harness::main_agent::ShownAgent>();
    app.finish();
    let world = app.world_mut();
    let agent = spawn_root(world, ()).id();
    world
        .resource_mut::<rig_harness::main_agent::ShownAgent>()
        .0 = Some(agent);
    world.spawn(TurnOf(agent));
    let (_, awake) = step(&mut app, agent, STEP);
    assert_eq!(awake, 0);
}

/// Whether the working animation runs in an app on the terminal view's
/// front with the plugins `add` adds, in their order.
fn animates(add: fn(&mut App)) -> bool {
    let mut app = App::new();
    let invoked = Invoked {
        terminal: true,
        ..Invoked::default()
    };
    app.insert_resource(rig_harness::settings::Settings::default())
        .insert_resource(invoked)
        .init_resource::<rig_harness::main_agent::ShownAgent>();
    add(&mut app);
    app.finish();
    app.world().contains_resource::<Pulse>()
}

#[test]
fn it_animates_with_the_activity_and_the_view_added_in_any_order() {
    assert!(animates(|app| {
        app.add_plugins((
            WorkingAnimationPlugin,
            ActivityPlugin,
            rig_tui::TuiViewPlugin,
        ));
    }));
    assert!(animates(|app| {
        app.add_plugins((
            rig_tui::TuiViewPlugin,
            ActivityPlugin,
            WorkingAnimationPlugin,
        ));
    }));
    assert!(!animates(|app| {
        app.add_plugins((WorkingAnimationPlugin, rig_tui::TuiViewPlugin));
    }));
    assert!(!animates(|app| {
        app.add_plugins((ActivityPlugin, WorkingAnimationPlugin));
    }));
}
