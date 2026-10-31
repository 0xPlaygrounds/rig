use std::sync::Arc;

use bevy_app::TaskPoolPlugin;
use rig_core::completion::ToolDefinition;
use rig_core::message::ToolName;
use rig_ecs::AgentPlugin;
use rig_ecs::prelude::*;
use rig_harness_protocol::Invocation;

use super::*;
use crate::host::cli::{AppCliExt, Front};
use crate::host::exit::ExitPlugin;

/// How many frames ran the agent loop.
#[derive(Resource, Default)]
struct Passes(usize);

/// An app run by `front`, on a terminal or not, with `--tools tools`, the
/// tool `read` and a start held by the returned entity.
fn app(front: Front, terminal: bool, tools: &str) -> (App, Entity) {
    let mut app = App::new();
    app.insert_resource(Invoked {
        args: Invocation {
            flags: vec![(TOOLS.to_owned(), Some(tools.to_owned()))],
            ..Invocation::default()
        },
        terminal,
    })
    .add_plugins((
        TaskPoolPlugin::default(),
        AgentPlugin,
        ExitPlugin,
        RunPlugin,
    ))
    .add_front(front)
    .init_resource::<Passes>()
    .add_systems(
        Update,
        (|mut passes: ResMut<Passes>| passes.0 += 1).in_set(AgentLoop),
    );
    register(&mut app, "read");
    let held = app.world_mut().spawn(Starting).id();
    app.finish();
    app.cleanup();
    (app, held)
}

/// Registers the tool `name`, as a plugin whose tools come in the
/// background does.
fn register(app: &mut App, name: &str) {
    if let Ok(name) = ToolName::new(name) {
        let definition = ToolDefinition::new(name, "", serde_json::json!({}));
        app.world_mut().spawn(ToolDef(Arc::new(definition)));
    }
}

#[test]
fn a_terminal_run_waits_for_its_start_and_takes_a_tool_registered_meanwhile() {
    let terminal = Front::new("view").on_terminal();
    let (mut app, held) = app(terminal, true, "read,mcp__echo");
    app.update();
    app.update();
    assert!(!app.world().resource::<RunStart>().is_over());
    assert_eq!(
        app.world().resource::<Passes>().0,
        0,
        "no loop before the start"
    );
    register(&mut app, "mcp__echo");
    app.world_mut().despawn(held);
    app.update();
    assert!(app.world().resource::<RunStart>().is_over());
    assert_eq!(app.should_exit(), None);
    app.update();
    assert!(app.world().resource::<Passes>().0 > 0);
}

#[test]
fn a_name_no_tool_has_refuses_the_run_once_the_start_is_over() {
    let (mut app, held) = app(Front::new("view").on_terminal(), true, "read,shel");
    app.update();
    assert_eq!(app.should_exit(), None, "held: a tool may still come");
    app.world_mut().despawn(held);
    app.update();
    app.update();
    assert_eq!(app.should_exit(), Some(AppExit::from_code(2)));
    assert_eq!(app.world().resource::<Passes>().0, 0);
}

#[test]
fn a_run_without_a_session_checks_no_tools() {
    let router = Front::new("router")
        .on_command("router", "", "Routes")
        .sessionless();
    let mut app = App::new();
    app.insert_resource(Invoked {
        args: Invocation {
            command: Some("router".to_owned()),
            flags: vec![(TOOLS.to_owned(), Some("bogus".to_owned()))],
            ..Invocation::default()
        },
        terminal: false,
    })
    .add_plugins((
        TaskPoolPlugin::default(),
        AgentPlugin,
        ExitPlugin,
        RunPlugin,
    ))
    .add_front(router);
    app.finish();
    app.cleanup();
    app.update();
    app.update();
    assert!(app.world().resource::<RunStart>().is_over());
    assert_eq!(app.should_exit(), None);
}

#[test]
fn the_unknown_names_are_named_with_the_tools_they_likely_mean() {
    let tools = |names: &[&str]| Tools(names.iter().map(|name| (*name).to_owned()).collect());
    let registered = ["read", "shell", "edit"];
    assert_eq!(tools(&["read", "shell"]).unknown(registered), None);
    assert_eq!(
        tools(&["shel"]).unknown(registered).as_deref(),
        Some("--tools: unknown tool `shel`; did you mean `shell`?")
    );
    assert_eq!(
        tools(&["bogus", "shel"]).unknown(registered).as_deref(),
        Some("--tools: unknown tools `bogus`, `shel`; the tools are edit, read, shell")
    );
}

#[test]
fn agents_spawned_during_a_pass_are_restricted_before_dispatch() {
    let mut app = App::new();
    app.insert_resource(Tools(vec!["allowed".to_owned()]));
    app.add_systems(
        AgentPass,
        (
            (|mut commands: Commands| {
                commands.spawn((Agent, ToolAccess::All.narrow(["denied"])));
                commands.spawn(Agent);
            })
            .in_set(Stage::Commit),
            limit_tools
                .after(Stage::Commit)
                .in_set(PrepareSystems::Input),
        ),
    );
    app.world_mut().run_schedule(AgentPass);
    let world = app.world_mut();
    let accesses: Vec<_> = world
        .query::<&ToolAccess>()
        .iter(world)
        .map(|access| (access.allows("allowed"), access.allows("denied")))
        .collect();
    assert_eq!(accesses.len(), 2);
    assert!(accesses.contains(&(false, false)));
    assert!(accesses.contains(&(true, false)));
}
