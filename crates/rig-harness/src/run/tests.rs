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

/// An app run by `front`, on a terminal or not, with `--tools tools` when
/// given, the tool `read` and a start held by the returned entity.
fn app(front: Front, terminal: bool, tools: Option<&str>) -> (App, Entity) {
    let mut app = App::new();
    let tools = tools.map(|tools| (TOOLS.to_owned(), Some(tools.to_owned())));
    app.insert_resource(Invoked {
        args: Invocation {
            flags: tools.into_iter().collect(),
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
    let (mut app, held) = app(terminal, true, Some("read,mcp__echo"));
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
    let (mut app, held) = app(Front::new("view").on_terminal(), true, Some("read,shel"));
    app.update();
    assert_eq!(app.should_exit(), None, "held: a tool may still come");
    app.world_mut().despawn(held);
    app.update();
    app.update();
    assert_eq!(app.should_exit(), Some(AppExit::from_code(2)));
    assert_eq!(app.world().resource::<Passes>().0, 0);
}

#[test]
fn a_terminal_run_without_tools_runs_while_its_start_is_held() {
    let (mut app, _held) = app(Front::new("view").on_terminal(), true, None);
    app.update();
    app.update();
    assert!(!app.world().resource::<RunStart>().is_over());
    assert!(app.world().resource::<Passes>().0 > 0);
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
fn the_start_makes_the_names_the_runs_tools_and_leaves_every_access() {
    let (mut app, held) = app(Front::new("view").on_terminal(), true, Some("read"));
    register(&mut app, "shell");
    let agent = app.world_mut().spawn(Agent).id();
    app.update();
    app.world_mut().despawn(held);
    app.update();
    let world = app.world();
    let run = world.get_resource::<RunTools>();
    assert_eq!(run, Some(&RunTools::new(["read"])));
    assert_eq!(world.get::<ToolAccess>(agent), Some(&ToolAccess::all()));
}
